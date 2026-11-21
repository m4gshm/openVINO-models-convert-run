"""End-to-end smoke helpers: model fetch, server bootstrap and HTTP calls.

Marked for opt-in e2e runs (`pytest -m e2e`); the plain unit run never imports
the server or touches the network except through the tests that declare it.
"""
# the server log handle outlives start(); pylint 4.0.6 ignores inline disables for R1732
# pylint: disable=consider-using-with
import contextlib
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Optional

import httpx

log = logging.getLogger(__name__)

# --- OVMS helpers ---

OVMS_BINARY_ENV = "OVMS_BINARY"
OVMS_MODEL_DIR_ENV = "OVMS_MODEL_DIR"
OVMS_HTTP_PORT_ENV = "OVMS_HTTP_PORT"
OVMS_GRPC_PORT_ENV = "OVMS_GRPC_PORT"
OVMS_READY_TIMEOUT_DEFAULT = 120.0
OVMS_HEALTH_INTERVAL_S = 2.0

HF_REPO_ID = "OpenVINO/Qwen3.5-2B-int4-ov"
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
MODEL_DIR_NAME = "Qwen3.5-2B-int4-ov"

# No official pip package ships the server binary, so the harness fetches the release zip.
OVMS_VERSION = os.environ.get("OVMS_VERSION", "2026.4.1")
OVMS_DOWNLOAD_DIR = REPO_ROOT / "bin" / f"ovms-{OVMS_VERSION}"
OVMS_RELEASE_URL = ("https://github.com/openvinotoolkit/model_server/releases/download/"
                    f"v{OVMS_VERSION}/ovms_windows_{OVMS_VERSION}_python_off.zip")

DEVICE_DEFAULT = "GPU"
READY_TIMEOUT_DEFAULT = 300.0
FETCH_SKIP_ENV = "OPENVINO_E2E_SKIP_FETCH"
DEVICE_ENV = "OPENVINO_E2E_DEVICE"
READY_TIMEOUT_ENV = "OPENVINO_E2E_READY_TIMEOUT_S"

# Server readiness probe and per-request timeouts.
HEALTH_INTERVAL_S = 2.0
REQUEST_TIMEOUT_S = 120.0

# Files whose presence marks a complete local model copy.
MODEL_REQUIRED_FILES = ("openvino_language_model.xml", "openvino_language_model.bin", "config.json")

# --- Expected e2e outcomes ---

EXPECTED_CHAT_PHRASE = "4"

CHAT_MESSAGES = [{"role": "user", "content": "What is 2+2? Answer with a single number."}]

# A deterministic fill-in-the-middle prompt: the model must return the middle line.
FIM_PREFIX = "def add(a, b):\n"
FIM_SUFFIX = "\n\nassert add(1, 2) == 3\n"
FIM_MIDDLE = "    return a + b"


class FetchFailure(Exception):
    """The model cannot be fetched: suites should skip, not fail."""


def model_dir() -> Path:
    """Local directory the reference model is expected to live in."""
    return MODELS_DIR / MODEL_DIR_NAME


def model_is_complete() -> bool:
    """Whether the local model copy carries the key OpenVINO IR files."""
    return model_dir().is_dir() and all((model_dir() / f).exists() for f in MODEL_REQUIRED_FILES)


def ensure_model() -> Path:
    """Make sure the reference model exists locally, reusing a complete copy.

    Raises FetchFailure when fetching is impossible (offline, skip requested),
    and callers turn that into a pytest skip.
    """
    if model_is_complete():
        log.info("model %s is already fetched", HF_REPO_ID)
        return model_dir()
    if os.environ.get(FETCH_SKIP_ENV) == "1":
        raise FetchFailure(f"{FETCH_SKIP_ENV}=1 and no complete model at {model_dir()}")
    try:
        # huggingface_hub is optional: the model fetch falls back to the hf CLI executable.
        from huggingface_hub import snapshot_download  # pylint: disable=import-outside-toplevel
    except ImportError:
        return _fetch_with_cli()
    log.info("downloading %s into %s", HF_REPO_ID, model_dir())
    try:
        snapshot_download(repo_id=HF_REPO_ID, local_dir=model_dir())
    except Exception as e:  # pylint: disable=broad-exception-caught  # any network failure skips the suite
        raise FetchFailure(f"fetch of {HF_REPO_ID} failed: {e}") from e
    if not model_is_complete():
        raise FetchFailure(f"download finished but key files are missing in {model_dir()}")
    return model_dir()


def _fetch_with_cli() -> Path:
    """Download the model with the hf/huggingface-cli executable (no python package)."""
    executable = "hf" if shutil.which("hf") else "huggingface-cli"
    command = [executable, "download", HF_REPO_ID, "--local-dir", str(model_dir())]
    log.info("downloading with CLI: %s", " ".join(command))
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as e:
        raise FetchFailure(f"neither huggingface_hub nor {executable} is available: {e}") from e
    if completed.returncode != 0:
        raise FetchFailure(f"{executable} download failed: {(completed.stderr or completed.stdout)[-800:]}")
    if not model_is_complete():
        raise FetchFailure(f"download finished but key files are missing in {model_dir()}")
    return model_dir()


def free_port() -> int:
    """Reserve a free localhost port, then release it for the server to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def ready_timeout_s() -> float:
    """Configured server startup budget (OPENVINO_E2E_READY_TIMEOUT_S, default 300 s)."""
    raw = os.environ.get(READY_TIMEOUT_ENV)
    return float(raw) if raw else READY_TIMEOUT_DEFAULT


def device() -> str:
    """Configured inference device (OPENVINO_E2E_DEVICE, default GPU)."""
    return os.environ.get(DEVICE_ENV, DEVICE_DEFAULT)


class ServerHarness:
    """One agent server subprocess serving the fetched model on a fixed port."""

    def __init__(self, server_model_dir: Path):
        self.model_dir = server_model_dir
        self.port = free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.log_path = REPO_ROOT / "logs" / f"e2e-{self.port}.log"
        self.process: Optional[subprocess.Popen] = None
        self._log_context = contextlib.ExitStack()

    def start(self) -> None:
        """Launch the server and wait until it accepts OpenAI-compatible requests."""
        self.log_path.parent.mkdir(exist_ok=True)
        # the file stays open for the whole server lifetime; closing it would drop early output
        log_file = self._log_context.enter_context(self.log_path.open("w", encoding="utf-8"))
        args = [str(Path(sys.executable).resolve()), "agent.py",
                "--model", str(self.model_dir), "--device", device(),
                "--port", str(self.port), "--skip_typecheck"]
        log.info("starting server: %s", " ".join(args))
        self.process = subprocess.Popen(args, stdout=log_file, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + ready_timeout_s()
        while time.monotonic() < deadline:
            if self._exited():
                raise RuntimeError(f"server exited during startup; log tail: {self.log_tail()}")
            try:
                with httpx.Client(timeout=5) as probe:
                    if probe.get(f"{self.base_url}/v1/models").status_code == 200:
                        log.info("server is ready on port %s", self.port)
                        return
            except httpx.HTTPError:
                pass
            time.sleep(HEALTH_INTERVAL_S)
        self.stop()
        raise RuntimeError(f"server did not become ready in {ready_timeout_s()}s; log tail: {self.log_tail()}")

    def _exited(self) -> bool:
        return self.process is None or self.process.poll() is not None

    def log_tail(self, lines: int = 40) -> str:
        """Trailing lines of the server log, for diagnosable failures."""
        if not self.log_path.exists():
            return "<no log file>"
        content = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:])

    def stop(self) -> None:
        """Terminate the server and its children, no matter what handlers throw."""
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                self.process.kill()  # kill() terminates the direct child on Windows too
            except OSError:
                pass
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                               capture_output=True, check=False)
            except OSError:
                pass
        self.process = None
        log.info("server on port %s stopped", self.port)


def client() -> httpx.Client:
    """HTTP client with the e2e-friendly request timeout."""
    return httpx.Client(timeout=REQUEST_TIMEOUT_S)


def chat_completion(stream: bool, base_url: str, messages: list[dict[str, str]] | None = None,
                    max_tokens: int = 200) -> tuple[dict[str, Any], str]:
    """Call /v1/chat/completions and return (payload, sentinel).

    For stream=false the payload is the response JSON and the sentinel is "".
    For stream=true the payload carries the parsed (content, finish_reason) and
    the sentinel is "done" when the data: [DONE] frame arrived.
    """
    body: dict[str, Any] = {"model": MODEL_DIR_NAME, "messages": CHAT_MESSAGES if messages is None else messages,
                            "stream": stream, "temperature": 0.0, "max_tokens": max_tokens}
    with client() as http:
        if not stream:
            response = http.post(f"{base_url}/v1/chat/completions", json=body)
            response.raise_for_status()
            return response.json(), ""
        with http.stream("POST", f"{base_url}/v1/chat/completions", json=body) as response:
            content, finish_reason, done = parse_sse_response(response.iter_lines())
        return {"content": content, "finish_reason": finish_reason}, "done" if done else ""


def parse_sse_response(lines: Any) -> tuple[str, Optional[str], bool]:
    """Accumulate OpenAI SSE frames: (delta text, finish reason, [DONE] sentinel seen)."""
    deltas: list[str] = []
    finish_reason: Optional[str] = None
    done = False
    for sse_frame in lines:
        raw = sse_frame.removeprefix("data: ")
        if raw == "[DONE]":
            done = True
            break
        if not raw.strip():
            continue
        chunk = json.loads(raw)
        choices = chunk.get("choices")
        if not choices:
            continue
        choice = choices[0]
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
        if choice["delta"].get("content"):
            deltas.append(choice["delta"]["content"])
    return "".join(deltas), finish_reason, done


def text_completion_fim(prefix: str, suffix: str, base_url: str,
                        max_tokens: int = 200,
                        fim: tuple[str, str, str, str] = (MODEL_DIR_NAME, "<|fim_prefix|>",
                                                          "<|fim_suffix|>", "<|fim_middle|>")) -> dict[str, Any]:
    """Fill-in-the-middle request on the legacy text-completions endpoint."""
    model, begin, end, middle = fim
    prompt = f"{begin}{prefix}{end}{suffix}{middle}"
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}
    with client() as http:
        response = http.post(f"{base_url}/v1/completions", json=body)
        response.raise_for_status()
        return response.json()


# --- OVMS harness ---


# The release ships ovms(.exe); some installs rename it to ovms_server(.exe).
OVMS_BINARY_NAMES = ("ovms", "ovms.exe", "ovms_server", "ovms_server.exe")


def ovms_binary_path() -> Optional[str]:
    """OVMS binary from OVMS_BINARY, PATH, a previous download, or a fresh download."""
    configured = os.environ.get(OVMS_BINARY_ENV)
    if configured:
        return configured
    for name in OVMS_BINARY_NAMES:
        found = shutil.which(name)
        if found:
            return found
    cached = _find_ovms_binary()
    if cached:
        return str(cached)
    return _download_ovms()


def _find_ovms_binary() -> Optional[Path]:
    """Already unpacked OVMS executable in the download directory, if any."""
    if not OVMS_DOWNLOAD_DIR.is_dir():
        return None
    for path in OVMS_DOWNLOAD_DIR.rglob("*"):
        if path.is_file() and path.name in OVMS_BINARY_NAMES:
            return path
    return None


def _download_ovms() -> Optional[str]:
    """Download and unpack the OVMS release once; None where no asset is automated."""
    # shortcut: only the Windows zip is fetched automatically; elsewhere use OVMS_BINARY or PATH
    if sys.platform != "win32":
        return None
    archive = OVMS_DOWNLOAD_DIR.parent / f"ovms_windows_{OVMS_VERSION}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading OVMS %s (~120 MB) into %s", OVMS_VERSION, archive)
    with urllib.request.urlopen(OVMS_RELEASE_URL) as response, archive.open("wb") as out:
        shutil.copyfileobj(response, out)
    with zipfile.ZipFile(archive) as package:
        package.extractall(OVMS_DOWNLOAD_DIR)
    binary = _find_ovms_binary()
    if binary is None:
        raise RuntimeError(f"OVMS executable is missing from {archive}")
    log.info("OVMS unpacked at %s", binary)
    return str(binary)


def ovms_http_port() -> int:
    """Pick a free localhost port for OVMS HTTP endpoint."""
    return free_port()


def ovms_grpc_port() -> int:
    """Pick a free localhost port for OVMS gRPC endpoint."""
    return free_port()


class OvmssHarness:
    """One OVMS subprocess serving the model on fixed HTTP/gRPC ports."""

    def __init__(self, model_path: Path, http_port: int | None = None,
                 grpc_port: int | None = None):
        self.model_path = model_path
        self.http_port = http_port or ovms_http_port()
        self.grpc_port = grpc_port or ovms_grpc_port()
        self.base_url = f"http://127.0.0.1:{self.http_port}"
        self.log_path = REPO_ROOT / "logs" / f"ovms-{self.http_port}.log"
        self.process: Optional[subprocess.Popen] = None
        self._log_context = contextlib.ExitStack()

    def start(self) -> None:
        """Launch OVMS and wait until it accepts HTTP requests."""
        if self.process is not None:
            return
        binary = ovms_binary_path()
        if binary is None:
            raise RuntimeError("OVMS binary not found in PATH; set OVMS_BINARY env var")
        self.log_path.parent.mkdir(exist_ok=True)
        log_file = self._log_context.enter_context(self.log_path.open("w", encoding="utf-8"))
        args = [binary, "--model_path", str(self.model_path),
                "--model_name", self.model_path.name,
                "--rest_port", str(self.http_port),
                "--port", str(self.grpc_port)]
        log.info("starting OVMS: %s", " ".join(args))
        self.process = subprocess.Popen(args, stdout=log_file, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + ovms_ready_timeout_s()
        while time.monotonic() < deadline:
            if self._exited():
                raise RuntimeError(f"OVMS exited during startup; log tail: {self.log_tail()}")
            try:
                with httpx.Client(timeout=5) as probe:
                    if probe.get(f"{self.base_url}/v2/health/ready").status_code == 200:
                        log.info("OVMS is ready on port %s", self.http_port)
                        return
            except httpx.HTTPError:
                pass
            time.sleep(OVMS_HEALTH_INTERVAL_S)
        self.stop()
        raise RuntimeError(f"OVMS did not become ready in {ovms_ready_timeout_s()}s; log tail: {self.log_tail()}")

    def _exited(self) -> bool:
        return self.process is None or self.process.poll() is not None

    def log_tail(self, lines: int = 40) -> str:
        """Trailing lines of the OVMS log, for diagnosable failures."""
        if not self.log_path.exists():
            return "<no log file>"
        content = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:])

    def stop(self) -> None:
        """Terminate the OVMS process and its children."""
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                self.process.kill()
            except OSError:
                pass
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                               capture_output=True, check=False)
            except OSError:
                pass
        self.process = None
        log.info("OVMS on port %s stopped", self.http_port)


def ovms_ready_timeout_s() -> float:
    """Configured OVMS startup budget (OVMS_READY_TIMEOUT_S, default 120 s)."""
    raw = os.environ.get("OVMS_READY_TIMEOUT_S")
    return float(raw) if raw else OVMS_READY_TIMEOUT_DEFAULT


# --- Proxy harness ---


PROXY_READY_TIMEOUT_DEFAULT = 10.0
PROXY_HEALTH_INTERVAL_S = 1.0


class ProxyServerHarness:
    """One agent subprocess in proxy mode, forwarding to a local OVMS instance."""

    def __init__(self, ovms_harness: "OvmssHarness"):
        self.ovms = ovms_harness
        self.port = free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.log_path = REPO_ROOT / "logs" / f"proxy-{self.port}.log"
        self.process: Optional[subprocess.Popen] = None
        self._log_context = contextlib.ExitStack()

    def start(self) -> None:
        """Launch the agent in proxy mode and wait until it accepts requests."""
        binary = ovms_binary_path()
        if binary is None:
            raise RuntimeError("OVMS binary not found; proxy mode requires OVMS")
        self.ovms.start()
        self.log_path.parent.mkdir(exist_ok=True)
        log_file = self._log_context.enter_context(self.log_path.open("w", encoding="utf-8"))
        args = [str(Path(sys.executable).resolve()), "agent.py",
                "--openai_base_url", f"{self.ovms.base_url}/v3",
                "--openai_api_key", "ovms-local",
                "--port", str(self.port), "--skip_typecheck"]
        log.info("starting proxy: %s", " ".join(args))
        self.process = subprocess.Popen(args, stdout=log_file, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + proxy_ready_timeout_s()
        while time.monotonic() < deadline:
            if self._exited():
                raise RuntimeError(f"proxy exited during startup; log tail: {self.log_tail()}")
            try:
                with httpx.Client(timeout=5) as probe:
                    if probe.get(f"{self.base_url}/v1/models").status_code == 200:
                        log.info("proxy is ready on port %s", self.port)
                        return
            except httpx.HTTPError:
                pass
            time.sleep(PROXY_HEALTH_INTERVAL_S)
        self.stop()
        raise RuntimeError(f"proxy did not become ready in {proxy_ready_timeout_s()}s; log tail: {self.log_tail()}")

    def _exited(self) -> bool:
        return self.process is None or self.process.poll() is not None

    def log_tail(self, lines: int = 40) -> str:
        """Trailing lines of the proxy log, for diagnosable failures."""
        if not self.log_path.exists():
            return "<no log file>"
        content = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:])

    def stop(self) -> None:
        """Terminate the proxy and OVMS processes."""
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.kill()
            except OSError:
                pass
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                               capture_output=True, check=False)
            except OSError:
                pass
            self.process = None
            log.info("proxy on port %s stopped", self.port)
        # Also stop OVMS if it's still running
        if self.ovms.process is not None:
            self.ovms.stop()


def proxy_ready_timeout_s() -> float:
    """Configured proxy startup budget (OPENAI_PROXY_READY_TIMEOUT_S, default 10 s)."""
    raw = os.environ.get("OPENAI_PROXY_READY_TIMEOUT_S")
    return float(raw) if raw else PROXY_READY_TIMEOUT_DEFAULT
