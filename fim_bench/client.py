"""HTTP client of the agent server for the benchmark."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from fim_bench.models import ModelInfo, build_prompt


class CompletionClient:
    """Minimal OpenAI-compatible /v1/completions caller."""

    def __init__(self, port: int, info: ModelInfo, timeout_s: float = 300.0) -> None:
        self._url = f"http://127.0.0.1:{port}/v1/completions"
        self._info = info
        self._timeout_s = timeout_s

    def fim(self, prefix: str, suffix: str, max_tokens: int = 96, temperature: float = 0.0) -> tuple[str, float]:
        """Run one FIM task, return (text, wall_seconds)."""
        payload = {
            "model": self._info.name,
            "prompt": build_prompt(self._info, prefix, suffix),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "stop": list(self._info.style.stop),
        }
        request = urllib.request.Request(
            self._url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self._timeout_s) as resp:
            out = json.loads(resp.read())
        return out["choices"][0]["text"], time.perf_counter() - started


def get_model_ids(port: int, timeout_s: float = 5.0) -> list[str]:
    """The model ids the server on the port reports, or an empty list."""
    url = f"http://127.0.0.1:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            out = json.loads(resp.read())
        return [item["id"] for item in out.get("data", [])]
    except (urllib.error.URLError, json.JSONDecodeError, KeyError):
        return []


def wait_ready(port: int, timeout_s: float = 600.0, poll_s: float = 10.0) -> bool:
    """Poll /v1/models until the server answers, return whether it became ready."""
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5.0) as resp:
                if resp.status == 200:
                    return True
        except urllib.error.URLError:
            pass
        time.sleep(poll_s)
    return False
