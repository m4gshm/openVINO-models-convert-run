"""Greedy baseline: same protocol as the sampled test, do_sample=false config."""
import json
import subprocess
import sys
import time
import urllib.request

PORT = 8893
PROMPT = "Write a short paragraph about why the sky is blue. Then write a haiku about mountains."
BASE = f"http://127.0.0.1:{PORT}"


def log(msg):
    print(msg, flush=True)


server = subprocess.Popen(
    [sys.executable, "agent.py", "--cache_cleanup_days", "0",
     "--model", "Qwen3.6-35B-A3B-int4-sym-g128-awq",
     "--detect_cycled_tool_call", "on", "--pipe", "VLM", "--kv_cache_precision", "u4",
     "--generate_config_file", ".config/generate_config_prompt_lookup.json",
     "--scheduler_config_file", ".config/scheduler_config_no_dsf.json",
     "--prompt_lookup", "on", "--port", str(PORT)],
    cwd=".", stdout=open("logs/greedy-baseline-server.log", "w", encoding="utf-8"),
    stderr=subprocess.STDOUT)
try:
    deadline = time.time() + 1200
    while time.time() < deadline:
        if server.poll() is not None:
            sys.exit(f"server died rc={server.returncode}")
        try:
            with urllib.request.urlopen(f"{BASE}/v1/models", timeout=3) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(5)
    else:
        sys.exit("server not ready")
    log("server ready")
    for i in range(3):
        body = json.dumps({"model": "x", "stream": False,
                           "messages": [{"role": "user", "content": PROMPT}]}).encode()
        req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
        dt = time.time() - t0
        ct = data.get("usage", {}).get("completion_tokens", 0)
        text = data["choices"][0]["message"]["content"]
        log(f"[{i}] {dt:.1f}s, {ct} tok, {ct / dt:.1f} tok/s, first80={text[:80]!r}")
finally:
    server.kill()
