"""Live validation of sampled MTP + prefix caching (Qwen3.6 linear-attention)."""
import json
import subprocess
import sys
import time
import urllib.request

PORT = 8895
BASE = f"http://127.0.0.1:{PORT}"
PROMPT = "Write a short paragraph about why the sky is blue. Then write a haiku about mountains."


def log(msg):
    print(msg, flush=True)


server = subprocess.Popen(
    [sys.executable, "agent.py", "--cache_cleanup_days", "0",
     "--model", "Qwen3.6-35B-A3B-int4-sym-g128-awq",
     "--draft_model", "Qwen3.6-35B-A3B-int4-sym-g128-awq",
     "--detect_cycled_tool_call", "on", "--pipe", "VLM", "--kv_cache_precision", "u4",
     "--generate_config_file", ".config/generate_config_prompt_lookup_sampled.json",
     "--scheduler_config_file", ".config/scheduler_config_no_dsf.json",
     "--port", str(PORT)],
    cwd=".", stdout=open("logs/mtp-test-server.log", "w", encoding="utf-8"),
    stderr=subprocess.STDOUT)
try:
    deadline = time.time() + 1500
    while time.time() < deadline:
        if server.poll() is not None:
            log("SERVER DIED during startup, rc=%s" % server.returncode)
            sys.exit(2)
        try:
            with urllib.request.urlopen(f"{BASE}/v1/models", timeout=3) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(5)
    else:
        log("SERVER NOT READY in 25 min")
        sys.exit(2)
    log("server ready, sending requests...")
    texts = []
    for i, seed in enumerate((0, 1, 2)):
        body = json.dumps({"model": "x", "stream": False, "temperature": 0.8, "top_p": 0.95,
                           "seed": seed,
                           "messages": [{"role": "user", "content": PROMPT}]}).encode()
        req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
        dt = time.time() - t0
        ct = data.get("usage", {}).get("completion_tokens", 0)
        text = (data["choices"][0]["message"].get("content") or "") + \
               (data["choices"][0]["message"].get("reasoning_content") or "")
        texts.append(text)
        log(f"[{i}] {dt:.1f}s, {ct} tok, {ct / dt:.1f} tok/s")
    for i, t in enumerate(texts):
        log(f"--- output {i} (first 200 chars) ---")
        log(t[:200].replace("\n", " | "))
    distinct = len(set(t.strip() for t in texts))
    log(f"DISTINCT OUTPUTS: {distinct} of 3")
    log(f"RESULT: {'PASS' if distinct > 1 else 'FAIL'}")
finally:
    server.kill()
