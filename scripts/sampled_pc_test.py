"""Live validation of sampled prompt-lookup speculative decoding (PC on, do_sample=true)."""
import json
import subprocess
import sys
import time
import urllib.request

PORT = 8891
BASE = f"http://127.0.0.1:{PORT}"
PROMPT = "Write a short paragraph about why the sky is blue. Then write a haiku about mountains."

def log(msg):
    print(msg, flush=True)

server = subprocess.Popen(
    [sys.executable, "agent.py", "--cache_cleanup_days", "0",
     "--model", "Qwen3.6-35B-A3B-int4-sym-g128-awq",
     "--detect_cycled_tool_call", "on", "--pipe", "VLM", "--kv_cache_precision", "u4",
     "--generate_config_file", ".config/generate_config_prompt_lookup_sampled.json",
     "--scheduler_config_file", ".config/scheduler_config_no_dsf.json",
     "--prompt_lookup", "on", "--port", str(PORT)],
    cwd=".", stdout=open("logs/sampled-test-server.log", "w", encoding="utf-8"),
    stderr=subprocess.STDOUT)
try:
    # wait for readiness (model load takes minutes)
    deadline = time.time() + 900
    ready = False
    while time.time() < deadline:
        if server.poll() is not None:
            log("SERVER DIED during startup, rc=%s" % server.returncode)
            sys.exit(2)
        try:
            with urllib.request.urlopen(f"{BASE}/v1/models", timeout=3) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(5)
    if not ready:
        log("SERVER NOT READY in 15 min")
        sys.exit(2)
    log("server ready, sending requests...")

    def ask(tag, seed):
        body = json.dumps({"model": "x", "stream": False, "temperature": 0.8, "top_p": 0.95,
                           "seed": seed,
                           "messages": [{"role": "user", "content": PROMPT}]}).encode()
        req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
        dt = time.time() - t0
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        ct = usage.get("completion_tokens", 0)
        log(f"[{tag}] {dt:.1f}s, {ct} tok, {ct / dt:.1f} tok/s")
        return text

    texts = [ask(i, seed) for i, seed in enumerate((0, 1, 2))]
    for i, t in enumerate(texts):
        log(f"--- output {i} (first 300 chars) ---")
        log(t[:300].replace("\n", " | "))
    distinct = len(set(t.strip() for t in texts))
    log(f"DISTINCT OUTPUTS: {distinct} of 3 (different seeds: >1 means sampling actually samples)")
    log(f"RESULT: {'PASS' if distinct > 1 else 'FAIL (outputs identical - sampling did not apply)'}")
finally:
    server.kill()
