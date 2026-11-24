"""Unit-test-writing benchmark: the case where prompt lookup should shine.

The model writes pytest tests for agent/preprocess/tool_call.py whose full source is
embedded in the prompt - the output naturally echoes long spans of the prompt
(function names, fixtures, assert lines), which is exactly what the n-gram draft hits.

Three server configurations, same 3 task prompts, fixed seed per config:
  pl_sampled : sampled + prompt lookup (the combination under test)
  nodraft    : sampled, no draft (baseline speed)
  pl_greedy  : greedy + prompt lookup (the old default)
"""
import json
import subprocess
import sys
import time
import urllib.request

PORT = 8896
BASE = f"http://127.0.0.1:{PORT}"
SRC = open("agent/preprocess/tool_call.py", encoding="utf-8").read()
TASKS = [
    "Write comprehensive pytest unit tests for the happy paths of the functions in this module.",
    "Write pytest unit tests focused on edge cases and invalid inputs of this module.",
    "Write pytest unit tests for the error handling and retry behavior of this module.",
]
CONFIGS = {
    "pl_sampled": ["--generate_config_file", ".config/generate_config_unittest_sampled.json",
                   "--scheduler_config_file", ".config/scheduler_config_no_dsf.json", "--prompt_lookup", "on"],
    "nodraft": ["--generate_config_file", ".config/generate_config_unittest_sampled.json",
                "--scheduler_config_file", ".config/scheduler_config_no_dsf.json"],
    "pl_greedy": ["--generate_config_file", ".config/generate_config_unittest_greedy.json",
                  "--scheduler_config_file", ".config/scheduler_config_no_dsf.json", "--prompt_lookup", "on"],
}
MAX_NEW_TOKENS = 600


def log(msg):
    print(msg, flush=True)


def run_config(name, extra_args):
    server = subprocess.Popen(
        [sys.executable, "agent.py", "--cache_cleanup_days", "0",
         "--model", "Qwen3.6-35B-A3B-int4-sym-g128-awq",
         "--detect_cycled_tool_call", "on", "--pipe", "VLM", "--kv_cache_precision", "u4",
         *extra_args, "--port", str(PORT)],
        cwd=".", stdout=open(f"logs/bench-{name}-server.log", "w", encoding="utf-8"),
        stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 1500
        while time.time() < deadline:
            if server.poll() is not None:
                sys.exit(f"{name}: server died rc={server.returncode}")
            try:
                with urllib.request.urlopen(f"{BASE}/v1/models", timeout=3) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(5)
        else:
            sys.exit(f"{name}: server not ready")
        log(f"[{name}] ready")
        stats = []
        for i, task in enumerate(TASKS):
            prompt = (f"Here is the module under test:\n```python\n{SRC}\n```\n\n{task}\n"
                      "Output only the test file content, no explanations.")
            body = json.dumps({"model": "x", "stream": False, "temperature": 0.8, "top_p": 0.95,
                               "seed": 7, "max_completion_tokens": MAX_NEW_TOKENS,
                               "messages": [{"role": "user", "content": prompt}]}).encode()
            req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.loads(r.read())
            dt = time.time() - t0
            msg = data["choices"][0]["message"]
            text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
            ct = data.get("usage", {}).get("completion_tokens", 0)
            has_code = "def test_" in (msg.get("content") or "")
            stats.append((dt, ct, ct / dt, has_code))
            log(f"[{name}] task {i}: {dt:.1f}s, {ct} tok, {ct / dt:.1f} tok/s, code={'Y' if has_code else 'N'}")
        # skip the first task (warmup) in the summary
        steady = stats[1:]
        mean_tps = sum(s[2] for s in steady) / len(steady)
        log(f"[{name}] MEAN (steady): {mean_tps:.1f} tok/s, all produced code: {all(s[3] for s in stats)}")
        return mean_tps
    finally:
        server.kill()


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    results = {}
    for name, extra in CONFIGS.items():
        if only and name != only:
            continue
        results[name] = run_config(name, extra)
    for name, tps in results.items():
        log(f"TOTAL: {name}: {tps:.1f} tok/s")
