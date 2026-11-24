"""edit_file simulation: output literally echoes/continues prompt content - the real agent case.

The prompt asks to apply a concrete change by returning the full modified file, so the
generated text must reproduce long spans of the prompt (as edit_file oldString/newString does).
Compares acceptance rate and tok/s: sampled+prompt-lookup vs sampled no-draft.
"""
import json
import subprocess
import sys
import time
import urllib.request

PORT = 8897
BASE = f"http://127.0.0.1:{PORT}"
SRC = open("agent/preprocess/tool_call.py", encoding="utf-8").read()

PROMPT = (
    "Here is the module:\n```python\n" + SRC + "\n```\n\n"
    "Return the full modified file with these changes applied:\n"
    "1. Rename every occurrence of `fix_tool_calls` to `repair_tool_calls`.\n"
    "2. Add a module docstring 'Tool call repair helpers.' right after the imports.\n"
    "3. In every log message replace 'tool call' with 'toolcall'.\n"
    "Output only the file content, nothing else."
)
CONFIGS = {
    "edit_pl_sampled": ["--generate_config_file", ".config/generate_config_unittest_sampled.json",
                        "--scheduler_config_file", ".config/scheduler_config_no_dsf.json", "--prompt_lookup", "on"],
    "edit_nodraft": ["--generate_config_file", ".config/generate_config_unittest_sampled.json",
                     "--scheduler_config_file", ".config/scheduler_config_no_dsf.json"],
    "edit_dflash2_greedy": ["--draft_model", "Qwen3.6-35B-A3B-dflash2-int4-sym-g128", "--dflash_mode", "on",
                            "--generate_config_file", ".config/generate_config_eagle3.json",
                            "--scheduler_config_file", ".config/scheduler_config.json"],
    "edit_mtp_sampled": ["--draft_model", "Qwen3.6-35B-A3B-int4-sym-g128-awq",
                         "--generate_config_file", ".config/generate_config_prompt_lookup_sampled.json",
                         "--scheduler_config_file", ".config/scheduler_config_no_dsf.json"],
    "edit_dflash2_sampled": ["--draft_model", "Qwen3.6-35B-A3B-dflash2-int4-sym-g128", "--dflash_mode", "on",
                             "--generate_config_file", ".config/generate_config_dflash2_sampled.json",
                             "--scheduler_config_file", ".config/scheduler_config.json"],
}


def log(msg):
    print(msg, flush=True)


def run_config(name, extra):
    server = subprocess.Popen(
        [sys.executable, "agent.py", "--cache_cleanup_days", "0",
         "--model", "Qwen3.6-35B-A3B-int4-sym-g128-awq",
         "--detect_cycled_tool_call", "on", "--pipe", "VLM", "--kv_cache_precision", "u4",
         *extra, "--port", str(PORT)],
        cwd=".", stdout=open(f"logs/{name}-server.log", "w", encoding="utf-8"),
        stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 1500
        while time.time() < deadline:
            if server.poll() is not None:
                sys.exit(f"{name}: died rc={server.returncode}")
            try:
                with urllib.request.urlopen(f"{BASE}/v1/models", timeout=3) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(5)
        else:
            sys.exit(f"{name}: not ready")
        for i in range(3):
            body = json.dumps({"model": "x", "stream": False, "temperature": 0.8, "top_p": 0.95,
                               "seed": 7, "max_completion_tokens": 1200,
                               "messages": [{"role": "user", "content": PROMPT}]}).encode()
            req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.loads(r.read())
            dt = time.time() - t0
            ct = data.get("usage", {}).get("completion_tokens", 0)
            text = data["choices"][0]["message"].get("content") or ""
            echo = ("repair_tool_calls" in text) and ("module docstring" not in text) and len(text) > 3000
            log(f"[{name}] {i}: {dt:.1f}s, {ct} tok, {ct / dt:.1f} tok/s, looks-edited={'Y' if echo else '?'}")
        accept = [l.split(':')[-1].strip() for l in open(f"logs/{name}-server.log", encoding='utf-8', errors='replace')
                  if 'acceptance rate' in l]
        log(f"[{name}] acceptance rates: {accept}")
    finally:
        server.kill()


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, extra in CONFIGS.items():
        if only and name != only:
            continue
        run_config(name, extra)
