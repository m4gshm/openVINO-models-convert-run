"""Exactness validation of sampled prompt-lookup decoding vs pure sampling (methodology of #4769).

Run twice: --mode draft (prompt lookup + PC + do_sample) and --mode nodraft (pure sampling + PC).
Collects N completions for seeds 1..N, then compares the empirical token distributions per position
(TV distance, like the TV(ref, sd) columns of #4769). Exact speculative sampling makes the two
distributions identical; a greedy fallback or broken rewind makes them diverge.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request

BASE_MODEL_ARGS = ["--model", "Qwen3.6-35B-A3B-int4-sym-g128-awq", "--detect_cycled_tool_call", "on",
                   "--pipe", "VLM", "--kv_cache_precision", "u4",
                   "--generate_config_file", ".config/generate_config_prompt_lookup_sampled.json",
                   "--scheduler_config_file", ".config/scheduler_config_no_dsf.json"]
PROMPT = "Write a short paragraph about why the sky is blue."
N_SAMPLES = 50
MAX_NEW_TOKENS = 160
PORT = 8892


def log(msg):
    print(msg, flush=True)


def collect(mode):
    args = [sys.executable, "agent.py", "--cache_cleanup_days", "0", *BASE_MODEL_ARGS, "--port", str(PORT)]
    if mode == "draft":
        args += ["--prompt_lookup", "on"]
    log(f"starting server ({mode})...")
    server = subprocess.Popen(args, cwd=".", stdout=open(f"logs/tv-{mode}-server.log", "w", encoding="utf-8"),
                              stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 1200
        while time.time() < deadline:
            if server.poll() is not None:
                sys.exit(f"server died rc={server.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=3) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(5)
        else:
            sys.exit("server not ready in 20 min")
        log("server ready")
        results = []
        for seed in range(1, N_SAMPLES + 1):
            body = json.dumps({"model": "x", "stream": False, "temperature": 0.8, "top_p": 0.95,
                               "seed": seed, "max_completion_tokens": MAX_NEW_TOKENS,
                               "messages": [{"role": "user", "content": PROMPT}]}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read())
            message = data["choices"][0]["message"]
            results.append((message.get("content") or "") + (message.get("reasoning_content") or ""))
            if seed % 10 == 0:
                log(f"  {seed}/{N_SAMPLES}")
        return results
    finally:
        server.kill()


def total_variation(samples_a, samples_b):
    """Mean TV distance between per-position character-gram empirical distributions.

    Compares the empirical distribution of the k-th 4-gram of the completion at each
    position across the two sample sets; identical generators give TV ~ 0.
    """
    def grams(text, pos):
        flat = "".join(text.split())
        return flat[pos * 4:(pos + 1) * 4]

    n_pos = 20
    tvs = []
    for pos in range(n_pos):
        dist_a, dist_b = {}, {}
        for s in samples_a:
            dist_a[grams(s, pos)] = dist_a.get(grams(s, pos), 0) + 1
        for s in samples_b:
            dist_b[grams(s, pos)] = dist_b.get(grams(s, pos), 0) + 1
        keys = set(dist_a) | set(dist_b)
        tv = 0.5 * sum(abs(dist_a.get(k, 0) / len(samples_a) - dist_b.get(k, 0) / len(samples_b)) for k in keys)
        tvs.append(tv)
    return tvs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["draft", "nodraft", "compare"], required=True)
    args = parser.parse_args()
    if args.mode == "compare":
        draft = json.load(open("logs/tv-draft.json", encoding="utf-8"))
        nodraft = json.load(open("logs/tv-nodraft.json", encoding="utf-8"))
        tvs = total_variation(draft, nodraft)
        mean_tv = sum(tvs) / len(tvs)
        log(f"per-position TV: {[round(t, 3) for t in tvs]}")
        log(f"mean TV(draft, nodraft) over {len(tvs)} positions: {mean_tv:.3f}")
        log("EXACTNESS: " + ("PASS (distributions match)" if mean_tv < 0.1 else
                             "INCONCLUSIVE/FAIL (distributions diverge; increase N before concluding)"))
    else:
        results = collect(args.mode)
        if len(set(r.strip() for r in results)) < 2:
            sys.exit("VALIDITY ERROR: all samples identical/empty - nothing to compare")
        with open(f"logs/tv-{args.mode}.json", "w", encoding="utf-8") as f:
            json.dump(results, f)
        log(f"saved {len(results)} samples to logs/tv-{args.mode}.json")
