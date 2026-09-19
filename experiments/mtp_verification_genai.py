import time
from pathlib import Path

import openvino_genai as ovg

MODEL_DIR = Path("../models/Qwen3.5-2B-int8-sym")
MAX_NEW_TOKENS = 30
NUM_ASSISTANT_TOKENS = 1

PROMPTS = [
    "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,",
    "Once upon a time there was a little girl named Goldilocks. She went for a walk in the forest. Pretty soon, she came upon a house. She knocked and",
    "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the",
]


def make_config():
    config = ovg.GenerationConfig()
    config.max_new_tokens = MAX_NEW_TOKENS
    config.num_assistant_tokens = NUM_ASSISTANT_TOKENS
    return config


def generate(pipe, prompt, config):
    t0 = time.perf_counter()
    result = pipe.generate([prompt], [config])[0]
    elapsed = time.perf_counter() - t0
    # ContinuousBatchingPipeline.generate on string prompts returns decoded text per sequence.
    text = result.m_generation_ids[0]
    return text, elapsed


def report_timing(tag, n_tokens, elapsed):
    ms_per_tok = 1000 * elapsed / n_tokens if n_tokens else float("nan")
    tok_per_s = n_tokens / elapsed if elapsed else float("nan")
    print(f"  {tag:9s} {elapsed:.3f}s for {n_tokens} tokens ({ms_per_tok:.1f} ms/tok, {tok_per_s:.1f} tok/s)")


def main():
    scheduler_config = ovg.SchedulerConfig()

    print("Loading pipelines...")
    device = "GPU"
    baseline_pipe = ovg.ContinuousBatchingPipeline(str(MODEL_DIR), scheduler_config, device, {})
    mtp_pipe = ovg.ContinuousBatchingPipeline(
        str(MODEL_DIR),
        scheduler_config,
        device,
        {"draft_model": ovg.draft_model(str(MODEL_DIR), device)},
    )
    tokenizer = baseline_pipe.get_tokenizer()

    matches = 0
    for prompt in PROMPTS:
        print(f"\n{'=' * 70}")
        print(f"Prompt: {repr(prompt)}")
        print(f"{'=' * 70}")

        baseline_text, baseline_time = generate(baseline_pipe, prompt, make_config())
        print(f"\nBaseline output: {baseline_text}")

        mtp_text, mtp_time = generate(mtp_pipe, prompt, make_config())
        print(f"MTP output:      {mtp_text}")

        match = baseline_text == mtp_text
        matches += int(match)
        print(f"\nOutputs match: {match}")

        n_baseline = len(tokenizer.encode(baseline_text).input_ids.data[0])
        n_mtp = len(tokenizer.encode(mtp_text).input_ids.data[0])
        print("\nTiming:")
        report_timing("Baseline:", n_baseline, baseline_time)
        report_timing("MTP:", n_mtp, mtp_time)
        if mtp_time:
            print(f"  Speedup:  {baseline_time / mtp_time:.2f}x")

    print(f"\n{'=' * 70}")
    print(f"Greedy-equivalence: {matches}/{len(PROMPTS)} prompts matched baseline")


if __name__ == "__main__":
    main()