import time
from pathlib import Path

import openvino_genai
from openvino_genai import SchedulerConfig

MODEL_PATH = Path("../models/Qwen3.5-2B-int4-sym")
DEVICE = "GPU"
MAX_NEW_TOKENS = 200

PASSAGE = (
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! Sphinx of black quartz, judge my vow. "
    "The five boxing wizards jump quickly. Bright vixens jump; dozy fowl quack. "
    "Jackdaws love my big sphinx of quartz. Two driven jocks help fax my big quiz. "
    "Crazy Fredrick bought many very exquisite opal jewels."
)
USER_MSG = (
    "Repeat the following text verbatim, then write it again with each sentence on a new line, "
    "then list every sentence as a numbered bullet. Do not add any commentary.\n\n"
    f"TEXT:\n{PASSAGE}"
)


def build_config():
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = MAX_NEW_TOKENS
    config.do_sample = False  # greedy: deterministic oracle
    return config


def run_baseline(chat_history):
    """Non-speculative reference: plain continuous batching, no prompt lookup."""
    scheduler_config = new_scheduler_config()

    pipe = openvino_genai.VLMPipeline(
        MODEL_PATH, DEVICE, scheduler_config=scheduler_config,
    )
    config = build_config()

    start = time.perf_counter()
    res = pipe.generate(history=chat_history, generation_config=config)
    elapsed = time.perf_counter() - start
    del pipe
    return res.texts[0], elapsed


def new_scheduler_config() -> SchedulerConfig:
    # SchedulerConfig() defaults enable_prefix_caching = false (plan scope).
    scheduler_config = openvino_genai.SchedulerConfig()
    # scheduler_config.enable_prefix_caching = True
    assert scheduler_config.enable_prefix_caching is False
    return scheduler_config


def run_prompt_lookup(chat_history, num_assistant_tokens, max_ngram_size=3):
    """Hybrid LA verifier speculative path: prompt-lookup, prefix caching off."""
    scheduler_config = new_scheduler_config()

    pipe = openvino_genai.VLMPipeline(
        MODEL_PATH, DEVICE, prompt_lookup=True, scheduler_config=scheduler_config,
    )
    config = build_config()
    config.num_assistant_tokens = num_assistant_tokens
    config.max_ngram_size = max_ngram_size

    start = time.perf_counter()
    res = pipe.generate(history=chat_history, generation_config=config)
    elapsed = time.perf_counter() - start
    del pipe
    return res.texts[0], elapsed


def main():
    messages = [{"role": "user", "content": USER_MSG}]
    chat_history = openvino_genai.ChatHistory(messages)

    print("=== Baseline (non-speculative continuous batching) ===")
    text_base, t_base = run_baseline(chat_history)
    print(text_base)
    print(f"\nBaseline time: {t_base:.2f}s\n")

    # Sweep candidate counts to exercise all-accepted / partial / full-rejection
    # advance values in the live/scratch rollback.
    for n in (3, 5, 6, 7, 8, 9, 10):
        print(f"=== Prompt-lookup (num_assistant_tokens={n}) ===")
        text_pl, t_pl = run_prompt_lookup(chat_history, num_assistant_tokens=n)
        same = text_pl == text_base
        print(text_pl)
        print(f"\nnum_assistant_tokens: {n}")
        print(f"\nPrompt-lookup time: {t_pl:.2f}s")
        print(f"Output matches baseline (ORACLE): {same}")
        if t_pl > 0:
            print(f"Speedup vs baseline: {t_base / t_pl:.2f}x")
        if not same:
            print("[!] OUTPUT MISMATCH — LA live/scratch rollback is incorrect")
        print()


if __name__ == "__main__":
    main()
