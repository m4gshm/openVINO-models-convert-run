from typing import Any

from pydantic import BaseModel


class GenerateOpts(BaseModel):
    do_sample: bool = True

    max_new_tokens: int | None = None
    max_prompt_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    preprocess_prompt_by_parser: bool = True
    model_parameters: dict[str, Any] = None

    num_beams: int | None = None
    num_beam_groups: int | None = None
    diversity_penalty: float | None = None
    length_penalty: float | None = None
    no_repeat_ngram_size: float | None = None

    num_return_sequences: int | None = None

    num_assistant_tokens: int | None = None
    max_ngram_size: int | None = None


def get_default_generate_opts():
    return GenerateOpts(
        max_new_tokens=4096
    )


class SparseAttentionOpts(BaseModel):
    sparse_attention_mode: str = "TRISHAPE",
    num_last_dense_tokens_in_prefill: int = 256
    num_retained_start_tokens_in_cache: int = 12288
    num_retained_recent_tokens_in_cache: int = 512
    xattention_threshold: float = 0.65
    xattention_block_size: int = 128
    xattention_stride: int = 32


class SchedulerOpts(BaseModel):
    max_num_batched_tokens: int | None = None
    cache_size: int | None = None
    cache_interval_multiplier: int | None = None
    max_num_seqs: int | None = None
    dynamic_split_fuse: bool = False
    enable_prefix_caching: bool = True
    use_sparse_attention: bool = False
    sparse_attention_config: SparseAttentionOpts | None = None


def get_default_scheduler_opts() -> SchedulerOpts:
    return SchedulerOpts(
        max_num_seqs=4,
        dynamic_split_fuse=False,
        enable_prefix_caching=True,
        cache_size=None,
        max_num_batched_tokens=512,
    )
