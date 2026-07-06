from pydantic import BaseModel


class GenerateOpts(BaseModel):
    max_new_tokens: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    preprocess_prompt_by_parser: bool = True


def get_default_generate_opts():
    return GenerateOpts(
        max_new_tokens=4096
    )


class SchedulerOpts(BaseModel):
    max_num_batched_tokens: int | None = None
    cache_size: int | None = None
    cache_interval_multiplier: int | None = None
    max_num_seqs: int | None = None
    dynamic_split_fuse: bool = False
    enable_prefix_caching: bool = True


def get_default_scheduler_opts() -> SchedulerOpts:
    return SchedulerOpts(
        max_num_seqs=4,
        dynamic_split_fuse=False,
        enable_prefix_caching=True,
        cache_size=None,
        max_num_batched_tokens=512,
    )
