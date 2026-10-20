"""Translation of the scheduler options into an openvino_genai SchedulerConfig."""
import logging

from openvino_genai import SchedulerConfig, SparseAttentionConfig, SparseAttentionMode

from agent.openai import get_default_scheduler_opts, SchedulerOpts

log = logging.getLogger(__name__)

DEFAULT_MAX_BATCHED_TOKENS = 8192


def build_scheduler_config(scheduler_opts: SchedulerOpts,
                          max_prompt_len: int | None) -> SchedulerConfig:
    """Build the pipeline scheduler config out of the options of the server.

    The options of the config file win over the built-in defaults; a value that
    is set in neither keeps the default of the scheduler itself.
    """
    default_opts = get_default_scheduler_opts()
    scheduler_config = SchedulerConfig()
    _apply_dynamic_split_fuse(scheduler_config, scheduler_opts, default_opts, max_prompt_len)
    _apply_cache_options(scheduler_config, scheduler_opts, default_opts)
    _apply_batching_options(scheduler_config, scheduler_opts, default_opts)
    _apply_sparse_attention(scheduler_config, scheduler_opts, default_opts)
    prefix_caching = _first_set(scheduler_opts.enable_prefix_caching,
                                default_opts.enable_prefix_caching)
    if prefix_caching:
        scheduler_config.enable_prefix_caching = prefix_caching
    scheduler_config.use_cache_eviction = False
    return scheduler_config


def _apply_dynamic_split_fuse(scheduler_config: SchedulerConfig, scheduler_opts: SchedulerOpts,
                              default_opts: SchedulerOpts, max_prompt_len: int | None) -> None:
    """Apply the dynamic split fuse switch and the batched token limit it implies."""
    dynamic_split_fuse = _first_set(scheduler_opts.dynamic_split_fuse,
                                    default_opts.dynamic_split_fuse)
    if dynamic_split_fuse is None:
        return
    scheduler_config.dynamic_split_fuse = dynamic_split_fuse
    if dynamic_split_fuse:
        num_batched_tokens = (_first_set(scheduler_opts.max_num_batched_tokens,
                                          default_opts.max_num_batched_tokens) or DEFAULT_MAX_BATCHED_TOKENS)
        scheduler_config.max_num_batched_tokens = num_batched_tokens
    else:
        scheduler_config.max_num_batched_tokens = max_prompt_len


def _apply_cache_options(scheduler_config: SchedulerConfig, scheduler_opts: SchedulerOpts,
                         default_opts: SchedulerOpts) -> None:
    """Apply the cache size and the cache interval multiplier."""
    cache_size = _first_set(scheduler_opts.cache_size, default_opts.cache_size)
    if cache_size:
        scheduler_config.cache_size = cache_size
    cache_interval_multiplier = _first_set(scheduler_opts.cache_interval_multiplier,
                                           default_opts.cache_interval_multiplier)
    if cache_interval_multiplier:
        scheduler_config.cache_interval_multiplier = cache_interval_multiplier


def _apply_batching_options(scheduler_config: SchedulerConfig, scheduler_opts: SchedulerOpts,
                            default_opts: SchedulerOpts) -> None:
    """Apply the sequence and the linear attention block limits."""
    max_num_seqs = _first_set(scheduler_opts.max_num_seqs, default_opts.max_num_seqs)
    if max_num_seqs:
        scheduler_config.max_num_seqs = max_num_seqs
    num_linear_attention_blocks = _first_set(scheduler_opts.num_linear_attention_blocks,
                                             default_opts.num_linear_attention_blocks)
    if num_linear_attention_blocks:
        scheduler_config.num_linear_attention_blocks = num_linear_attention_blocks


def _apply_sparse_attention(scheduler_config: SchedulerConfig, scheduler_opts: SchedulerOpts,
                            default_opts: SchedulerOpts) -> None:
    """Apply the sparse attention switch and its parameters."""
    use_sparse_attention = _first_set(scheduler_opts.use_sparse_attention,
                                      default_opts.use_sparse_attention)
    if use_sparse_attention is None:
        return
    scheduler_config.use_sparse_attention = use_sparse_attention
    attention_opts = scheduler_opts.sparse_attention_config
    if not use_sparse_attention or not attention_opts:
        return
    sparse_attention_config = SparseAttentionConfig()
    sparse_attention_config.mode = (SparseAttentionMode.TRISHAPE
                                    if attention_opts.sparse_attention_mode == "TRISHAPE"
                                    else SparseAttentionMode.XATTENTION)
    sparse_attention_config.num_last_dense_tokens_in_prefill = attention_opts.num_last_dense_tokens_in_prefill
    sparse_attention_config.num_retained_start_tokens_in_cache = attention_opts.num_retained_start_tokens_in_cache
    sparse_attention_config.num_retained_recent_tokens_in_cache = attention_opts.num_retained_recent_tokens_in_cache
    sparse_attention_config.xattention_threshold = attention_opts.xattention_threshold
    sparse_attention_config.xattention_block_size = attention_opts.xattention_block_size
    sparse_attention_config.xattention_stride = attention_opts.xattention_stride
    scheduler_config.sparse_attention_config = sparse_attention_config


def _first_set(value, default):
    """Return the value when it is set, otherwise the default one.

    ``False`` and ``0`` count as set: the options of the config file must be
    able to switch a feature off, not only on.
    """
    return value if value is not None else default
