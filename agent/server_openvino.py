import json
import logging.config
import multiprocessing
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from openvino_genai import SchedulerConfig, SparseAttentionConfig, SparseAttentionMode, py_openvino_genai

from agent.__main__ import log, DeviceType, Pipe, ParserType, Turn, default_batch_size, \
    enum_value, AttentionBackend, stop_signal
from agent.common.log import logging_config
from agent.common.metric_mem import get_current_memory
from agent.inference.token_handler import TokenHandlerConfig
from agent.openai import get_default_generate_opts, GenerateOpts, get_default_scheduler_opts, SchedulerOpts
from agent.openai.engine_rest_cb import ContinuousBatchingController
from agent.openai.engine_rest_common import ControllerConfig
from agent.openai.engine_rest_vlm import VlmController
from agent.parser import Parser
from agent.parser.gemma4 import Gemma4ChannelParser
from agent.parser.lfm2 import Lfm2Parser
from agent.parser.qwen2 import Qwen2Parser
from agent.parser.qwen3 import Qwen3MoeParser
from agent.server import log, new_app


def run_openvino(args):
    """Run application in OpenVINO mode with local model loading."""
    model = args.model

    model_path = Path(model)
    if model_path.is_absolute():
        if model_path.is_file():
            # remove gguf ext
            model_name = model_path.with_suffix("").name
        else:
            model_name = model_path.name
    else:
        model_name = model
        model_path = Path(f"{args.models_dir}/{model}")

    model_cache_dir = f"{args.models_cache_dir}/{model_name}"
    logs_dir = f"./logs/{model_name}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"
    base_log_config = logging_config(logs_dir)
    log.info(f"logs dir {logs_dir}")

    logging.config.dictConfig(base_log_config)

    log.info("server starting")

    model_architectures: set[str] = set()
    max_position_embeddings: int | None = None

    if model_path.is_dir():
        openvino_model_config_json = model_path / "config.json"
        if openvino_model_config_json.is_file():
            try:
                config = json.loads(openvino_model_config_json.read_text(encoding="utf-8"))
                arch = config.get("architectures")
                if isinstance(arch, list):
                    model_architectures = set(arch)

                text_config = config.get("text_config")
                if isinstance(text_config, dict):
                    max_position_embeddings = text_config.get("max_position_embeddings")
                else:
                    # old models like Qwen2.5
                    max_position_embeddings = config.get("max_position_embeddings")
            except Exception as e:
                log.error(f"error on read {openvino_model_config_json}: {e}")
    elif not model_path.exists():
        log.error(f"file or directory doesn't exists: '{model_path}'")
        sys.exit(1)

    default_generate_opts = get_default_generate_opts()
    generate_opts_file = args.generate_config_file
    generate_opts: GenerateOpts
    if generate_opts_file:
        log.info(f"load {generate_opts_file}")
        try:
            with open(generate_opts_file, "r", encoding="utf-8") as file:
                generate_opts: GenerateOpts = GenerateOpts.model_validate_json(file.read())
        except FileNotFoundError as e:
            log.error(f"{e}")
            raise e
    else:
        generate_opts = default_generate_opts

    default_scheduler_opts = get_default_scheduler_opts()
    scheduler_opts_file = args.scheduler_config_file
    scheduler_opts: SchedulerOpts
    if scheduler_opts_file:
        log.info(f"load {scheduler_opts_file}")
        try:
            with open(scheduler_opts_file, "r", encoding="utf-8") as file:
                scheduler_opts = SchedulerOpts.model_validate_json(file.read())
        except FileNotFoundError as e:
            log.error(f"{e}")
            raise e
    else:
        scheduler_opts = default_scheduler_opts

    chat_template = ''
    chat_template_file = args.chat_template_file
    if chat_template_file:
        log.info(f"load {chat_template_file}")
        try:
            with open(chat_template_file, "r", encoding="utf-8") as file:
                chat_template = file.read()
        except FileNotFoundError as e:
            log.error(f"{e}")
            raise e

    max_prompt_len = args.max_prompt_len
    if not max_prompt_len:
        max_prompt_len = generate_opts.max_prompt_tokens or default_generate_opts.max_prompt_tokens
    device: DeviceType = DeviceType[args.device]
    is_device_npu = device == DeviceType.NPU
    if not max_prompt_len:
        max_prompt_len = max_position_embeddings

    generate_opts.max_prompt_tokens = max_prompt_len

    scheduler_config = SchedulerConfig()
    use_sparse_attention = scheduler_opts.use_sparse_attention or default_scheduler_opts.use_sparse_attention
    dynamic_split_fuse = scheduler_opts.dynamic_split_fuse or default_scheduler_opts.dynamic_split_fuse
    num_batched_tokens = scheduler_opts.max_num_batched_tokens or default_scheduler_opts.max_num_batched_tokens
    if dynamic_split_fuse and num_batched_tokens:
        scheduler_config.max_num_batched_tokens = num_batched_tokens
    else:
        scheduler_config.max_num_batched_tokens = max_prompt_len
    opts_cache_size = scheduler_opts.cache_size or default_scheduler_opts.cache_size
    if opts_cache_size:
        scheduler_config.cache_size = opts_cache_size
    cache_interval_multiplier = scheduler_opts.cache_interval_multiplier or default_scheduler_opts.cache_interval_multiplier
    if cache_interval_multiplier:
        scheduler_config.cache_interval_multiplier = cache_interval_multiplier
    opts_max_num_seqs = scheduler_opts.max_num_seqs or default_scheduler_opts.max_num_seqs
    if opts_max_num_seqs:
        scheduler_config.max_num_seqs = opts_max_num_seqs
    scheduler_config.dynamic_split_fuse = dynamic_split_fuse
    # scheduler_config.num_kv_blocks = 2048
    # scheduler_config.num_linear_attention_blocks = 256
    scheduler_config.use_sparse_attention = use_sparse_attention
    attention_opts = scheduler_opts.sparse_attention_config
    if use_sparse_attention and attention_opts:
        sparse_attention_config = SparseAttentionConfig()
        sparse_attention_config.mode = SparseAttentionMode.TRISHAPE if attention_opts.sparse_attention_mode == "TRISHAPE" else SparseAttentionMode.XATTENTION
        sparse_attention_config.num_last_dense_tokens_in_prefill = attention_opts.num_last_dense_tokens_in_prefill
        sparse_attention_config.num_retained_start_tokens_in_cache = attention_opts.num_retained_start_tokens_in_cache
        sparse_attention_config.num_retained_recent_tokens_in_cache = attention_opts.num_retained_recent_tokens_in_cache
        sparse_attention_config.xattention_threshold = attention_opts.xattention_threshold
        sparse_attention_config.xattention_block_size = attention_opts.xattention_block_size
        sparse_attention_config.xattention_stride = attention_opts.xattention_stride
        scheduler_config.sparse_attention_config = sparse_attention_config

    prefix_caching = get_or_default(scheduler_opts.enable_prefix_caching, default_scheduler_opts.enable_prefix_caching)
    if prefix_caching:
        scheduler_config.enable_prefix_caching = prefix_caching
    scheduler_config.use_cache_eviction = False
    # max_cache_size = 4096 * 4
    # kv_crush_config = KVCrushConfig(budget=max_cache_size, anchor_point_mode=KVCrushAnchorPointMode.MEAN)
    # eviction_config = CacheEvictionConfig(start_size=1024 * 4, recent_size=512, max_cache_size=max_cache_size,
    #                                       aggregation_mode=AggregationMode.NORM_SUM,
    #                                       apply_rotation=False, snapkv_window_size=8,
    #                                       kvcrush_config=kv_crush_config)
    # eviction_config.adaptive_rkv_config = AdaptiveRKVConfig()
    # scheduler_config.cache_eviction_config = eviction_config

    tokenizer_properties = {
    }

    pipe: Pipe = Pipe[args.pipe] if args.pipe else None
    parser_type = args.parser
    if not parser_type:
        is_qwen3_5 = any("qwen3_5" in model_arch.lower() for model_arch in model_architectures)
        is_qwen3moe = any("qwen3moe" in model_arch.lower() for model_arch in model_architectures)
        is_qwen3 = any("qwen3" in model_arch.lower() for model_arch in model_architectures)
        is_qwen2 = any("qwen2" in model_arch.lower() for model_arch in model_architectures)
        is_gemma4 = any("gemma4" in model_arch.lower() for model_arch in model_architectures)
        is_lfm = any("lfm2" in model_arch.lower() for model_arch in model_architectures)
        if is_gemma4:
            pipe = or_default_pipe(pipe, Pipe.VLM)
            parser_type = ParserType.gemma4
        elif is_qwen3_5:
            parser_type = ParserType.qwen3moe
            pipe = or_default_pipe(pipe, Pipe.VLM)
        elif is_qwen3moe:
            parser_type = ParserType.qwen3moe
            pipe = or_default_pipe(pipe, Pipe.LLM)
        elif is_qwen3 or is_qwen2:
            parser_type = ParserType.qwen2
            pipe = or_default_pipe(pipe, Pipe.LLM)
        elif is_lfm or is_qwen2:
            parser_type = ParserType.lfm2
            pipe = or_default_pipe(pipe, Pipe.LLM)

    if not pipe:
        log.error(f"need define --pipe for model architectures={model_architectures}")
        sys.exit(1)

    is_fix_tool_type = args.fix_tool_type != Turn.off.value
    is_detect_cycled_tool_call = args.detect_cycled_tool_call != Turn.off.value
    is_detect_looped_inference = args.detect_looped_inference != Turn.off.value

    model_parser = Qwen3MoeParser() if parser_type == ParserType.qwen3moe else \
        Gemma4ChannelParser() if parser_type == ParserType.gemma4 else \
            Qwen2Parser() if parser_type == ParserType.qwen2 else \
                Lfm2Parser() if parser_type == ParserType.lfm2 else \
                    Parser()

    log.info(
        f"model: path='{model_path}', architectures={model_architectures}, pipe={pipe}, parser='{parser_type}', "
        f"parser_type='{type(model_parser)}'")
    log.debug(f"cache dir {model_cache_dir}")

    npu_generate_hint = args.npu_generate_hint
    performance_hint = args.performance_hint

    cores_available = multiprocessing.cpu_count()
    cpu_pipeline_properties = {
        "CACHE_DIR": model_cache_dir,
        "PERFORMANCE_HINT": performance_hint,
        "ENABLE_MMAP": "YES",
        "INFERENCE_NUM_THREADS": cores_available,
    }

    gpu_enable_large_allocations = args.gpu_enable_large_allocations
    gpu_priorities = args.gpu_priorities
    gpu_pipeline_properties = {
        "CACHE_DIR": model_cache_dir,
        "PERFORMANCE_HINT": performance_hint,
        "ENABLE_MMAP": "YES",

        # "DYNAMIC_QUANTIZATION_GROUP_SIZE": "128",
        # "PERFORMANCE_HINT_NUM_REQUESTS": 1,

        "GPU_ENABLE_LARGE_ALLOCATIONS": gpu_enable_large_allocations,
        "GPU_QUEUE_THROTTLE": gpu_priorities,
        "MODEL_PRIORITY": gpu_priorities,
        "GPU_HOST_TASK_PRIORITY": gpu_priorities,
        "GPU_QUEUE_PRIORITY": gpu_priorities,

        "COMPILATION_NUM_THREADS": cores_available,
        # "INFERENCE_PRECISION_HINT": 'dynamic'
    }

    npu_prefill_hint = args.npu_prefill_hint
    npu_turbo = args.npu_turbo
    npu_compiler_type = args.npu_compiler_type

    npu_pipeline_properties: dict[str, str | int] = {
        "CACHE_DIR": model_cache_dir,
        "PERFORMANCE_HINT": performance_hint,
        "ENABLE_MMAP": "YES",
        # "PERF_COUNT": "YES",

        # "DYNAMIC_QUANTIZATION_GROUP_SIZE": "128",

        "NPU_COMPILER_TYPE": npu_compiler_type,
        "NPU_USE_NPUW": "YES",
        "NPUW_LLM": "YES",
        # "NPUW_DEVICES": "NPU,CPU",
        "NPU_TURBO": npu_turbo,
        "NPUW_LLM_GENERATE_HINT": npu_generate_hint,
        "NPUW_LLM_PREFILL_HINT": npu_prefill_hint,
        "NPUW_LLM_PREFILL_ATTENTION_HINT": "PYRAMID",
        "NPUW_LLM_GENERATE_PYRAMID": "YES",
        "NPUW_PARALLEL_COMPILE": "YES",
        "NPUW_LLM_SHARED_HEAD": "YES",

        "LOG_LEVEL": "LOG_WARNING",
    }

    if max_prompt_len:
        npu_pipeline_properties["MAX_PROMPT_LEN"] = max_prompt_len

    # max_generation_token_len = args.max_generation_token_len
    # if not max_generation_token_len:
    #     max_generation_token_len = generate_opts.max_new_tokens or default_generate_opts.max_new_tokens

    # if max_generation_token_len:
    #     npu_pipeline_properties["NPUW_LLM_MAX_GENERATION_TOKEN_LEN"] = max_generation_token_len

    if default_batch_size:
        npu_pipeline_properties["NPUW_LLM_PREFILL_CHUNK_SIZE"] = default_batch_size

    if not model_path.exists():
        log.error(f"model path is not existed: {model_path}")

    pipeline_properties = npu_pipeline_properties if is_device_npu \
        else cpu_pipeline_properties if device == DeviceType.CPU \
        else cpu_pipeline_properties | gpu_pipeline_properties if device == DeviceType.AUTO \
        else gpu_pipeline_properties
    kv_cache_precision = args.kv_cache_precision
    if kv_cache_precision:
        pipeline_properties["KV_CACHE_PRECISION"] = kv_cache_precision

    is_not_cb = is_device_npu or pipe != Pipe.CB

    attention_backend = args.attention_backend
    if attention_backend is None and is_not_cb:
        attention_backend = enum_value(AttentionBackend.PA)
        log.info(f"force attention_backend={attention_backend}")

    if attention_backend:
        pipeline_properties["ATTENTION_BACKEND"] = attention_backend

    device_value: str = enum_value(device)

    draft_model = args.draft_model
    draft_model_path = str(Path(f"{args.models_dir}/{draft_model}")) if draft_model else None
    if draft_model_path:
        draft_properties = {}
        eagle3_mode = args.eagle3_mode != Turn.off.value
        if eagle3_mode:
            draft_properties["eagle3_mode"] = True
        pipeline_properties["draft_model"] = py_openvino_genai.draft_model(draft_model_path, device_value,
                                                                           **draft_properties)

    controller_config = ControllerConfig(model_name=model, max_prompt_len=max_prompt_len,
                                         model_architectures=model_architectures,
                                         is_fix_tool_type=is_fix_tool_type,
                                         chat_template=chat_template,
                                         is_detect_cycled_tool_call=is_detect_cycled_tool_call)

    handler_config = TokenHandlerConfig(is_detect_looped_inference=is_detect_looped_inference)

    log.info(f"model={model_path}, device={device}, properties={pipeline_properties}, "
             f"generate_opts={generate_opts.model_dump(exclude_unset=True, exclude_none=True)}, "
             f"scheduler_config={scheduler_config}")
    prompt_lookup = args.prompt_lookup == Turn.on.value
    if prompt_lookup:
        pipeline_properties["prompt_lookup"] = True

    app = init_sequential_engine(
        controller_config=controller_config,
        model_path=str(model_path),
        device=device_value,
        vlm=pipe == Pipe.VLM,
        parser=model_parser,
        scheduler_config=scheduler_config if not is_device_npu else None,
        generate_opts=generate_opts,
        handler_config=handler_config,
        pipeline_properties=pipeline_properties,
        stop_signal=stop_signal,
    ) if is_not_cb else init_continuous_batching_engine(
        controller_config=controller_config,
        model_path=str(model_path),
        device=device_value,
        parser=model_parser,
        generate_opts=generate_opts,
        handler_config=handler_config,
        scheduler_config=scheduler_config,
        pipeline_properties=pipeline_properties,
        tokenizer_properties=tokenizer_properties,
        stop_signal=stop_signal,
    )

    log.info(f"listening {args.host}:{args.port}")

    def server_handle():
        uvicorn.run(app, host=args.host, port=args.port, reload=False, timeout_graceful_shutdown=0)

    server_thread = threading.Thread(target=server_handle, daemon=True)
    server_thread.start()
    # handle()
    try:
        stopped = False
        while not stopped:
            stopped = stop_signal.wait(timeout=1)
        log.debug(f"main thread finish")
    except Exception as e:
        log.debug(f"main thread finish with error: {e}")


def get_or_default[T](val: T | None, default: T | None) -> T | None:
    return val if not val is None else default


def or_default_pipe(pipe: Pipe, default: Pipe) -> Pipe:
    return pipe or default


def init_continuous_batching_engine(controller_config: ControllerConfig,
                                    handler_config: TokenHandlerConfig,
                                    model_path: str, device: str, parser: Parser,
                                    stop_signal: threading.Event,
                                    scheduler_config=py_openvino_genai.SchedulerConfig(),
                                    generate_opts=GenerateOpts(),
                                    pipeline_properties: dict[str, Any] | None = None,
                                    tokenizer_properties: dict[str, Any] | None = None,
                                    vision_encoder_properties: dict[str, Any] | None = None) -> FastAPI:
    start_mem = get_current_memory()
    log.debug(f"consumed memory: {start_mem:.2f} MB")

    if not pipeline_properties:
        pipeline_properties = {}
    if not tokenizer_properties:
        tokenizer_properties = {}
    if not vision_encoder_properties:
        vision_encoder_properties = {}
    try:
        pipe = py_openvino_genai.ContinuousBatchingPipeline(models_path=model_path,
                                                            scheduler_config=scheduler_config,
                                                            device=device,
                                                            properties=pipeline_properties,
                                                            tokenizer_properties=tokenizer_properties,
                                                            vision_encoder_properties=vision_encoder_properties)
        log.info(f"model loaded successfully, pipe {type(pipe)}")

        loaded_pipe_mem = get_current_memory()
        delta = loaded_pipe_mem - start_mem

        log.debug(f"consumed memory: {loaded_pipe_mem:.2f} MB, delta: {delta:.2f} MB")
    except Exception as e:
        log.error(f"instantiate pipeline error: {e}", exc_info=e)
        sys.exit(1)

    return new_app(ContinuousBatchingController(config=controller_config,
                                                parser=parser, pipe=pipe,
                                                generate_opts=generate_opts,
                                                handler_config=handler_config,
                                                stop_signal=stop_signal))


def init_sequential_engine(controller_config: ControllerConfig,
                           handler_config: TokenHandlerConfig,
                           model_path: str,
                           device: str, vlm: bool, parser: Parser,
                           stop_signal: threading.Event,
                           scheduler_config: py_openvino_genai.SchedulerConfig | None = None,
                           generate_opts=GenerateOpts(),
                           pipeline_properties: dict[str, Any] | None = None) -> FastAPI:
    if not pipeline_properties:
        pipeline_properties = {}
    if scheduler_config:
        pipeline_properties["scheduler_config"] = scheduler_config

    start_mem = get_current_memory()
    log.debug(f"consumed memory: {start_mem:.2f} MB")

    pipe = (
        py_openvino_genai.VLMPipeline(models_path=model_path, device=device, **pipeline_properties) if vlm else
        py_openvino_genai.LLMPipeline(models_path=model_path, device=device, **pipeline_properties)
    )

    log.info(f"model loaded successfully, pipe {type(pipe)}")
    loaded_pipe_mem = get_current_memory()
    delta = loaded_pipe_mem - start_mem

    log.debug(f"consumed memory: {loaded_pipe_mem:.2f} MB, delta: {delta:.2f} MB")

    return new_app(VlmController(config=controller_config,
                                 parser=parser, pipe=pipe,
                                 generate_opts=generate_opts,
                                 handler_config=handler_config,
                                 stop_signal=stop_signal))
