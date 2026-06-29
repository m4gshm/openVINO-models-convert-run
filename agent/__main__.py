import argparse
import json
import logging.config
import os
import sys
from enum import Enum
from pathlib import Path

import uvicorn
from openvino_genai.py_openvino_genai import SchedulerConfig
from pydantic.json import pydantic_encoder

from agent.openai import GenerateConfig, default_generate_config
from agent.parser import Parser
from agent.parser.qwen2 import Qwen2Parser
from agent.server import init_continuous_batching_engine, init_sequential_engine
from .common.log import logging_config
from .inference.token_handler import TokenHandlerConfig
from .parser.gemma4 import Gemma4ChannelParser
from .parser.qwen3 import Qwen3Parser

default_device = "GPU"

default_model = "OmniCoder-9B-int4-sym-g128"
default_models_dir = f"./models"
default_models_cache_dir = f"./models_cache"

default_batch_size = 1024

os.environ["OPENVINO_LOG_LEVEL"] = "4"
os.environ["ONEDNN_VERBOSE"] = "ON"
os.environ["ONEDNN_VERBOSE_TIMESTAMP"] = "1"


class Pipe(Enum):
    CB = 'CB'
    VLM = 'VLM'
    LLM = 'LLM'


class ParserType(Enum):
    qwen2 = 'qwen2'
    qwen3 = 'qwen3'
    gemma4 = 'gemma4'


class CachePrecision(Enum):
    u8 = 'u8'
    u4 = 'u4'
    f16 = 'f16'


class AttentionBackend(Enum):
    PA = 'PA'
    SPDA = 'SPDA'


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--host", default="127.0.0.1", help="%(default)s")
    args_parser.add_argument("--port", type=int, default=8888, help="%(default)s")
    args_parser.add_argument("--models_dir", type=str, default=default_models_dir, required=False, help="%(default)s")
    args_parser.add_argument("--models_cache_dir", type=str, default=default_models_cache_dir, help="%(default)s")
    args_parser.add_argument("--model", type=str, default=default_model, help="%(default)s")
    args_parser.add_argument("--device", type=str, default=default_device, help="%(default)s")
    args_parser.add_argument("--parser", type=lambda c: ParserType[c], required=False,
                             default=None, choices=list(ParserType), help="%(default)s")
    args_parser.add_argument("--pipe", type=lambda c: Pipe[c], required=False,
                             default=None, choices=list(Pipe), help="%(default)s")
    args_parser.add_argument("--attention_backend", type=lambda c: AttentionBackend[c], required=False,
                             default=None, choices=list(AttentionBackend), help="%(default)s")
    args_parser.add_argument("--no_prefix_caching", type=bool, required=False, default=False, help="%(default)s")
    args_parser.add_argument("--max_prompt_len", type=int, required=False, default=None, help="%(default)s")
    args_parser.add_argument("--cache_size", type=int, required=False, default=None, help="%(default)s")
    args_parser.add_argument("--cache_precision", type=lambda c: CachePrecision[c], required=False,
                             default=None, choices=list(CachePrecision), help="%(default)s")
    args_parser.add_argument("--chat_template_file", type=str, required=False, default=None, help="%(default)s")
    args_parser.add_argument("--generate_config_file", type=str, required=False,
                             default=".config/generate_config.json",
                             help="%(default)s")
    args = args_parser.parse_args()

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
    base_log_config = logging_config(f"./logs/{model_name}")

    # uvcorn_logs = uvicorn.config.LOGGING_CONFIG
    # uvcorn_logs["formatters"]["default"]["format"] = log_format_simple
    # uvcorn_logs["formatters"]["access"]["format"] = (
    #         log_format_prefix + " - %(client_addr)s - '%(request_line)s' %(status_code)s"
    # )

    logging.config.dictConfig(base_log_config)

    log = logging.getLogger(__name__)
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
            except Exception as e:
                log.error(f"error on read {openvino_model_config_json}: {e}")

    handler_config = TokenHandlerConfig()

    generate_config_file = args.generate_config_file
    generate_config: GenerateConfig | None = None
    if generate_config_file:
        log.info(f"load {generate_config_file}")
        try:
            with open(generate_config_file, "r", encoding="utf-8") as file:
                generate_config: GenerateConfig = GenerateConfig.model_validate_json(file.read())
        except FileNotFoundError as e:
            log.error(f"{e}")
            raise e

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

    if not generate_config:
        generate_config = default_generate_config()

    max_prompt_len = args.max_prompt_len
    if not max_prompt_len:
        max_prompt_len = generate_config.max_tokens
    device = args.device
    is_device_npu = device == "NPU"
    if not max_prompt_len:
        max_prompt_len = max_position_embeddings

    generate_config.max_tokens = max_prompt_len

    dynamic_split_fuse = True

    scheduler_config = SchedulerConfig()
    scheduler_config.max_num_batched_tokens = default_batch_size if dynamic_split_fuse else max_prompt_len
    # scheduler_config.num_kv_blocks = 2048
    scheduler_config.cache_size = args.cache_size if args.cache_size else 0
    # scheduler_config.cache_interval_multiplier = 1
    # scheduler_config.num_linear_attention_blocks = 256
    scheduler_config.max_num_seqs = 1
    scheduler_config.dynamic_split_fuse = dynamic_split_fuse
    # scheduler_config.use_sparse_attention = True
    # scheduler_config.sparse_attention_config
    scheduler_config.enable_prefix_caching = False if args.no_prefix_caching else True
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

    pipe: Pipe = args.pipe
    parser_type: ParserType = args.parser
    if not parser_type:
        is_qwen3_5 = any("qwen3_5" in model_arch.lower() for model_arch in model_architectures)
        is_qwen3 = any("qwen3moe" in model_arch.lower() for model_arch in model_architectures)
        is_qwen2 = any("qwen2" in model_arch.lower() for model_arch in model_architectures)
        is_gemma4 = any("gemma4" in model_arch.lower() for model_arch in model_architectures)
        if is_qwen3 or is_qwen3_5:
            parser_type = ParserType.qwen3
            if not pipe:
                if is_qwen3_5:
                    # pipe = Pipe.CB
                    pipe = Pipe.VLM
                else:
                    pipe = Pipe.LLM
        elif is_qwen2:
            parser_type = ParserType.qwen2
            pipe = Pipe.LLM
        elif is_gemma4:
            pipe = Pipe.VLM
            parser_type = ParserType.gemma4

    if not pipe:
        log.error("need define --pipe")
        sys.exit(1)

    model_parser = Qwen3Parser() if parser_type == ParserType.qwen3 else \
        Gemma4ChannelParser() if parser_type == ParserType.gemma4 else \
            Qwen2Parser() if parser_type == ParserType.qwen2 else \
                Parser()

    log.info(
        f"model: path='{model_path}', architectures={model_architectures}, parser='{parser_type}', parser_type='{type(model_parser)}'")
    log.debug(f"cache dir {model_cache_dir}")

    gpu_pipeline_properties = {
        "CACHE_DIR": model_cache_dir,
        "PERFORMANCE_HINT": "LATENCY",
        "ENABLE_MMAP": "YES",
        # "PERF_COUNT": "YES",

        # "LOG_LEVEL": "LOG_WARNING",
        # "KEY_CACHE_QUANT_MODE": "BY_CHANNEL",
        # "DYNAMIC_QUANTIZATION_GROUP_SIZE": "128",
    }

    npu_pipeline_properties = {
        "CACHE_DIR": model_cache_dir,
        "PERFORMANCE_HINT": "LATENCY",
        "ENABLE_MMAP": "YES",
        # "PERF_COUNT": "YES",

        # "DYNAMIC_QUANTIZATION_GROUP_SIZE": "128",

        "NPU_COMPILER_TYPE": "PLUGIN",
        "NPU_USE_NPUW": "YES",
        "NPUW_LLM": "YES",
        "NPUW_LLM_PREFILL_CHUNK_SIZE": default_batch_size,
        "NPUW_LLM_GENERATE_HINT": "BEST_PERF",
        "NPUW_LLM_PREFILL_ATTENTION_HINT": "PYRAMID",
        "MAX_PROMPT_LEN": max_prompt_len,

        "LOG_LEVEL": "LOG_WARNING",
    }

    if not model_path.exists():
        log.error(f"model path is not existed: {model_path}")

    pipeline_properties = npu_pipeline_properties if is_device_npu else gpu_pipeline_properties
    cache_precision: CachePrecision = args.cache_precision
    if cache_precision:
        pipeline_properties["KV_CACHE_PRECISION"] = cache_precision.value

    attention_backend: AttentionBackend = args.attention_backend
    if attention_backend:
        pipeline_properties["ATTENTION_BACKEND"] = attention_backend.value

    if is_device_npu or pipe != Pipe.CB:
        app = init_sequential_engine(model_name=model_name,
                                     model_path=str(model_path),
                                     device=device,
                                     vlm=pipe == Pipe.VLM,
                                     parser=model_parser,
                                     generate_config=generate_config,
                                     handler_config=handler_config,
                                     chat_template=chat_template,
                                     pipeline_properties=pipeline_properties)
    else:
        app = init_continuous_batching_engine(model=model_name,
                                              model_path=str(model_path),
                                              device=device,
                                              parser=model_parser,
                                              generate_config=generate_config,
                                              handler_config=handler_config,
                                              scheduler_config=scheduler_config,
                                              pipeline_properties=pipeline_properties,
                                              chat_template=chat_template,
                                              tokenizer_properties=tokenizer_properties)

    log.info(f"listening {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
