import argparse
import logging.config
import os

import uvicorn
from openvino_genai import SchedulerConfig
from pydantic.json import pydantic_encoder

from agent.openai import default_generate_config, GenerateConfig
from agent.server import init_continuous_batching_engine, init_sequential_engine
from .common.log import log_format_prefix, log_format_simple, logging_config
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


def main():
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument("--host", default="127.0.0.1", help="%(default)s)")
    model_parser.add_argument("--port", type=int, default=8888, help="%(default)s)")
    model_parser.add_argument("--models_dir", type=str, default=default_models_dir, help="%(default)s)")
    model_parser.add_argument("--models_cache_dir", type=str, default=default_models_cache_dir, help="%(default)s)")
    model_parser.add_argument("--model", type=str, default=default_model, help="%(default)s)")
    model_parser.add_argument("--device", type=str, default=default_device, help="%(default)s)")
    model_parser.add_argument("--parser", type=str, required=False, default=None, help="%(default)s)")
    model_parser.add_argument("--no_continuous_batching", type=bool, required=False, default=False, help="%(default)s)")
    model_parser.add_argument("--no_vlm", type=bool, required=False, default=None, help="%(default)s)")
    model_parser.add_argument("--max_prompt_len", type=int, required=False, default=None, help="%(default)s)")
    model_parser.add_argument("--cache_size", type=int, required=False, default=None, help="%(default)s)")
    model_parser.add_argument("--generate_config_file", type=str, required=False,
                              default=".config/generate_config.json",
                              help="%(default)s)")
    args = model_parser.parse_args()

    model = args.model
    base_log_config = logging_config(f"./logs/{model}")

    # uvcorn_logs = uvicorn.config.LOGGING_CONFIG
    # uvcorn_logs["formatters"]["default"]["format"] = log_format_simple
    # uvcorn_logs["formatters"]["access"]["format"] = (
    #         log_format_prefix + " - %(client_addr)s - '%(request_line)s' %(status_code)s"
    # )

    logging.config.dictConfig(base_log_config)

    log = logging.getLogger(__name__)
    log.info("server starting")


    handler_config = TokenHandlerConfig()

    scheduler_config = SchedulerConfig()
    scheduler_config.max_num_batched_tokens = default_batch_size
    scheduler_config.cache_size = args.cache_size if args.cache_size else 0
    scheduler_config.max_num_seqs = 2
    scheduler_config.dynamic_split_fuse = True
    scheduler_config.enable_prefix_caching = True
    scheduler_config.use_cache_eviction = False

    generate_config_file = args.generate_config_file
    generate_config: GenerateConfig | None = None
    if generate_config_file:
        log.info(f"load {generate_config_file}")
        try:
            with open(generate_config_file, "r", encoding="utf-8") as file:
                # Read the file as a raw string directly into the validator
                generate_config: GenerateConfig = GenerateConfig.model_validate_json(file.read())
        except FileNotFoundError as e:
            log.debug(f"{e}")

    if not generate_config:
        generate_config = GenerateConfig()

    max_prompt_len = args.max_prompt_len
    if not max_prompt_len:
        max_prompt_len = generate_config.max_tokens
    if not max_prompt_len:
        max_prompt_len = 16384 if args.device == "NPU" else 65536

    generate_config.max_tokens = max_prompt_len

    tokenizer_properties = {
    }

    args_parser = args.parser
    if not args_parser:
        model_lower = model.lower()
        qwen3_models = ["omnicoder", "qwen"]
        is_qwen = any(model in model_lower for model in qwen3_models)
        if is_qwen:
            args_parser = "qwen3"
        elif "gemma" in model_lower:
            args_parser = "gemma4"
        log.info(f"model parser '{args_parser}'")

    model_parser = Qwen3Parser() if args_parser == "qwen3" else Gemma4ChannelParser() if args_parser == "gemma4" else Parser()

    model_path = f"{args.models_dir}/{model}"
    model_cache_dir = f"{args.models_cache_dir}/{model}"

    log.info(f"loading model from {model_path}, cache dir {model_cache_dir}")

    gpu_pipeline_properties = {
        "CACHE_DIR": model_cache_dir,
        "PERFORMANCE_HINT": "LATENCY",
        "ENABLE_MMAP": "YES",
        # "PERF_COUNT": "YES"

        "KV_CACHE_PRECISION": "u8",
        # "KEY_CACHE_GROUP_SIZE": 128,
        # "VALUE_CACHE_GROUP_SIZE": 128,
    }

    npu_pipeline_properties = {
        "CACHE_DIR": model_cache_dir,
        "PERFORMANCE_HINT": "LATENCY",
        "ENABLE_MMAP": "YES",
        # "PERF_COUNT": "YES",

        # "KV_CACHE_PRECISION": "u8",
        "KEY_CACHE_GROUP_SIZE": 128,
        "VALUE_CACHE_GROUP_SIZE": 128,

        "NPU_COMPILER_TYPE": "PLUGIN",
        "NPU_USE_NPUW": "YES",
        "NPUW_LLM": "YES",
        "NPUW_LLM_PREFILL_CHUNK_SIZE": default_batch_size,
        "NPUW_LLM_GENERATE_HINT": "BEST_PERF",
        "NPUW_LLM_PREFILL_ATTENTION_HINT": "PYRAMID",
        "MAX_PROMPT_LEN": max_prompt_len,

        "LOG_LEVEL": "LOG_WARNING",

        "DYNAMIC_QUANTIZATION_GROUP_SIZE": 128,
        "ATTENTION_BACKEND": "PA",
    }

    no_vlm = args.no_vlm
    if args.device == "NPU" or no_vlm or args.no_continuous_batching:
        if no_vlm is None:
            no_vlm = False
        app = init_sequential_engine(model=(model), model_path=model_path, device=args.device, vlm=not no_vlm,
                                     parser=model_parser,
                                     generate_config=generate_config,
                                     handler_config=handler_config,
                                     pipeline_properties=npu_pipeline_properties if args.device == "NPU" else gpu_pipeline_properties)
    else:
        app = init_continuous_batching_engine(model=(model),
                                              model_path=model_path,
                                              device=args.device,
                                              parser=model_parser,
                                              generate_config=generate_config,
                                              handler_config=handler_config,
                                              scheduler_config=scheduler_config,
                                              pipeline_properties=gpu_pipeline_properties,
                                              tokenizer_properties=tokenizer_properties)

    log.info(f"listening {args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
