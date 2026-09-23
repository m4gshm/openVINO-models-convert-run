import argparse
import logging.config
import os
import signal
import sys
import threading
from enum import Enum
from typing import Any

from pydantic.json import pydantic_encoder

from agent.server_openai import run_openai_proxy
from agent.server_openvino import run_openvino

default_model = "OmniCoder-9B-int4-sym-g128-se-awq"
default_models_dir = f"./models"
default_models_cache_dir = f"./models_cache"

default_batch_size = 1024

os.environ["OPENVINO_LOG_LEVEL"] = "4"
os.environ["ONEDNN_VERBOSE"] = "ON"
os.environ["ONEDNN_VERBOSE_TIMESTAMP"] = "1"

log = logging.getLogger(__name__)


class Pipe(Enum):
    CB = 'CB'
    VLM = 'VLM'
    LLM = 'LLM'


class ParserType(Enum):
    qwen2 = 'qwen2'
    qwen3moe = 'qwen3moe'
    gemma4 = 'gemma4'
    lfm2 = 'lfm2'


class KvCachePrecision(Enum):
    u8 = 'u8'
    u4 = 'u4'
    f16 = 'f16'


class AttentionBackend(Enum):
    PA = 'PA'
    SDPA = 'SDPA'


class Turn(Enum):
    on = 'on'
    off = 'off'


class NpuCompilerType(Enum):
    DRIVER = 'DRIVER'
    PLUGIN = 'PLUGIN'


class YesNo(Enum):
    YES = 'YES'
    NO = 'NO'


class Level(Enum):
    HIGH = 'HIGH'
    MEDIUM = 'MEDIUM'


class NpuGenerateHint(Enum):
    BEST_PERF = 'BEST_PERF'
    FAST_COMPILE = 'FAST_COMPILE'


class PrefillHint(Enum):
    DYNAMIC = 'DYNAMIC'
    STATIC = 'STATIC'


class DeviceType(Enum):
    GPU = 'GPU'
    NPU = 'NPU'
    CPU = 'CPU'
    AUTO = 'AUTO'


class PerformanceHint(Enum):
    LATENCY = 'LATENCY'
    THROUGHPUT = 'THROUGHPUT'
    CUMULATIVE_THROUGHPUT = 'CUMULATIVE_THROUGHPUT'


stop_signal = threading.Event()


def handle_exit_signal(signum, frame):
    log.info(f"handle signal {signum}")
    stop_signal.set()


signal.signal(signal.SIGINT, handle_exit_signal)
signal.signal(signal.SIGTERM, handle_exit_signal)


def enum_values[T: Enum](enum_class: type[T]) -> list[str]:
    return [member.value for member in enum_class]


def enum_value[T: Enum](member: T) -> Any:
    return member.value


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--host", default="127.0.0.1", help="%(default)s")
    args_parser.add_argument("--port", type=int, default=8888, help="%(default)s")
    args_parser.add_argument("--models_dir", type=str, default=default_models_dir, required=False, help="%(default)s")
    args_parser.add_argument("--models_cache_dir", type=str, default=default_models_cache_dir, help="%(default)s")
    args_parser.add_argument("--model", type=str, default=default_model, help="%(default)s")
    args_parser.add_argument("--draft_model", type=str, help="%(default)s")
    args_parser.add_argument("--openai_api_key", type=str, default="", help="OpenAI API key")
    args_parser.add_argument("--openai_base_url", type=str, default="https://api.openai.com/v1",
                             help="OpenAI API base URL")
    args_parser.add_argument("--openai_model", type=str, default="", help="Override model name for OpenAI API")
    args_parser.add_argument("--device", type=str, required=False,
                             default=enum_value(DeviceType.GPU), choices=enum_values(DeviceType), help="%(default)s")
    args_parser.add_argument("--performance_hint", type=str, required=False,
                             default=enum_value(PerformanceHint.LATENCY), choices=enum_values(PerformanceHint),
                             help="%(default)s")
    args_parser.add_argument("--parser", type=str, required=False,
                             default=None, choices=enum_values(ParserType), help="%(default)s")
    args_parser.add_argument("--pipe", type=str, required=False,
                             default=None, choices=enum_values(Pipe), help="%(default)s")
    args_parser.add_argument("--attention_backend", type=str, required=False,
                             default=None, choices=enum_values(AttentionBackend), help="%(default)s")
    args_parser.add_argument("--max_prompt_len", type=int, required=False, default=None, help="%(default)s")
    # args_parser.add_argument("--max_generation_token_len", type=int, required=False, default=None, help="%(default)s")
    args_parser.add_argument("--kv_cache_precision", type=str, required=False,
                             default=None, choices=enum_values(KvCachePrecision), help="%(default)s")

    args_parser.add_argument("--chat_template_file", type=str, required=False, default=None, help="%(default)s")
    args_parser.add_argument("--fix_tool_type", type=str, required=False,
                             default=None, choices=enum_values(Turn), help="%(default)s")
    args_parser.add_argument("--detect_cycled_tool_call", type=str, required=False,
                             default=None, choices=enum_values(Turn), help="%(default)s")
    args_parser.add_argument("--detect_looped_inference", type=str, required=False,
                             default=None, choices=enum_values(Turn), help="%(default)s")
    args_parser.add_argument("--generate_config_file", type=str, required=False,
                             default=".config/generate_config.json",
                             help="%(default)s")
    args_parser.add_argument("--scheduler_config_file", type=str, required=False,
                             default=".config/scheduler_config.json",
                             help="%(default)s")
    args_parser.add_argument("--npu_compiler_type", type=str, required=False,
                             default=enum_value(NpuCompilerType.DRIVER), choices=enum_values(NpuCompilerType),
                             help="%(default)s")
    args_parser.add_argument("--npu_generate_hint", type=str, required=False,
                             default=enum_value(NpuGenerateHint.FAST_COMPILE), choices=enum_values(NpuGenerateHint),
                             help="%(default)s")
    args_parser.add_argument("--npu_prefill_hint", type=str, required=False,
                             default=enum_value(PrefillHint.DYNAMIC), choices=enum_values(PrefillHint),
                             help="%(default)s")
    args_parser.add_argument("--npu_turbo", type=str, required=False,
                             default=enum_value(YesNo.NO), choices=enum_values(YesNo), help="%(default)s")

    args_parser.add_argument("--gpu_enable_large_allocations", type=str, required=False,
                             default=enum_value(YesNo.YES), choices=enum_values(YesNo), help="%(default)s")
    args_parser.add_argument("--gpu_priorities", type=str, required=False,
                             default=enum_value(Level.HIGH), choices=enum_values(Level), help="%(default)s")
    args_parser.add_argument("--prompt_lookup", type=str, required=False,
                             default=enum_value(Turn.off), choices=enum_values(Turn), help="%(default)s")
    args_parser.add_argument("--eagle3_mode", type=str, required=False,
                             default=enum_value(Turn.on), choices=enum_values(Turn), help="%(default)s")

    args = args_parser.parse_args()

    # Determine mode: --model for OpenVINO, --openai_base_url for OpenAI proxy
    has_model = args.model != default_model
    has_openai_base = bool(args.openai_base_url and args.openai_base_url != "https://api.openai.com/v1")

    if has_model and has_openai_base:
        log.error("ERROR: Cannot specify both --model and --openai_base_url. Use one or the other.")
        sys.exit(1)

    if not has_model and not has_openai_base:
        log.error("ERROR: Must specify either --model for OpenVINO mode or --openai_base_url for OpenAI proxy mode.")
        sys.exit(1)

    is_openai_mode = has_openai_base

    if is_openai_mode:
        run_openai_proxy(args)
    else:
        run_openvino(args)


if __name__ == "__main__":
    main()
