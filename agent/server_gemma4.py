import argparse
import logging.config
import os

import uvicorn
from openvino_genai.py_openvino_genai import SchedulerConfig
from pydantic.json import pydantic_encoder

import openai
from agent.common.log import log_format_prefix, log_format_simple
from agent.parser.gemma4 import Gemma4ChannelParser
from agent.server import init_continuous_batching_engine
from inference.token_handler import TokenHandlerConfig
from server import init_sequential_engine

device = "NPU"
continuous_batching_support = False

model = "gemma-4-E2B-it-int4-sym-g128"
# model = "gemma-4-E4B-it-int8-sym"
model_path = f"../models/{model}/1"
model_cache_dir = f"../models_cache/{model}"

handler_config = TokenHandlerConfig()

max_prompt_len = 16384 if device == "NPU" else 65536

scheduler_config = SchedulerConfig()
scheduler_config.max_num_batched_tokens = 1024
scheduler_config.cache_size = 0
scheduler_config.max_num_seqs = 1
scheduler_config.dynamic_split_fuse = True
scheduler_config.enable_prefix_caching = True
scheduler_config.use_cache_eviction = False

generate_config = openai.GenerateConfig(
    default_max_tokens=max_prompt_len,
    default_temperature=1.0,
    default_top_p=0.95,
    default_top_k=64,
    default_min_p=0.05,
    default_repetition_penalty=1.1,
)

tokenizer_properties = {
}

gpu_pipeline_properties = {
    "CACHE_DIR": model_cache_dir,
    "PERFORMANCE_HINT": "LATENCY"
}

npu_pipeline_properties = {
    "CACHE_DIR": model_cache_dir,
    "PERFORMANCE_HINT": "LATENCY",

    "NPU_COMPILER_TYPE": "PLUGIN",
    "NPUW_LLM_GENERATE_HINT": "BEST_PERF",
    "NPUW_LLM_PREFILL_ATTENTION_HINT": "PYRAMID",
    "LOG_LEVEL": "LOG_WARNING",
    "MAX_PROMPT_LEN": max_prompt_len,
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": 128,
    "ATTENTION_BACKEND": "SDPA",
}

os.environ["OPENVINO_LOG_LEVEL"] = "4"
os.environ["ONEDNN_VERBOSE"] = "ON"
os.environ["ONEDNN_VERBOSE_TIMESTAMP"] = "1"

if __name__ == "__main__":
    log = logging.getLogger(__name__)
    log.info("server starting")
    log_config = uvicorn.config.LOGGING_CONFIG

    log_config["formatters"]["default"]["format"] = log_format_simple
    log_config["formatters"]["access"]["format"] = (
            log_format_prefix + " - %(client_addr)s - '%(request_line)s' %(status_code)s"
    )

    parser = Gemma4ChannelParser()

    if device == "GPU" and continuous_batching_support:
        app = init_continuous_batching_engine(model=model, model_path=model_path, device=device, parser=parser,
                                              generate_config=generate_config,
                                              handler_config=handler_config,
                                              scheduler_config=scheduler_config,
                                              pipeline_properties=gpu_pipeline_properties,
                                              tokenizer_properties=tokenizer_properties)
    else:
        app = init_sequential_engine(model=model, model_path=model_path, device=device, vlm=True, parser=parser,
                                     generate_config=generate_config,
                                     handler_config=handler_config,
                                     pipeline_properties=npu_pipeline_properties if device == "NPU" else gpu_pipeline_properties)

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="%(default)s)")
    parser.add_argument("--port", type=int, default=8888, help="%(default)s)")
    args = parser.parse_args()
    log.info(f"listening {args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port)
