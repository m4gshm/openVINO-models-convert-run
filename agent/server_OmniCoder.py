import logging.config
import os

import uvicorn
from pydantic.json import pydantic_encoder

import openai
from agent.common.log import log_format_prefix, log_format_simple
from inference.streamer import StreamerConfig
from server import init_engine

device = "GPU"

model = "OmniCoder-9B-int4-sym-g128"
model_path = f"../models/{model}/1"
model_cache_dir = f"../models_cache/{model}"

streamer_config = StreamerConfig()

generate_config = openai.GenerateConfig()

# scheduler_config = ov_genai.SchedulerConfig()
# scheduler_config.enable_prefix_caching = True
# scheduler_config.max_num_batched_tokens = 256
# scheduler_config.max_num_seqs = 1
# scheduler_config.cache_interval_multiplier = None  # 2
# scheduler_config.dynamic_split_fuse = True
# scheduler_config.use_sparse_attention = False
# scheduler_config.cache_size = 8
# scheduler_config.use_cache_eviction = True

pipe_config = {
    "CACHE_DIR": model_cache_dir,
    # "GPU_ENABLE_LARGE_ALLOCATIONS": "YES",
    # "KV_CACHE_PRECISION": "u4",
    "PERFORMANCE_HINT": "LATENCY",  # THROUGHPUT crashes process
    # "scheduler_config": scheduler_config,
    "ATTENTION_BACKEND": "PA",
    # "ATTENTION_BACKEND": "SDPA",
}

# os.environ["LOG_LEVEL"] = "4"
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

    app = init_engine(model=model, model_path=model_path, device=device, generate_config=generate_config,
                      streamer_config=streamer_config, pipe_config=pipe_config)

    uvicorn.run(app, host="127.0.0.1", port=8888)
