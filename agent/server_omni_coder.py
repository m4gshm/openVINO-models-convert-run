import logging.config
import os
from zipapp import shebang_encoding

import uvicorn
from openvino_genai import CacheEvictionConfig, AggregationMode, SchedulerConfig
from pydantic.json import pydantic_encoder

import openai
from agent.common.log import log_format_prefix, log_format_simple
from inference.token_handler import TokenHandlerConfig
from server import init_engine

device = "GPU"

model = "OmniCoder-9B-int4-sym-g128"
model_path = f"../models/{model}/1"
model_cache_dir = f"../models_cache/{model}"

handler_config = TokenHandlerConfig()

scheduler_config = SchedulerConfig()
scheduler_config.max_num_batched_tokens = 4096
scheduler_config.cache_size = 0
scheduler_config.max_num_seqs = 1
scheduler_config.dynamic_split_fuse = True
scheduler_config.enable_prefix_caching = True
scheduler_config.use_cache_eviction = False
# scheduler_config.cache_eviction_config = CacheEvictionConfig(
#     start_size=1024,
#     recent_size=1024,
#     max_cache_size=4096,
#     aggregation_mode=AggregationMode.NORM_SUM,
#     apply_rotation=False,
#     snapkv_window_size=32
# )

generate_config = openai.GenerateConfig(
    default_temperature=0.4,
    default_top_p=0.95,
    default_top_k=40,
    default_min_p=0.05,
    default_repetition_penalty=1.1,
)

pipeline_properties = {
    "CACHE_DIR": model_cache_dir,
    "PERFORMANCE_HINT": "LATENCY",
}

tokenizer_properties = {
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

    app = init_engine(model=model, model_path=model_path, device=device,
                      generate_config=generate_config,
                      handler_config=handler_config,
                      scheduler_config=scheduler_config,
                      pipeline_properties=pipeline_properties,
                      tokenizer_properties=tokenizer_properties)

    uvicorn.run(app, host="127.0.0.1", port=8888)
