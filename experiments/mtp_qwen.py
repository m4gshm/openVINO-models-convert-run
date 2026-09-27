import os
import time

import openvino_genai as ov_genai

os.environ["OPENVINO_LOG_LEVEL"] = "4"
os.environ["ONEDNN_VERBOSE"] = "ON"
os.environ["ONEDNN_VERBOSE_TIMESTAMP"] = "1"

model_path = "..\\models\\Ornith-1.5-9B-int4-sym-g128-awq"
cache_path = "..\\models_cache\\Ornith-1.5-9B-int4-sym-g128-awq"

device = "GPU"
scheduler_config = ov_genai.SchedulerConfig()
scheduler_config.enable_prefix_caching = False
draft = ov_genai.draft_model(model_path, device, **{"mtp_mode": True})
gpu_pipeline_properties = {
    "CACHE_DIR": cache_path,
    "PERFORMANCE_HINT": "LATENCY",
    "ENABLE_MMAP": "YES",

    "GPU_ENABLE_LARGE_ALLOCATIONS": "YES",
    "GPU_QUEUE_THROTTLE": "HIGH",
    "MODEL_PRIORITY": "HIGH",
    "GPU_HOST_TASK_PRIORITY": "HIGH",
    "GPU_QUEUE_PRIORITY": "HIGH",

    "COMPILATION_NUM_THREADS": 16,
}

pipe = ov_genai.VLMPipeline(
    model_path,
    device,
    draft_model=draft,
    scheduler_config=scheduler_config,
    **gpu_pipeline_properties
)

generation_config = ov_genai.GenerationConfig()
generation_config.max_new_tokens = 20000000
generation_config.do_sample = False
generation_config.num_return_sequences = 1
generation_config.num_assistant_tokens = 2

start = time.time()
print(pipe.generate("2+2=?", generation_config=generation_config))
end = time.time()
print(end - start)
