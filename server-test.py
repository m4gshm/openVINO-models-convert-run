
import os
# Переменная для ядра OpenVINO (значения от 1 до 5: 1-ошибки, 4-дебаг, 5-трассировка)
# os.environ["OPENVINO_LOG_LEVEL"] = "4" 
# Специфичное логирование для компиляции GPU-плагина (OpenCL/Neo)
# os.environ["OV_GPU_PRINT_PROGRAM_LOG"] = "1"

import openvino_genai as ov_genai

model_name = "OmniCoder-9B-int8-sym"
model_path = f"./models/{model_name}/1"
config = {"CACHE_DIR": f"./models_cache/{model_name}"}

print("Загрузка мультимодальной модели на GPU...")
# Используем VLM Pipeline вместо LLM Pipeline
pipe = ov_genai.VLMPipeline(model_path, "GPU", **config)

prompt = "def is_prime(n):"
print(f"\nЗапрос (Текст): {prompt}\n---")

result = pipe.generate(prompt, max_new_tokens=100)
print(result)
