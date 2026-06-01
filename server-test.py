import openvino_genai as ov_genai
import openvino as ov
import os
from PIL import Image  # Понадобится для работы с картинками, если решите их подать

# 1. Путь к вашей модели OpenVINO IR
model_path = "./models/gemma-4-E2B-it-int4-sym-g128/1"

# Переменная для ядра OpenVINO (значения от 1 до 5: 1-ошибки, 4-дебаг, 5-трассировка)
# os.environ["OPENVINO_LOG_LEVEL"] = "4" 
# Специфичное логирование для компиляции GPU-плагина (OpenCL/Neo)
# os.environ["OV_GPU_PRINT_PROGRAM_LOG"] = "1"

# 2. Настройка кэша для GPU, чтобы модель при втором запуске загружалась мгновенно
config = {"CACHE_DIR": "./cache_dir"}

print("Загрузка мультимодальной модели на GPU...")
# Используем VLM Pipeline вместо LLM Pipeline
pipe = ov_genai.VLMPipeline(model_path, "GPU", **config)

# 3. Пример 1: Отправка только текстового промпта
prompt = "def is_prime(n):"
print(f"\nЗапрос (Текст): {prompt}\n---")

result = pipe.generate(prompt, max_new_tokens=100)
print(result)
