import torch
import openvino as ov
import nncf
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

model_id = "./google/gemma-4-E2B-it"
save_dir = "../models/gemma-4-E2B-it-int4-sym-g128-r1/1"

print("1. Загрузка процессора и PyTorch модели...")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, 
    trust_remote_code=True,
    torch_dtype=torch.float32
)
model.eval()

print("2. Создание фиктивного входа для генерации трассировочного графа...")
dummy_image = Image.new("RGB", (224, 224), color="white")
dummy_prompt = "Hello"
inputs = processor(text=dummy_prompt, images=dummy_image, return_tensors="pt")

print("3. Экспорт через torch.export с отключением генерации сторонних объектов...")
with torch.no_grad():
    # Модифицируем параметры вызова: отключаем использование KV-cache на момент экспорта,
    # чтобы DynamicCache не попадал в выходной поток и не вызывал ошибку компилятора
    inputs["use_cache"] = True 
    
    # Извлекаем именованные аргументы для передачи в трассировщик
    example_kwargs = {k: v for k, v in inputs.items()}
    
    # Современный Export-метод PyTorch 2.x без использования устаревшего JIT
    exported_program = torch.export.export(model, args=(), kwargs=example_kwargs)

print("4. Конвертация полученного FX-графа в формат OpenVINO FP16...")
ov_model_fp16 = ov.convert_model(exported_program)

print("5. Запуск квантования весов через NNCF (INT4 Symmetric, g128)...")
compressed_model = nncf.compress_weights(
    ov_model_fp16,
    mode=nncf.CompressWeightsMode.INT4_SYM,
    group_size=128,
    ratio=1.0
)

print(f"6. Сохранение скомпилированной OpenVINO IR модели в {save_dir}...")
ov.save_model(compressed_model, f"{save_dir}/openvino_model.xml")
processor.save_pretrained(save_dir)

print("Экспорт успешно завершен!")
