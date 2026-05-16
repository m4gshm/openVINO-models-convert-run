import torch
import openvino as ov
import nncf
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from datasets import load_dataset

model_id = "./google/gemma-4-E2B-it"
save_dir = "../models/gemma-4-E2B-it-int4-sym-g-1-r1/1"

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
    inputs["use_cache"] = False 
    
    # Извлекаем именованные аргументы для передачи в трассировщик
    example_kwargs = {k: v for k, v in inputs.items()}
    
    # Современный Export-метод PyTorch 2.x без использования устаревшего JIT
    exported_program = torch.export.export(model, args=(), kwargs=example_kwargs)

print("4. Конвертация полученного FX-графа в формат OpenVINO FP16...")
ov_model_fp16 = ov.convert_model(exported_program)

print("5. Запуск квантования весов через NNCF (INT4 Symmetric, group_size=-1, AWQ)...")

# 5.1. Загрузка и подготовка калибровочного датасета WikiText-2
# Используем небольшой срез для ускорения процесса калибровки
calibration_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:128]")

def transform_fn(data_item):
    """Функция преобразует текстовую строку в формат входных тензоров OpenVINO модели"""
    text = data_item["text"]
    if not text.strip():
        return None 
        
    # Токенизируем текст
    inputs = processor(text=text, return_tensors="pt")
    
    # Конвертируем в NumPy только реальные тензоры данных (input_ids, attention_mask и т.д.)
    ov_inputs = {k: v.numpy() for k, v in inputs.items() if hasattr(v, "numpy")}
    
    # ВАЖНО: Больше НЕ добавляем ov_inputs["use_cache"] = False здесь.
    # Для калибровки NNCF нужны только сырые тензоры данных.
    
    return ov_inputs

# Фильтруем None значения
calibration_data = [transform_fn(item) for item in calibration_dataset]
calibration_data = [item for item in calibration_data if item is not None]

# Создаем объект NNCF Dataset
nncf_calibration_dataset = nncf.Dataset(calibration_data)

# Запуск квантования весов через NNCF (INT4 Symmetric, group_size=-1, AWQ)
compressed_model = nncf.compress_weights(
    ov_model_fp16,
    mode=nncf.CompressWeightsMode.INT4_SYM,
    group_size=-1,                       
    ratio=1.0,
    dataset=nncf_calibration_dataset,    
    awq=True                             
)


print(f"6. Сохранение скомпилированной OpenVINO IR модели в {save_dir}...")
ov.save_model(compressed_model, f"{save_dir}/openvino_model.xml")
processor.save_pretrained(save_dir)

print("Экспорт успешно завершен!")


print(f"6. Сохранение скомпилированной OpenVINO IR модели в {save_dir}...")
ov.save_model(compressed_model, f"{save_dir}/openvino_model.xml")
processor.save_pretrained(save_dir)

print("Экспорт успешно завершен!")
