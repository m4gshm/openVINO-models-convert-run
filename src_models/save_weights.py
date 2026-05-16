import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "./google/gemma-4-E2B-it"
raw_out_dir = "../models/gemma4_raw_safetensors"

print("1. Скачивание и сохранение весов в Safetensors...")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, 
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# Сохраняем модель как чистые бинарные веса .safetensors
model.save_pretrained(raw_out_dir, safe_serialization=True)
processor.save_pretrained(raw_out_dir)
print(f"Веса успешно сохранены в {raw_out_dir}")
