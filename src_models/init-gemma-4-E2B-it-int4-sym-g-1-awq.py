import os
from optimum.intel.openvino import OVModelForVisualCausalLM  # <- Исправлено здесь
from transformers import AutoTokenizer

# Включаем современный бэкенд трассировки для обработки слоев Gemma-4
os.environ["OV_USE_TORCH_EXPORT"] = "1"

model_id = "./google/gemma-4-E2B-it"
output_dir = "../models/gemma-4-E2B-it-int4-sym-g-1-r1-awq/1"

print("=== Загрузка и экспорт модели Gemma-4 в OpenVINO INT4 (AWQ) ===")

# Конфигурация квантования весов
quantization_config = {
    "bits": 4,
    "sym": True,
    "group_size": -1,
    "ratio": 1.0,
    "dataset": "contextual",
    "awq": True
}

# Загружаем и экспортируем модель
model = OVModelForVisualCausalLM.from_pretrained(
    model_id,
    export=True,
    trust_remote_code=True,
    quantization_config=quantization_config
)

# Сохраняем результат
model.save_pretrained(output_dir)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.save_pretrained(output_dir)

print(f"Успешно экспортировано в: {output_dir}")
