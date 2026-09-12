import os
from pathlib import Path
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
from openvino_tokenizers import convert_tokenizer
from openvino import save_model

model_id = "./josephmayo/LFM2.5-8B-A1B-Coder"
save_dir = Path("../models/LFM2.5-8B-A1B-Coder-int4-sym-g128-awq")

quantization_config = {
    "bits": 4,
    "sym": True,
    "group_size": 128,
    "ratio": 1.0,
    "dataset": "wikitext2",
    "awq": True,
    "scale_estimation": False,
    "backup_precision": "int8_sym",
}

print("Загрузка и квантование модели...")
model = OVModelForCausalLM.from_pretrained(
    model_id,
    export=True,
    trust_remote_code=True,
    quantization_config=quantization_config,
    compile=False
)

print(f"Сохранение весов модели в {save_dir}...")
model.save_pretrained(save_dir)

print("Загрузка и сохранение оригинального токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.save_pretrained(save_dir)

print("Компиляция openvino_tokenizer и openvino_detokenizer...")
ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True)

# Сохраняем скомпилированные .xml и .bin файлы токенизаторов в папку к модели
save_model(ov_tokenizer, save_dir / "openvino_tokenizer.xml")
save_model(ov_detokenizer, save_dir / "openvino_detokenizer.xml")

print("Готово! Все файлы, включая детокенизатор, успешно созданы.")
