from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

model_id = "./josephmayo/LFM2.5-8B-A1B-Coder"
save_dir = "../models/LFM2.5-8B-A1B-Coder-int4-sym-g128-awq"

# Настройка конфигурации квантования
quantization_config = {
    "bits": 4,
    "sym": True,
    "group_size": 128,
    "ratio": 1.0,
    "dataset": "wikitext2",
    "awq": True
}

print("Загрузка и квантование модели...")
model = OVModelForCausalLM.from_pretrained(
    model_id,
    export=True,
    trust_remote_code=True,
    quantization_config=quantization_config,
    compile=False
)

print(f"Сохранение модели в {save_dir}...")
model.save_pretrained(save_dir)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.save_pretrained(save_dir)
print("Готово!")
