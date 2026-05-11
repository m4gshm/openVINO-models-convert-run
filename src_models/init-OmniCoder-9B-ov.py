from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

model_id = "./Tesslate/OmniCoder-9B"
save_dir = "../models/OmniCoder-9B-ov"

# Загружаем и экспортируем (библиотека сама применит патчи)
model = OVModelForCausalLM.from_pretrained(
    model_id, 
    export=True, 
    trust_remote_code=True,
    task="image-text-to-text", # Обман системы регистрации
    attn_implementation="eager", # Добавьте эту строку
    load_in_8bit=False # или используйте NNCF для int4 позже
)
model.save_pretrained(save_dir)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(save_dir)
