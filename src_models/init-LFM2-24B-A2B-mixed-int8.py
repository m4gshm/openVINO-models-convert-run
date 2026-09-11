import openvino as ov
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer
import nncf
from nncf import CompressWeightsMode, IgnoredScope  # Правильный импорт

model_id = "./LiquidAI/LFM2-24B-A2B"
save_dir = "../models/LFM2-24B-A2B-mixed-int8"

print("Loading model...")
# Загружаем модель для экспорта (без компиляции на лету)
model = OVModelForCausalLM.from_pretrained(model_id, export=True, trust_remote_code=True, compile=False)

print("Configuring NNCF Weight Compression...")

# Передаем параметры напрямую в функцию сжатия весов
compressed_model = nncf.compress_weights(
    model.model,
    mode=CompressWeightsMode.INT8_ASYM,
    ignored_scope=IgnoredScope(
        pattern=".*conv.*|.*ssm.*|.*recurrent.*"  # Регулярка защитит слои Liquid от сжатия
    )
)
model.model = compressed_model

print(f"Saving OpenVINO IR model to {save_dir}...")
model.save_pretrained(save_dir)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.save_pretrained(save_dir)
print("Done!")
