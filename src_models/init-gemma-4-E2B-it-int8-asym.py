# /// script
# dependencies = [
# dependencies = [
#   "transformers==5.5.4",
#   "optimum-intel",
#   "openvino-tokenizers",
#   "openvino"
# ]
# ///

from optimum.intel import OVModelForVisualCausalLM
from transformers import AutoProcessor
from openvino_tokenizers import convert_tokenizer
from openvino import save_model
import os

model_id = "./google/gemma-4-E2B-it"
output_dir = "../models/gemma-4-E2B-it-int8-asym/1"

os.makedirs(output_dir, exist_ok=True)

quantization_config = {
    "bits": 8,
    "sym": False,
    "group_size": -1,
}

model = OVModelForVisualCausalLM.from_pretrained(
    model_id,
    export=True,
    trust_remote_code=True,
    quantization_config=quantization_config,
    task="image-text-to-text"
)
model.save_pretrained(output_dir)

processor = AutoProcessor.from_pretrained(model_id)
processor.save_pretrained(output_dir)

ov_tokenizer, ov_detokenizer = convert_tokenizer(processor.tokenizer, with_detokenizer=True)

save_model(ov_tokenizer, os.path.join(output_dir, "openvino_tokenizer.xml"))

save_model(ov_detokenizer, os.path.join(output_dir, "openvino_detokenizer.xml"))

print(f"success: {output_dir}")

