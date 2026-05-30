# /// script
# dependencies = [
#   "transformers>5.5.4",
#   "optimum-intel",
#   "openvino-tokenizers",
#   "openvino"
# ]
# ///

from optimum.intel import OVModelForVisualCausalLM
from transformers import AutoProcessor
from openvino_tokenizers import convert_tokenizer
from openvino import save_model
from optimum.intel import OVPipelineQuantizationConfig, OVWeightQuantizationConfig

import os

import transformers.models.gemma4.modeling_gemma4 as _gemma4_mod

_orig_init = _gemma4_mod.Gemma4TextAttention.__init__

def _patched_init(self, config, layer_idx):
    _orig_init(self, config, layer_idx)
    n = config.num_hidden_layers
    n_shared = getattr(config, "num_kv_shared_layers", 0)
    first_shared = n - n_shared
    prev = list(getattr(config, "layer_types", []))[:first_shared]
    last_sliding = max((i for i, t in enumerate(prev) if t == "sliding_attention"), default=-1)
    last_full    = max((i for i, t in enumerate(prev) if t == "full_attention"), default=-1)
    if getattr(self, "is_kv_shared_layer", False):
        if self.layer_type == "sliding_attention":
            self.kv_shared_layer_index = last_sliding
        elif self.layer_type == "full_attention":
            self.kv_shared_layer_index = last_full
        else:
            self.kv_shared_layer_index = -1
    else:
        self.kv_shared_layer_index = -1

_gemma4_mod.Gemma4TextAttention.__init__ = _patched_init

model_id = "./google/gemma-4-26B-A4B-it"
output_dir = "../models/gemma-4-26B-A4B-it-int4-sym/1"

os.makedirs(output_dir, exist_ok=True)

quantization_config = OVPipelineQuantizationConfig(
    quantization_configs={
        "lm_model": OVWeightQuantizationConfig(
            bits=4,
            sym=True,
            group_size=-1,
            # awq=True,
            backup_precision="int8_sym"
        )
    },
    # dataset="contextual"
)

model = OVModelForVisualCausalLM.from_pretrained(
    model_id,
    export=True,
    trust_remote_code=True,
    task="image-text-to-text",
    low_cpu_mem_usage=True,
    quantization_config=quantization_config
)
model.save_pretrained(output_dir)

processor = AutoProcessor.from_pretrained(model_id)
processor.save_pretrained(output_dir)

ov_tokenizer, ov_detokenizer = convert_tokenizer(processor.tokenizer, with_detokenizer=True)

save_model(ov_tokenizer, os.path.join(output_dir, "openvino_tokenizer.xml"))

save_model(ov_detokenizer, os.path.join(output_dir, "openvino_detokenizer.xml"))

print(f"success: {output_dir}")

