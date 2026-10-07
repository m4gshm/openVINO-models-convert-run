call agent.bat --model OmniCoder-9B-int4-sym-g128-se-awq ^
 --device NPU ^
 --max_prompt_len 61440 ^
 --npu_turbo YES ^
 --npu_compiler_type PLUGIN ^
 --npuw_llm_enable_block_based_kv_cache NO ^
 --npuw_llm_enable_continuous_prefill NO ^
 --npuw_llm_enable_prefix_caching YES ^
 --npuw_devices NPU,CPU ^
 --npuw_llm_prefill_moe_hint HOST_ROUTED ^
 --npuw_llm_generate_moe_hint DEVICE_ROUTED ^
 --generate_config_file .config/generate_config.json

@REM   --npu_prefill_attention_hint HFA ^
