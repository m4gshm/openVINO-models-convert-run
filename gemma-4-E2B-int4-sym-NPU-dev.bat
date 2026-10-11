call agent-dev.bat --model gemma-4-E2B-it-int4-sym-g128-se-awq ^
 --detect_cycled_tool_call off ^
 --device NPU ^
 --max_prompt_len 49152 ^
 --generate_config_file .config/gemma4_generate_config_npu.json ^
 --chat_template_file .config/gemma4_chat_template.jinja
@REM  --max_prompt_len 61440 ^
@REM  --npu_turbo YES ^
@REM  --npu_compiler_type PLUGIN ^
@REM  --npuw_llm_enable_block_based_kv_cache NO ^
@REM  --npuw_llm_enable_continuous_prefill NO ^
@REM  --npuw_llm_enable_prefix_caching YES ^
@REM  --npu_prefill_attention_hint HFA ^
@REM  --npuw_devices NPU,GPU ^
@REM  --npuw_llm_prefill_moe_hint HOST_ROUTED ^
@REM  --npuw_llm_generate_moe_hint DEVICE_ROUTED ^

