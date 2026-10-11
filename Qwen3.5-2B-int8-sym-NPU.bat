call agent.bat --model Qwen3.5-2B-int8-sym ^
 --detect_cycled_tool_call on ^
 --device NPU ^
 --max_prompt_len 61440 ^
 --npu_turbo YES
@REM  --npu_compiler_type PLUGIN ^
@REM  --npuw_llm_enable_block_based_kv_cache NO ^
@REM  --npuw_llm_enable_continuous_prefill NO ^
@REM  --npuw_llm_enable_prefix_caching YES ^
@REM  --npu_prefill_attention_hint HFA ^
@REM  --npuw_devices NPU,GPU ^
@REM  --npuw_llm_prefill_moe_hint HOST_ROUTED ^
@REM  --npuw_llm_generate_moe_hint DEVICE_ROUTED
