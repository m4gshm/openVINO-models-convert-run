call agent.bat --model Qwen3.5-2B-int8-sym ^
 --detect_cycled_tool_call on ^
 --device NPU ^
 --max_prompt_len 32768 ^
 --npu_compiler_type PLUGIN ^
 --npu_compilation_mode_params "optimization-level=2 performance-hint-override=latency" ^
 --npuw_llm_enable_block_based_kv_cache NO ^
 --npuw_llm_enable_prefix_caching NO ^
 --npuw_llm_generate_pyramid YES ^
 --npuw_devices NPU,GPU,CPU ^
 --npuw_llm_prefill_attention_hint PYRAMID ^
 --npuw_llm_generate_attention_hint DYNAMIC ^

@REM  --npuw_llm_propagate_slice_up YES ^
@REM  --npuw_llm_enable_continuous_prefill NO ^

