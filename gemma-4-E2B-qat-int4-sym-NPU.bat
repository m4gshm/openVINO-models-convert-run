call agent.bat --model gemma-4-E2B-it-qat-q4_0-unquantized-int4-sym-g128-se-awq ^
 --detect_cycled_tool_call off ^
 --device NPU ^
 --max_prompt_len 49152 ^
 --npu_turbo YES ^
 --npu_compiler_type PLUGIN ^
 --npuw_llm_enable_block_based_kv_cache NO ^
 --npuw_llm_enable_continuous_prefill NO ^
 --npuw_llm_enable_prefix_caching YES ^
 --npu_prefill_attention_hint HFA ^
 --generate_config_file .config/gemma4_generate_config_npu.json ^
 --chat_template_file .config/gemma4_chat_template.jinja

@REM  --npu_generate_attention_hint PYRAMID ^