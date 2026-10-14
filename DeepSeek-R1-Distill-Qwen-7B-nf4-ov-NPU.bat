call agent.bat --model DeepSeek-R1-Distill-Qwen-7B-nf4-ov ^
 --detect_cycled_tool_call on ^
 --device NPU ^
 --max_prompt_len 61440 ^
 --npu_turbo YES ^
 --npu_compiler_type PLUGIN ^
 --npu_compilation_mode_params "optimization-level=2 performance-hint-override=latency" ^
 --npuw_llm_enable_block_based_kv_cache NO ^
 --npuw_llm_enable_continuous_prefill NO ^
 --npuw_llm_enable_prefix_caching YES ^
 --npuw_llm_generate_pyramid YES ^
 --npuw_llm_prefill_attention_hint PYRAMID ^
 --npuw_llm_generate_attention_hint PYRAMID ^
 --npuw_devices NPU,CPU ^
 --npuw_llm_propagate_slice_up YES ^

