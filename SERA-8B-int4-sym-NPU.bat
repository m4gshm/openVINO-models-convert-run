call agent.bat --model SERA-8B-int4-sym-g128-se-awq ^
 --device NPU ^
 --max_prompt_len 32768 ^
 --npu_generate_hint BEST_PERF ^
 --npu_compiler_type PLUGIN
