call agent.bat --model Qwen2.5-Coder-1.5B-Instruct-int4-sym-g128-r1-se-awq ^
 --device NPU ^
 --max_prompt_len 4096 ^
 --port 8887 ^
 --npu_generate_hint BEST_PERF ^
 --npu_compiler_type DRIVER ^
 --attention_backend PA
