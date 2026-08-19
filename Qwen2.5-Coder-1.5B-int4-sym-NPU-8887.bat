call agent.bat --model Qwen2.5-Coder-1.5B-Instruct-int4-sym-g128-r1-se-awq ^
 --port 8887 ^
 --device NPU ^
 --npu_generate_hint BEST_PERF ^
 --npu_compiler_type DRIVER ^
 --attention_backend PA ^
 --max_prompt_len 4096 ^
 --attention_backend PA
