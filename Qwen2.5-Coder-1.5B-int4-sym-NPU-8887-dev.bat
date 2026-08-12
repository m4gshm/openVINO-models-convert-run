call agent-dev.bat --model Qwen2.5-Coder-1.5B-Instruct-int4-sym-g128-r1-se-awq ^
 --device NPU --max_prompt_len 16384 --port 8887 ^
 --npu_generate_hint BEST_PERF --npu_prefill_hint DYNAMIC --npu_turbo YES
