call agent.bat --model Qwen2.5-Coder-1.5B-Instruct-int8-sym ^
 --device NPU --max_prompt_len 16384 --port 8887 ^
 --generate_hint BEST_PERF --npu_prefill_hint DYNAMIC --npu_turbo NO
