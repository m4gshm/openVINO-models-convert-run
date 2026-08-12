call agent.bat --model Vibe-Coding-Claude-Fable-5-int4-sym-g128-r1-se-awq ^
 --device NPU --max_prompt_len 32768 --port 8888 ^
 --npu_generate_hint BEST_PERF --npu_prefill_hint DYNAMIC --npu_turbo YES ^
 --attention_backend SDPA
