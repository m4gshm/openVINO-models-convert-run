call agent.bat --model Qwen3.5-2B-int4-sym ^
 --device NPU ^
 --max_prompt_len 8192 ^
 --npuw_devices NPU,GPU ^
 --npu_turbo NO
@REM  --max_prompt_len 65536 ^
