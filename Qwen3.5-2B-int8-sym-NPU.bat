call agent.bat --model Qwen3.5-2B-int8-sym ^
 --device NPU ^
 --max_prompt_len 4096 ^
 --npu_compiler_type DRIVER ^
 --npu_turbo YES
@REM  --max_prompt_len 65536 ^
