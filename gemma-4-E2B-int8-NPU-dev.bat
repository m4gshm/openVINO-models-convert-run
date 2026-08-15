call agent-dev.bat --model gemma-4-E2B-it-int8-sym ^
 --device NPU ^
 --max_prompt_len 32768 ^
 --npu_generate_hint BEST_PERF ^
 --npu_compiler_type PLUGIN ^
 --generate_config_file .config/generate_config_gemma4_npu.json ^
 --chat_template_file .config/gemma4_chat_template.jinja
@REM  --max_prompt_len 49152 ^
