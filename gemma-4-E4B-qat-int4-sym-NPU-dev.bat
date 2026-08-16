call agent-dev.bat --model gemma-4-E4B-it-qat-q4_0-unquantized-int4-sym-g128 ^
 --device NPU ^
 --max_prompt_len 32768 ^
 --cache_precision u4 ^
 --npu_generate_hint BEST_PERF ^
 --npu_compiler_type PLUGIN ^
 --attention_backend PA ^
 --generate_config_file .config/generate_config_gemma4_npu.json ^
 --chat_template_file .config/gemma4_chat_template.jinja
@REM  --max_prompt_len 49152 ^
