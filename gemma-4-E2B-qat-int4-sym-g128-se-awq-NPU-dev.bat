call agent-dev.bat --model gemma-4-E4B-it-qat-q4_0-unquantized-int4-sym-g128-se-awq ^
 --device NPU ^
 --npu_generate_hint BEST_PERF ^
 --npu_compiler_type PLUGIN ^
 --attention_backend PA ^
 --max_prompt_len 32768 ^
 --generate_config_file .config/generate_config_gemma4_npu.json ^
 --chat_template_file .config/gemma4_chat_template.jinja
