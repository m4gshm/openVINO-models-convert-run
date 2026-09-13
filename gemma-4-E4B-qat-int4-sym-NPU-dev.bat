call agent-dev.bat --model gemma-4-E4B-it-qat-q4_0-unquantized-int4-sym-g128-se-awq ^
 --device NPU ^
 --npu_generate_hint BEST_PERF ^
 --npu_compiler_type PLUGIN ^
 --max_prompt_len 49152 ^
 --generate_config_file .config/gemma4_generate_config_npu.json ^
 --chat_template_file .config/gemma4_chat_template.jinja
