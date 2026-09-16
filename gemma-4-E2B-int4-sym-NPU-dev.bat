call agent-dev.bat --model gemma-4-E2B-it-int4-sym-g128-se-awq ^
 --device NPU --npu_compiler_type PLUGIN --max_prompt_len 32768 ^
 --generate_config_file .config/gemma4_generate_config.json ^
 --npu_turbo YES ^
 --chat_template_file .config/gemma4_chat_template.jinja
