call agent-dev.bat --model gemma-4-E2B-it-qat-q4_0-unquantized-int4-sym-g128-se-awq ^
 --device GPU ^
 --generate_config_file .config/gemma4_generate_config.json ^
 --chat_template_file .config/gemma4_chat_template.jinja
