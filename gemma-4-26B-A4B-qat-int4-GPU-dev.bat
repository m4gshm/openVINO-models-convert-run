call agent-dev.bat --model gemma-4-26B-A4B-it-qat-q4_0-unquantized-int4-sym-g64 ^
 --device GPU ^
 --pipe VLM ^
 --generate_config_file .config/gemma4_generate_config.json ^
 --chat_template_file .config/gemma4_chat_template.jinja
