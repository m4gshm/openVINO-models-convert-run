call agent-dev.bat --model gemma-4-E4B-it-int8-sym ^
 --device GPU ^
 --pipe VLM ^
 --generate_config_file .config/gemma4_generate_config.json ^
 --chat_template_file .config/gemma4_chat_template.jinja
