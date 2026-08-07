call agent-dev.bat --model gemma-4-E4B-it-int4-asym-g128-se-awq ^
 --device GPU ^
 --pipe VLM ^
 --generate_config_file .config/generate_config_gemma4.json ^
 --chat_template_file .config/gemma4_chat_template.jinja
