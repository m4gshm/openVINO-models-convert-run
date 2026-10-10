call agent-dev.bat --model gemma-4-26b-a4b-it-int4-ov ^
 --device GPU ^
 --pipe VLM ^
 --generate_config_file .config/gemma4_generate_config.json ^
 --chat_template_file .config/gemma4_chat_template.jinja
