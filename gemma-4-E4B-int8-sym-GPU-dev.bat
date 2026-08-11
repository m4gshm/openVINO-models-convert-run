call agent-dev.bat --model gemma-4-E4B-it-int8-sym ^
 --device GPU ^
 --pipe VLM ^
 --attention_backend PA ^
 --generate_config_file .config/generate_config_gemma4.json ^
 --chat_template_file .config/gemma4_chat_template.jinja
