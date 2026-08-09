call agent.bat --model gemma-4-12B-it-int8-asym ^
 --device GPU ^
 --pipe VLM ^
 --attention_backend PA ^
 --generate_config_file .config/generate_config_gemma4.json ^
 --chat_template_file .config/gemma4_chat_template.jinja
