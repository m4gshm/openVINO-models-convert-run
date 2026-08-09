call agent.bat --model gemma-4-12B-it-int4-asym-g128-se-awq ^
 --device GPU ^
 --pipe VLM ^
 --attention_backend PA ^
 --generate_config_file .config/generate_config_gemma4.json ^
 --chat_template_file .config/gemma4_chat_template.jinja
