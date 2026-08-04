call agent-dev.bat --model gemma-4-E4B-it-int8-asym ^
 --device GPU ^
 --pipe CB ^
 --generate_config_file .config/generate_config_gemma4.json ^
 --chat_template_file .config/gemma4.lm.studio.chat.template.jinja
