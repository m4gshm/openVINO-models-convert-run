call agent-dev.bat --model LFM2.5-8B-A1B-int8-asym ^
 --device GPU ^
 --generate_config_file .config/lfm2_generate_config.json ^
 --chat_template_file .config/lmf25_fix_chat_template.jinja
