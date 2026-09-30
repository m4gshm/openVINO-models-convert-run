call agent.bat --model LFM2-24B-A2B-int8-sym ^
 --device GPU ^
 --generate_config_file .config/lfm2_generate_config.json ^
 --chat_template_file .config/lmf25_fix_chat_template.jinja
