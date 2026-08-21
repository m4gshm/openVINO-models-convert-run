call agent.bat --model LFM2.5-8B-A1B-int8-sym ^
 --device GPU ^
--generate_config_file .config/generate_config_lfm2.json ^
--chat_template_file .config/lmf25_fix_chat_template.jinja
