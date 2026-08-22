call agent.bat --model LFM2-24B-A2B-int4-sym-g128 ^
 --device GPU ^
--generate_config_file .config/generate_config_lfm2.json ^
--chat_template_file .config/lmf25_fix_chat_template.jinja
