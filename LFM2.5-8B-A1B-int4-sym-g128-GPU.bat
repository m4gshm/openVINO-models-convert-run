call agent.bat --model LFM2.5-8B-A1B-int4-sym-g128 ^
 --device GPU ^
 --generate_config_file .config/lfm2_generate_config.json ^
 --chat_template_file .config/lmf25_fix_chat_template.jinja
