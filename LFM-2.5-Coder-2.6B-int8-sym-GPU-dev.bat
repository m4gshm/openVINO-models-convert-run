call agent-dev.bat --model LFM-2.5-Coder-2.6B-int8-sym ^
 --device GPU ^
 --chat_template_file .config/lmf25_fix_chat_template.jinja ^
 --generate_config_file .config/lfm2_generate_config.json
