call agent-dev.bat --model lfm2.5-2.6b-fable5-coding-agent-int8-sym ^
 --device GPU ^
 --generate_config_file .config/lfm2_generate_config.json ^
 --chat_template_file .config/lmf25_fix_chat_template.jinja
