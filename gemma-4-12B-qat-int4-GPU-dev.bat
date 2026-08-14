call agent-dev.bat --model gemma-4-12b-it-qat-q4_0-unquantized-int4-sym-g128 ^
 --device GPU ^
 --pipe VLM ^
 --generate_config_file .config/generate_config_gemma4.json ^
 --chat_template_file .config/gemma4_chat_template.jinja
