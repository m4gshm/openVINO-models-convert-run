call agent.bat --model gemma-4-E2B-it-qat-q4_0-unquantized-int4-sym-g128-se-awq ^
 --detect_cycled_tool_call off ^
 --device GPU ^
 --generate_config_file .config/gemma4_generate_config_npu.json ^
 --chat_template_file .config/gemma4_chat_template.jinja
