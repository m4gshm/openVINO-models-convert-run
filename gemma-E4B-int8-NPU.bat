call agent.bat --model gemma-4-E4B-it-int8-sym ^
 --device NPU ^
 --max_prompt_len 49152 ^
 --generate_config_file .config/generate_config_gemma4_npu.json ^
 --chat_template_file .config/gemma4.lm.studio.chat.template.jinja