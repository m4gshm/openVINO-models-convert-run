call agent.bat --model gemma-4-E2B-it-int4-sym-g128-se-awq --device NPU ^
 --max_prompt_len 49152 ^
 --attention_backend SDPA ^
 --generate_config_file .config/generate_config_gemma4_npu.json ^
 --chat_template_file .config/gemma4.lm.studio.chat.template.jinja