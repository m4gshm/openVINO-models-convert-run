set OPENVINO_DEV_NAME=openvino_genai_windows_2026.4.0.0.dev20260727_x86_64
call agent-dev.bat --model gemma-4-E4B-it-int8-sym ^
 --device GPU ^
 --pipe CB ^
 --generate_config_file .config/generate_config_gemma4.json ^
 --chat_template_file .config/gemma4.lm.studio.chat.template.jinja