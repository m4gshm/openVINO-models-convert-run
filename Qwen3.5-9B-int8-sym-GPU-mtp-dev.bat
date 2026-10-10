call agent-dev.bat ^
 --model Qwen3.5-9B-int8-sym ^
 --draft_model Qwen3.5-9B-int8-sym ^
 --detect_cycled_tool_call on ^
 --pipe VLM ^
 --kv_cache_precision u8 ^
 --generate_config_file .config/generate_config_mtp.json ^
 --scheduler_config_file .config/scheduler_config.json ^
 --mtp_mode on
