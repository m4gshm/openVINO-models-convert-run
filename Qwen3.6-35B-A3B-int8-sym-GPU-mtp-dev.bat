call agent-dev.bat --model Qwen3.6-35B-A3B-int8-sym ^
 --draft_model Qwen3.6-35B-A3B-int8-sym ^
 --detect_cycled_tool_call on ^
 --kv_cache_precision u8 ^
 --generate_config_file .config/generate_config_mtp.json ^
 --scheduler_config_file .config/scheduler_config.json ^
 --mtp_mode on
