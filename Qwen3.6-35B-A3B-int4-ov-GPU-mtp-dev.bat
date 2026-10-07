call agent-dev.bat --model Qwen3.6-35B-A3B-int4-ov ^
 --draft_model Qwen3.6-35B-A3B-int4-ov ^
 --detect_cycled_tool_call on ^
 --pipe VLM ^
 --kv_cache_precision u4 ^
 --generate_config_file .config/generate_config_mtp.json ^
 --scheduler_config_file .config/scheduler_config.json ^
 --mtp_mode on
