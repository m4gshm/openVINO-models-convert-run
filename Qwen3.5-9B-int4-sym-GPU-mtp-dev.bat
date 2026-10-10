call agent-dev.bat ^
 --model Qwen3.5-9B-int4-sym-g128-awq ^
 --draft_model Qwen3.5-9B-int4-sym-g128-awq ^
 --detect_cycled_tool_call on ^
 --pipe VLM ^
 --kv_cache_precision u4 ^
 --generate_config_file .config/generate_config_mtp.json ^
 --scheduler_config_file .config/scheduler_config.json ^
 --mtp_mode on
