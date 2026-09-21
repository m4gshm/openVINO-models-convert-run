call agent-dev.bat --model Qwen3.6-35B-A3B-int4-sym-g128-awq ^
 --detect_cycled_tool_call on ^
 --kv_cache_precision u4 ^
 --generate_config_file .config/generate_config.json ^
 --scheduler_config_file .config/scheduler_config_sa.json
