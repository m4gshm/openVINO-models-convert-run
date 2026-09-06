call agent-dev.bat --model Qwen3.6-35B-A3B-int4-sym-g128-awq ^
 --detect_cycled_tool_call off ^
 --kv_cache_precision u4 ^
 --generate_config_file .config/qwen_3.6_generate_config.json ^
 --scheduler_config_file .config/scheduler_config_cb.json ^
 --pipe CB
