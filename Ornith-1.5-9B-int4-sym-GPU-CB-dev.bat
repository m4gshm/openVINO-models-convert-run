call agent-dev.bat --model Ornith-1.5-9B-int4-sym-g128-awq ^
 --detect_cycled_tool_call on ^
 --kv_cache_precision u4 ^
 --scheduler_config_file .config/scheduler_config_cb.json ^
 --pipe CB
