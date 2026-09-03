call agent-dev.bat --model Qwen3.8-9B-int4-sym-g128-se-awq ^
 --detect_cycled_tool_call off ^
 --kv_cache_precision u4 ^
 --pipe CB ^
 --attention_backend PA
