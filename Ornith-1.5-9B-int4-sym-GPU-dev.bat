call agent-dev.bat --model Ornith-1.5-9B-int4-sym-g128-awq ^
 --detect_cycled_tool_call off ^
 --kv_cache_precision u4 ^
 --pipe VLM ^
 --attention_backend PA
