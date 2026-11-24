@REM Baseline: sampled decoding, NO draft (pure autoregressive, prefix caching on).
@REM Reference speed: ~41 tok/s. The fastest option when no draft matches.
call agent-dev.bat ^
 --model Qwen3.6-35B-A3B-int4-sym-g128-awq ^
 --detect_cycled_tool_call on ^
 --pipe VLM ^
 --kv_cache_precision u4 ^
 --generate_config_file .config/generate_config_prompt_lookup_sampled.json ^
 --scheduler_config_file .config/scheduler_config_no_dsf.json
