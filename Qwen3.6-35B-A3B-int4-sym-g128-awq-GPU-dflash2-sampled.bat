@REM DFlash2 block drafting + sampling (upstream fbcd4428 supports sampled DFlash drafting).
@REM Draft: models/Qwen3.6-35B-A3B-dflash2-int4-sym-g128. Prefix caching on.
call agent-dev.bat ^
 --model Qwen3.6-35B-A3B-int4-sym-g128-awq ^
 --draft_model Qwen3.6-35B-A3B-dflash2-int4-sym-g128 ^
 --dflash_mode on ^
 --detect_cycled_tool_call on ^
 --pipe VLM ^
 --kv_cache_precision u4 ^
 --generate_config_file .config/generate_config_dflash2_sampled.json ^
 --scheduler_config_file .config/scheduler_config.json
