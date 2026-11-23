@REM DFlash2 block drafting on GPU. Draft: models/Qwen3.6-35B-A3B-dflash2-int4-sym-g128
@REM (convert via src_models/init-incoai-DFlash2-Qwen3.6-35B-A3B-int4-sym-g128.bat).
@REM Same sampler constraint as the eagle3 bench: greedy only (do_sample false in
@REM .config/generate_config_eagle3.json) while prefix caching is on.
call agent-dev.bat ^
 --model Qwen3.6-35B-A3B-int4-sym-g128-awq ^
 --draft_model Qwen3.6-35B-A3B-dflash2-int4-sym-g128 ^
 --dflash_mode on ^
 --detect_cycled_tool_call on ^
 --pipe VLM ^
 --kv_cache_precision u4 ^
 --generate_config_file .config/generate_config_eagle3.json ^
 --scheduler_config_file .config/scheduler_config.json
