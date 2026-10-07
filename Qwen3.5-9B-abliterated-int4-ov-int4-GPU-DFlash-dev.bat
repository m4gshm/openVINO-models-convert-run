call agent-dev.bat --model Qwen3.5-9B-abliterated-int4-ov ^
 --draft_model Qwen3.5-9B-abliterated-DFlash-int4-ov ^
 --detect_cycled_tool_call on ^
 --pipe LLM ^
 --kv_cache_precision u4 ^
 --generate_config_file .config/generate_config_draft.json ^
 --scheduler_config_file .config/scheduler_config.json ^
 --dflash_mode on
