@REM EAGLE3 chain drafting on GPU. Draft: models/Qwen3.6-35B-A3B-eagle3-caption-tok-int4-sym-g128
@REM (convert via src_models/init-Dogacel-specdrift-Qwen3.6-35B-A3B-eagle3-caption-tok-int4-sym-g128.bat).
@REM do_sample stays false: with prefix caching on, the linear-attention verifier of this
@REM hybrid-attention model asserts on multinomial sampling for ANY draft strategy
@REM (ContinuousBatchingImpl::_validate_linear_verifier_constraints in openvino.genai).
@REM To experiment with sampling: flip do_sample in .config/generate_config_eagle3.json
@REM AND set "enable_prefix_caching": false in .config/scheduler_config.json.
call agent-dev.bat ^
 --model Qwen3.6-35B-A3B-int4-sym-g128-awq ^
 --draft_model Qwen3.6-35B-A3B-eagle3-caption-tok-int4-sym-g128 ^
 --eagle3_mode on ^
 --detect_cycled_tool_call on ^
 --pipe CB ^
 --kv_cache_precision u4 ^
 --generate_config_file .config/generate_config_eagle3.json ^
 --scheduler_config_file .config/scheduler_config.json



