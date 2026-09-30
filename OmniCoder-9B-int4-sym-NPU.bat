call agent.bat --model OmniCoder-9B-int4-sym-g128-se-awq ^
 --device NPU ^
 --max_prompt_len 65536 ^
 --npu_compiler_type DRIVER ^
 --npu_turbo YES ^
 --detect_cycled_tool_call on ^
 --kv_cache_precision u4 ^
 --generate_config_file .config/generate_config.json
