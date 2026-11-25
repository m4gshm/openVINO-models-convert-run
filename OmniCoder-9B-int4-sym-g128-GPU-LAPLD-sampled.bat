@REM OmniCoder-9B (dense Qwen3.5, GPU) with prompt lookup drafting + sampling.
@REM Same SD relaxes as the Qwen3.6 LAPLD-sampled starter.
call agent-dev.bat ^
 --model OmniCoder-9B-int4-sym-g128-se-awq ^
 --device GPU ^
 --detect_cycled_tool_call on ^
 --generate_config_file .config/generate_config_prompt_lookup_sampled.json ^
 --scheduler_config_file .config/scheduler_config_no_dsf.json ^
 --prompt_lookup on
