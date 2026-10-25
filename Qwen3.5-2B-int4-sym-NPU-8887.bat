setlocal
rem ------------------------------------------------
rem Debug logging of OpenVINO / NPUW plugin is set in agent/__main__.py:
rem   OPENVINO_LOG_LEVEL="4" (=LOG_DEBUG)
rem   OPENVINO_NPUW_LOG_LEVEL="DEBUG"
rem (NOTE: NPUW LOG_INFO/LOG_DEBUG are compiled out in stock wheels —
rem  see bug report bug-report-openvino-38246-npuw-strided-view.md)
rem ------------------------------------------------
rem PYRAMID prefill hint disabled: openvino#38246
rem (StridedViewTensor assert at make_tensor.cpp:213).
rem Restore the line below after PR openvino#38670 is merged:
rem   --npuw_llm_prefill_attention_hint PYRAMID ^
call agent.bat --model Qwen3.5-2B-int4-sym ^
 --port 8887 ^
 --device NPU ^
 --max_prompt_len 8192 ^
 --npu_turbo YES ^
 --npu_compiler_type PLUGIN ^
 --npuw_llm_enable_block_based_kv_cache NO ^
 --npuw_llm_enable_continuous_prefill NO ^
 --npuw_llm_enable_prefix_caching NO ^
 --npu_prefill_hint STATIC ^
 --npuw_devices NPU,GPU