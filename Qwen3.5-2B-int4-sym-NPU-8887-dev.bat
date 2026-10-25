@echo off
setlocal
rem ------------------------------------------------
rem Dev-diagnostics launcher: runs the agent against the
rem developer-built OpenVINO/GenAI from openvinotoolkit\dist-dev
rem (ENABLE_NPU_DEBUG_CAPS=ON -> NPUW LOG_INFO/LOG_DEBUG enabled).
rem Env vars for OpenVINO logging are already set in agent/__main__.py:
rem   OPENVINO_LOG_LEVEL="4" (=LOG_DEBUG)
rem   OPENVINO_NPUW_LOG_LEVEL="DEBUG"
rem See bug-report-openvino-38246-npuw-strided-view.md
rem ------------------------------------------------
set "DEVROOT=C:\alex\github\openvinotoolkit\dist-dev"

if not exist "%DEVROOT%\runtime\cmake\OpenVINOConfig.cmake" (
    echo ERROR: dev dist not found at %DEVROOT%
    echo Build it first:  C:\alex\github\openvinotoolkit\build.bat dev fast
    exit /b 1
)

set "PYTHONPATH=%DEVROOT%\python"
set "OPENVINO_LIB_PATHS="
call "%DEVROOT%\setupvars.bat"

call "%~dp0agent.bat" ^
 --model Qwen3.5-2B-int4-sym ^
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
