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

@REM Make the dev dist authoritative: OPENVINO_LIB_PATHS may hold stale paths
@REM from a previously called bin\dist\setupvars.bat (agent-dev.bat). The
@REM openvino python package resolves openvino.dll via OPENVINO_LIB_PATHS in
@REM order, and a stale bin\dist (release) entry shadows the dev runtime ->
@REM "versions_compatible" mismatch (23587 bindings vs 23452 runtime).
set "PYTHONPATH=%DEVROOT%\python"
set "OPENVINO_LIB_PATHS="
call "%DEVROOT%\setupvars.bat"

@REM GQA-TRACE logging ([GQA-TRACE] lines in stdout, dev build only):
@REM shows tensor names/shapes of every KV-cache copy in the GQA wrapper.
set "OPENVINO_NPUW_GQA_LOG=1"

call "%~dp0agent.bat" ^
 --model LFM-2.5-Coder-2.6B-nf4-sym-g-1-se-awq ^
 --port 8887 ^
 --device NPU ^
 --max_prompt_len 8192 ^
 --npu_turbo YES ^
 --npu_compiler_type PLUGIN ^
 --npuw_llm_enable_block_based_kv_cache NO ^
 --npuw_llm_enable_continuous_prefill NO ^
 --npuw_llm_enable_prefix_caching YES ^
 --npu_prefill_hint STATIC ^
 --npuw_devices NPU,GPU
