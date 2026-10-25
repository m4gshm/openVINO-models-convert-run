@echo off
setlocal
rem Generic bench launcher: _bench_model.bat <model_dir_name> [extra agent args...]
rem Extra args are appended verbatim, e.g.:
rem   _bench_model.bat LFM-2.5-Coder-2.6B-nf4-sym-g-1-se-awq --npu_prefill_hint DYNAMIC --npuw_llm_prefill_attention_hint PYRAMID
set "DEVROOT=C:\alex\github\openvinotoolkit\dist-dev"
if not exist "%DEVROOT%\runtime\cmake\OpenVINOConfig.cmake" (
    echo ERROR: dev dist not found at %DEVROOT%
    exit /b 1
)
set "PYTHONPATH=%DEVROOT%\python"
set "OPENVINO_LIB_PATHS="
call "%DEVROOT%\setupvars.bat"

set "EXTRA=%*"
if not "%~1"=="" call set "EXTRA=%%EXTRA:*%~1=%%"

call "%~dp0..\agent.bat" ^
 --model %~1 ^
 --port 8887 ^
 --device NPU ^
 --max_prompt_len 8192 ^
 --npu_turbo YES ^
 --npu_compiler_type PLUGIN ^
 --npuw_llm_enable_block_based_kv_cache NO ^
 --npuw_llm_enable_continuous_prefill NO ^
 --npu_prefill_hint STATIC ^
 --npuw_devices NPU,GPU %EXTRA%
