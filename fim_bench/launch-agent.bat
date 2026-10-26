@echo off
setlocal
rem Generic agent launcher for the bench: launch-agent.bat <model_dir_name> <port> [extra agent args...]
rem All NPU defaults (device, prompt len, prefill hints, ...) live in the bench
rem (fim_bench/__main__.py) and arrive via extra args; override them there.
rem Example:
rem   launch-agent.bat Qwen2.5-Coder-3B-nf4-sym-g-1-se-awq 54321 --npu_prefill_hint DYNAMIC
set "DEVROOT=C:\alex\github\openvinotoolkit\dist-dev"
if not exist "%DEVROOT%\runtime\cmake\OpenVINOConfig.cmake" (
    echo ERROR: dev dist not found at %DEVROOT%
    exit /b 1
)
set "PYTHONPATH=%DEVROOT%\python"
set "OPENVINO_LIB_PATHS="
call "%DEVROOT%\setupvars.bat"

if "%~2"=="" (
    echo ERROR: port is required: launch-agent.bat ^<model^> ^<port^> [extra args...]
    exit /b 1
)

set "MODEL=%~1"
set "PORT=%~2"
set "EXTRA=%*"
call set "EXTRA=%%EXTRA:*%MODEL%=%%"
call set "EXTRA=%%EXTRA:*%PORT%=%%"

call "%~dp0..\agent.bat" ^
 --model %MODEL% ^
 --port %PORT% ^
 --npu_compiler_type PLUGIN ^
 --device NPU ^
 --npuw_devices NPU,GPU %EXTRA%
