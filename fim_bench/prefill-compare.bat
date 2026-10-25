@echo off
setlocal enabledelayedexpansion
rem ------------------------------------------------
rem Compare prefill modes (STATIC / DYNAMIC / HFA / PYRAMID)
rem for one model through the FIM benchmark.
rem
rem Usage:
rem   prefill-compare.bat <model_dir_name> [extra server args...]
rem Example:
rem   prefill-compare.bat Qwen2.5-Coder-3B-nf4-sym-g-1-se-awq
rem ------------------------------------------------
set "MODEL=%~1"
if "%MODEL%"=="" (
    echo Usage: prefill-compare.bat ^<model_dir_name^> [extra server args...]
    exit /b 1
)
shift

rem collect extra server args (all args after the model name)
set "EXTRA="
:collect
if not "%~1"=="" (
    set "EXTRA=!EXTRA! --server-arg=%~1"
    shift
    goto :collect
)

set "PY=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo ERROR: venv not found at %~dp0..\.venv
    exit /b 1
)

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmm"') do set "STAMP=%%i"
set "SUMMARY=%~dp0..\reports\prefill-compare-%MODEL%-%STAMP%.md"
if not exist "%~dp0..\reports" mkdir "%~dp0..\reports"

echo === prefill mode comparison for %MODEL%
echo === summary: %SUMMARY%
type nul > "%SUMMARY%"

call :run STATIC
call :run DYN-DEFAULT --npu_prefill_hint DYNAMIC
call :run DYN-HFA --npu_prefill_hint DYNAMIC --npuw_llm_prefill_attention_hint HFA
call :run DYN-PYRAMID --npu_prefill_hint DYNAMIC --npuw_llm_prefill_attention_hint PYRAMID

echo.
echo === done, summary in %SUMMARY%
echo === combined summary: %~dp0..\reports\fim-summary.md
"%~dp0..\.venv\Scripts\python.exe" -m fim_bench.summary --results-log "%~dp0..\reports\fim-results.jsonl" --out "%~dp0..\reports\fim-summary.md"
endlocal
exit /b 0

:run
rem %1 = mode label, %2..%9 = extra server args for this mode
set "MODE=%~1"
set "MODE_ARGS="
shift
:collect_mode
if not "%~1"=="" (
    set "MODE_ARGS=!MODE_ARGS! --server-arg=%~1"
    shift
    goto :collect_mode
)

set "REPORT=%~dp0..\reports\prefill-%MODEL%-%MODE%.md"
echo.
echo === [%MODE%] running bench...
%PY% -m fim_bench --model %MODEL% --label %MODE% --warmup --launch !MODE_ARGS! !EXTRA! --report "%REPORT%"
if errorlevel 1 (
    echo === [%MODE%] bench FAILED, see logs\bench-%MODEL%.log
    echo ## %MODE%: BENCH FAILED>> "%SUMMARY%"
    exit /b 0
)
echo ## %MODE%>> "%SUMMARY%"
echo ```>> "%SUMMARY%"
type "%REPORT%" >> "%SUMMARY%"
echo.>> "%SUMMARY%"
echo ```>> "%SUMMARY%"
exit /b 0
