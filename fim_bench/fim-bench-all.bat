@echo off
setlocal
rem ------------------------------------------------
rem FIM benchmark across all models under test.
rem Starts/stops the agent server per model via
rem _bench_model.bat and writes a markdown report.
rem
rem Usage:
rem   fim-bench-all.bat              - default model set, report to reports\
rem   fim-bench-all.bat <report.md>  - custom report path
rem ------------------------------------------------
set "MODELS=Qwen2.5-Coder-3B-nf4-sym-g-1-se-awq Qwen3.5-2B-int4-sym LFM-2.5-Coder-2.6B-nf4-sym-g-1-se-awq Mellum-4b-dpo-all-nf4-sym-g-1-se-awq"

set "REPORT=%~1"
if "%REPORT%"=="" (
    for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmm"') do set "STAMP=%%i"
    set "REPORT=%~dp0..\reports\fim-%STAMP%.md"
)

if not exist "%~dp0..\.venv\Scripts\python.exe" (
    echo ERROR: venv not found at %~dp0..\.venv
    exit /b 1
)

echo === FIM bench, models: %MODELS%
echo === report: %REPORT%

set "ARGS="
for %%M in (%MODELS%) do call :add %%M

"%~dp0..\.venv\Scripts\python.exe" -m fim_bench %ARGS% --launch --report "%REPORT%"
if errorlevel 1 echo === WARNING: bench reported errors

echo === summary: %~dp0..\reports\fim-summary.md
"%~dp0..\.venv\Scripts\python.exe" -m fim_bench.summary --results-log "%~dp0..\reports\fim-results.jsonl" --out "%~dp0..\reports\fim-summary.md"
goto :done

:add
set "ARGS=%ARGS% --model %1"
exit /b 0

:done
echo === finished, report: %REPORT%
endlocal
