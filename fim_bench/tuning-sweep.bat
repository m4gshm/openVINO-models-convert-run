@echo off
setlocal enabledelayedexpansion
rem ------------------------------------------------
rem Universal parameter sweep for the FIM benchmark.
rem
rem Runs every model through the same set of server
rem configurations (prefill hints, chunk sizes, KV cache,
rem max_prompt_len) and prints a final pivot table.
rem
rem Usage:
rem   tuning-sweep.bat <sweep-name> <model_dir_name> [<model_dir_name>...]
rem
rem Example:
rem   tuning-sweep.bat qwen25-3b Qwen2.5-Coder-3B-nf4-sym-g-1-se-awq ...
rem
rem Outputs:
rem   reports\tuning-<sweep-name>-<stamp>.jsonl  raw case results
rem   reports\tuning-<sweep-name>-<stamp>.md     final comparison table
rem ------------------------------------------------
if "%~1"=="" (
    echo Usage: tuning-sweep.bat ^<sweep-name^> ^<model_dir_name^> [^<model_dir_name^>...]
    exit /b 1
)
set "SWEEP=%~1"
shift

set "MODELS="
:collect_models
if not "%~1"=="" (
    set "MODELS=!MODELS! %~1"
    shift
    goto :collect_models
)
if "%MODELS%"=="" (
    echo ERROR: at least one model dir name is required
    exit /b 1
)

set "PY=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo ERROR: venv not found at %~dp0..\.venv
    exit /b 1
)

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmm"') do set "STAMP=%%i"
set "RESULTS=%~dp0..\reports\tuning-%SWEEP%-%STAMP%.jsonl"
set "SUMMARY=%~dp0..\reports\tuning-%SWEEP%-%STAMP%.md"
if not exist "%~dp0..\reports" mkdir "%~dp0..\reports"
type nul > "%RESULTS%"

echo === parameter sweep "%SWEEP%"
echo === models:%MODELS%
echo === results and summary paths are printed at the end (clickable)

set "NMODELS=0"
for %%M in (%MODELS%) do set /a NMODELS+=1
rem keep TOTAL_CONFIGS in sync with the number of "call :run" lines below
set /a TOTAL_RUNS=NMODELS * 11
if defined FIM_NO_GPU set /a TOTAL_RUNS=NMODELS * 9
set /a RUN_NO=0

call :run dyn-default        --npu_prefill_hint DYNAMIC
call :run dyn-hfa            --npu_prefill_hint DYNAMIC --npuw_llm_prefill_attention_hint HFA
call :run dyn-pyramid        --npu_prefill_hint DYNAMIC --npuw_llm_prefill_attention_hint PYRAMID
call :run dyn-pyramid-chunk512 --npu_prefill_hint DYNAMIC --npuw_llm_prefill_attention_hint PYRAMID --npuw_llm_prefill_chunk_size 512
call :run dyn-hfa-chunk512   --npu_prefill_hint DYNAMIC --npuw_llm_prefill_attention_hint HFA --npuw_llm_prefill_chunk_size 512
call :run dyn-chunk128       --npu_prefill_hint DYNAMIC --npuw_llm_prefill_chunk_size 128
call :run dyn-chunk256       --npu_prefill_hint DYNAMIC --npuw_llm_prefill_chunk_size 256
call :run dyn-chunk512       --npu_prefill_hint DYNAMIC --npuw_llm_prefill_chunk_size 512
call :run static-maxlen2048  --max_prompt_len 2048

rem GPU cases are skipped when the wrapper opts out (small models
rem that have no use for the GPU device comparison).
if not defined FIM_NO_GPU (
    call :run gpu-default        --device GPU --npuw_devices GPU
    call :run gpu-maxlen2048     --device GPU --npuw_devices GPU --max_prompt_len 2048
)

echo.
echo === all runs done, building the summary table...
"%PY%" -m fim_bench.summary --results-log "%RESULTS%" --out "%SUMMARY%"
if errorlevel 1 (
    echo === ERROR: summary failed
    exit /b 1
)
endlocal
exit /b 0

:run
rem %1 = label, %2.. = extra agent args for this configuration
set "LABEL=%~1"
shift
set "MODE_ARGS="
:collect
if not "%~1"=="" (
    set "MODE_ARGS=!MODE_ARGS! --server-arg=%~1"
    shift
    goto :collect
)
for %%M in (%MODELS%) do (
    set /a RUN_NO+=1
    echo.
    echo === [!RUN_NO!/!TOTAL_RUNS!]. !LABEL! %%M
    %PY% -m fim_bench --model %%M --label %LABEL% --warmup --launch !MODE_ARGS! --results-log "%RESULTS%"
    if errorlevel 1 (
        echo === [!RUN_NO!/!TOTAL_RUNS!] bench FAILED for %%M ^(!LABEL!^), see the log path printed above
    )
)
exit /b 0
