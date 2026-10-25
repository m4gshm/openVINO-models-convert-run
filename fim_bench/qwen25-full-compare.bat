@echo off
setlocal enabledelayedexpansion
rem ------------------------------------------------
rem Full comparison: every local Qwen2.5-Coder version
rem x every prefill mode (STATIC / DYN-DEFAULT / DYN-HFA / DYN-PYRAMID),
rem via prefill-compare.bat. Results accumulate in
rem reports\prefill-compare-<model>-<stamp>.md per model.
rem
rem Usage:
rem   qwen25-full-compare.bat
rem   qwen25-full-compare.bat <extra server args...>   e.g. --npu_turbo NO
rem ------------------------------------------------
set "MODELS=Qwen2.5-Coder-0.5B-nf4-sym-g-1-se-awq Qwen2.5-Coder-1.5B-nf4-sym-g-1-se-awq Qwen2.5-Coder-3B-nf4-sym-g-1-se-awq Qwen2.5-Coder-3B-int4-sym-g128-se-awq Qwen2.5-Coder-3B-int8-sym"

set "EXTRA=%*"

echo === Full Qwen2.5-Coder comparison
echo === models: %MODELS%
echo === extra args: %EXTRA%
echo.

for %%M in (%MODELS%) do (
    echo ============================================================
    echo === MODEL: %%M
    echo ============================================================
    call "%~dp0prefill-compare.bat" %%M %EXTRA%
    if errorlevel 1 (
        echo === WARNING: comparison for %%M failed, continuing
    )
)

echo.
echo === all done. Reports are in %~dp0..\reports\prefill-compare-*.md
endlocal
