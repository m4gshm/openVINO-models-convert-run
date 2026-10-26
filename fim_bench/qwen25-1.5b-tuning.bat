@echo off
rem ------------------------------------------------
rem Parameter sweep for Qwen2.5-Coder-1.5B.
rem See tuning-sweep.bat for the configurations and outputs.
rem The model is small: NPU alone is enough, no GPU cases.
rem ------------------------------------------------
set "FIM_NO_GPU=1"
call "%~dp0tuning-sweep.bat" qwen25-1.5b ^
    Qwen2.5-Coder-1.5B-nf4-sym-g-1-se-awq ^
    Qwen2.5-Coder-1.5B-int4-sym-g128-se-awq ^
    Qwen2.5-Coder-1.5B-int8-sym
