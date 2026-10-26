@echo off
rem ------------------------------------------------
rem Parameter sweep for every local Qwen2.5-Coder-3B version:
rem See tuning-sweep.bat for the configurations and outputs.
rem The sweep includes the GPU device cases (gpu-default, gpu-maxlen2048).
rem ------------------------------------------------
call "%~dp0tuning-sweep.bat" qwen25-3b ^
    Qwen2.5-Coder-3B-nf4-sym-g-1-se-awq ^
    Qwen2.5-Coder-3B-int4-sym-g128-se-awq ^
    Qwen2.5-Coder-3B-int8-sym