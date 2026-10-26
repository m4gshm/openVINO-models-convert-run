@echo off
rem ------------------------------------------------
rem Parameter sweep for the Mellum-4b family:
rem   Mellum-4b-base-nf4-sym-g-1-se-awq        (pretrained base, nf4)
rem   Mellum-4b-dpo-all-nf4-sym-g-1-se-awq     (DPO variant, nf4)
rem   Mellum-4b-base-int4-sym-g128-se-awq      (base, int4 sym g128;
rem                                             built by
rem                                             mellum-4b-base-init-int4-sym-g128.bat
rem                                             on first run)
rem See tuning-sweep.bat for the configurations and outputs.
rem The int4-sym-g128 version must be converted in advance:
rem   src_models\init-Mellum-4b-base-int4-sym-g128-se-awq.bat
rem The sweep includes the GPU device cases (gpu-default, gpu-maxlen2048).
rem ------------------------------------------------
call "%~dp0tuning-sweep.bat" mellum-4b ^
    Mellum-4b-base-nf4-sym-g-1-se-awq ^
    Mellum-4b-dpo-all-nf4-sym-g-1-se-awq ^
    Mellum-4b-base-int4-sym-g128-se-awq
