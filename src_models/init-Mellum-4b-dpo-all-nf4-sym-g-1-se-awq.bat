set MODEL_NAME=Mellum-4b-dpo-all
set MODEL_DEVELOPER=JetBrains
set MODEL_NAME_OUT=%MODEL_NAME%
set MODEL_PATH=./%MODEL_DEVELOPER%/%MODEL_NAME%
set OUTPUT_DIR=../models/%MODEL_NAME_OUT%

set GROUP_SIZE=-1
set WEIGHT_FORMAT=nf4

optimum-cli export openvino ^
  --model %MODEL_PATH% ^
  --task text-generation-with-past ^
  --weight-format %WEIGHT_FORMAT% ^
  --backup-precision int8_sym ^
  --sym ^
  --group-size %GROUP_SIZE% ^
  --trust-remote-code ^
  --dataset gsm8k ^
  --scale-estimation ^
  --awq ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-sym-g%GROUP_SIZE%-se-awq

pause

@REM INFO:nncf:Statistics of the bitwidth distribution:
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM +===========================+=============================+========================================+
@REM | float                     | 0% (1 / 213)                | 0% (0 / 210)                           |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | int8_sym, per-channel     | 15% (2 / 213)               | 0% (0 / 210)                           |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | nf4, per-channel          | 85% (210 / 213)             | 100% (210 / 210)                       |
@REM +---------------------------+-----------------------------+----------------------------------------+
