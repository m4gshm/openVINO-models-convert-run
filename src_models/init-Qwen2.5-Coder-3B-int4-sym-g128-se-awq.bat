set MODEL_NAME=Qwen2.5-Coder-3B
set MODEL_DEVELOPER=Qwen
set MODEL_NAME_OUT=%MODEL_NAME%
set MODEL_PATH=./%MODEL_DEVELOPER%/%MODEL_NAME%
set OUTPUT_DIR=../models/%MODEL_NAME_OUT%

set GROUP_SIZE=128
set WEIGHT_FORMAT=int4

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

@REM Statistics collection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 128/128 • 0:27:43 • 0:00:00
@REM INFO:nncf:Statistics of the bitwidth distribution:
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM +===========================+=============================+========================================+
@REM | float                     | 0% (1 / 254)                | 0% (0 / 252)                           |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | int8_sym, per-channel     | 10% (1 / 254)               | 0% (0 / 252)                           |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | int4_sym, group size 128  | 90% (252 / 254)             | 100% (252 / 252)                       |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM Applying data-aware AWQ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 36/36 • 0:04:05 • 0:00:00
@REM Applying Scale Estimation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 253/253 • 0:10:39 • 0:00:00
@REM Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:16 • 0:00:00
