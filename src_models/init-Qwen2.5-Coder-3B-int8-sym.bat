set MODEL_NAME=Qwen2.5-Coder-3B
set MODEL_DEVELOPER=Qwen
set MODEL_NAME_OUT=%MODEL_NAME%
set MODEL_PATH=./%MODEL_DEVELOPER%/%MODEL_NAME%
set OUTPUT_DIR=../models/%MODEL_NAME_OUT%

set WEIGHT_FORMAT=int8

optimum-cli export openvino ^
  --model %MODEL_PATH% ^
  --task text-generation-with-past ^
  --weight-format %WEIGHT_FORMAT% ^
  --sym ^
  --trust-remote-code ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-sym

pause

@REM INFO:nncf:Statistics of the bitwidth distribution:
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM +===========================+=============================+========================================+
@REM | float                     | 0% (1 / 254)                | 0% (0 / 253)                           |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | int8_sym, per-channel     | 100% (253 / 254)            | 100% (253 / 253)                       |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:18 • 0:00:00
