set MODEL_NAME=gemma-4-E2B-it
set MODEL_DEVELOPER=google
set MODEL_NAME_OUT=%MODEL_NAME%
set MODEL_PATH=./%MODEL_DEVELOPER%/%MODEL_NAME%
set OUTPUT_DIR=../models/%MODEL_NAME_OUT%

set WEIGHT_FORMAT=int8

optimum-cli export openvino ^
  --model %MODEL_PATH% ^
  --task image-text-to-text ^
  --weight-format %WEIGHT_FORMAT% ^
  --trust-remote-code ^
  --sym ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-sym

pause

@REM  INFO:nncf:Statistics of the bitwidth distribution:
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM  +===========================+=============================+========================================+
@REM  | float                     | 0% (2 / 279)                | 0% (0 / 277)                           |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | int8_sym, per-channel     | 100% (277 / 279)            | 100% (277 / 277)                       |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:09 • 0:00:00
@REM  INFO:nncf:Statistics of the bitwidth distribution:
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM  +===========================+=============================+========================================+
@REM  | int8_sym, per-channel     | 100% (1 / 1)                | 100% (1 / 1)                           |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:00 • 0:00:00
@REM  INFO:nncf:Statistics of the bitwidth distribution:
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM  +===========================+=============================+========================================+
@REM  | float                     | 0% (2 / 118)                | 0% (0 / 116)                           |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | int8_sym, per-channel     | 100% (116 / 118)            | 100% (116 / 116)                       |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:02 • 0:00:00
@REM  INFO:nncf:Statistics of the bitwidth distribution:
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM  +===========================+=============================+========================================+
@REM  | int8_sym, per-channel     | 100% (1 / 1)                | 100% (1 / 1)                           |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:03 • 0:00:00
@REM  INFO:nncf:Statistics of the bitwidth distribution:
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM  +===========================+=============================+========================================+
@REM  | int8_sym, per-channel     | 100% (137 / 137)            | 100% (137 / 137)                       |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:02 • 0:00:00

