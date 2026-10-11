set MODEL_NAME=Qwen2.5-Coder-1.5B-Instruct
set MODEL_DEVELOPER=Qwen
set MODEL_NAME_OUT=%MODEL_NAME%
set MODEL_PATH=./%MODEL_DEVELOPER%/%MODEL_NAME%
set OUTPUT_DIR=../models/%MODEL_NAME_OUT%

set GROUP_SIZE=-1
set WEIGHT_FORMAT=nf4

optimum-cli export openvino ^
  --model %MODEL_PATH% ^
  --task text-generation-with-past ^
  --weight-format %WEIGHT_FORMAT% ^
  --group-size %GROUP_SIZE% ^
  --ratio 1.0 ^
  --trust-remote-code ^
  --dataset gsm8k ^
  --scale-estimation ^
  --awq ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-g%GROUP_SIZE%-r1-se-awq

pause

@REM  Statistics collection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 128/128 • 0:14:33 • 0:00:00
@REM  INFO:nncf:Statistics of the bitwidth distribution:
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM  +===========================+=============================+========================================+
@REM  | float                     | 0% (1 / 198)                | 0% (0 / 196)                           |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | int8_asym, per-channel    | 15% (1 / 198)               | 0% (0 / 196)                           |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | nf4, per-channel          | 85% (196 / 198)             | 100% (196 / 196)                       |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  Applying data-aware AWQ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 28/28 • 0:23:39 • 0:00:00
@REM  Applying Scale Estimation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 197/197 • 0:06:28 • 0:00:00
@REM  Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:10 • 0:00:00
