@REM transformers==5.7.0
set MODEL_NAME=LFM2-24B-A2B
set MODEL_DEVELOPER=LiquidAI
set MODEL_NAME_OUT=%MODEL_NAME%
set MODEL_PATH=./%MODEL_DEVELOPER%/%MODEL_NAME%
set OUTPUT_DIR=../models/%MODEL_NAME_OUT%

set GROUP_SIZE=128
set WEIGHT_FORMAT=int4

optimum-cli export openvino ^
  --model %MODEL_PATH% ^
  --task text-generation-with-past ^
  --weight-format %WEIGHT_FORMAT% ^
  --backup-precision int8_asym ^
  --group-size %GROUP_SIZE% ^
  --trust-remote-code ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-asym-g%GROUP_SIZE%

pause
