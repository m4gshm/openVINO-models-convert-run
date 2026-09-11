set MODEL_NAME=LFM2-8B-A1B
set MODEL_DEVELOPER=LiquidAI
set MODEL_NAME_OUT=%MODEL_NAME%
set MODEL_PATH=./%MODEL_DEVELOPER%/%MODEL_NAME%
set OUTPUT_DIR=../models/%MODEL_NAME_OUT%

set WEIGHT_FORMAT=fp16

optimum-cli export openvino ^
  --model %MODEL_PATH% ^
  --task text-generation-with-past ^
  --weight-format %WEIGHT_FORMAT% ^
  --trust-remote-code ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%

pause
