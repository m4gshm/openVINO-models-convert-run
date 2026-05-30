set MODEL_NAME=Ministral-3-3B-Instruct-2512
set MODEL_DEVELOPER=ministralai
set MODEL_NAME_OUT=%MODEL_NAME%
set MODEL_PATH=./%MODEL_DEVELOPER%/%MODEL_NAME%
set OUTPUT_DIR=../models/%MODEL_NAME_OUT%

set WEIGHT_FORMAT=int8

optimum-cli export openvino ^
  --model %MODEL_PATH% ^
  --task text-generation-with-past ^
  --weight-format %WEIGHT_FORMAT% ^
  --sym ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-sym/1

pause
