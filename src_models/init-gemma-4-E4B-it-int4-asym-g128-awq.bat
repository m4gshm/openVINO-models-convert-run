set MODEL_NAME=gemma-4-E4B-it
set MODEL_DEVELOPER=google
set MODEL_NAME_OUT=%MODEL_NAME%
set MODEL_PATH=./%MODEL_DEVELOPER%/%MODEL_NAME%
set OUTPUT_DIR=../models/%MODEL_NAME_OUT%

set GROUP_SIZE=128
set WEIGHT_FORMAT=int4

optimum-cli export openvino ^
  --model %MODEL_PATH% ^
  --task image-text-to-text ^
  --weight-format %WEIGHT_FORMAT% ^
  --backup-precision int8_asym ^
  --group-size %GROUP_SIZE% ^
  --trust-remote-code ^
  --dataset contextual ^
  --awq ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-asym-g%GROUP_SIZE%-awq

pause
