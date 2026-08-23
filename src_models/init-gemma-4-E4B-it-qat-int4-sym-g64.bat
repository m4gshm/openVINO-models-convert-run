@REM pip install --upgrade "transformers==5.7.0"
set MODEL_NAME=gemma-4-E4B-it-qat-q4_0-unquantized
set MODEL_DEVELOPER=google
set MODEL_NAME_OUT=%MODEL_NAME%
set MODEL_PATH=./%MODEL_DEVELOPER%/%MODEL_NAME%
set OUTPUT_DIR=../models/%MODEL_NAME_OUT%

set GROUP_SIZE=64
set WEIGHT_FORMAT=int4

optimum-cli export openvino ^
  --model %MODEL_PATH% ^
  --task image-text-to-text ^
  --weight-format %WEIGHT_FORMAT% ^
  --backup-precision int8_sym ^
  --group-size %GROUP_SIZE% ^
  --trust-remote-code ^
  --sym ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-sym-g%GROUP_SIZE%

pause
