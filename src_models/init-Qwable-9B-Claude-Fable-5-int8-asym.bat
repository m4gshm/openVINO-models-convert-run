@REM pip install --upgrade "transformers==5.2.0"
 
set MODEL_NAME=Qwable-9B-Claude-Fable-5
set MODEL_DEVELOPER=empero-ai
set MODEL_NAME_OUT=%MODEL_NAME%
set MODEL_PATH=./%MODEL_DEVELOPER%/%MODEL_NAME%
set OUTPUT_DIR=../models/%MODEL_NAME_OUT%

set WEIGHT_FORMAT=int8

optimum-cli export openvino ^
  --model %MODEL_PATH% ^
  --task image-text-to-text ^
  --weight-format %WEIGHT_FORMAT% ^
  --trust-remote-code ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-asym

pause
