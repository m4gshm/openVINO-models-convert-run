python -m venv gemma4
if not exist ".\gemma4\Scripts\activate.bat" (
  echo ERROR: failed to create venv gemma4
  pause
  exit /b 1
)
call .\gemma4\Scripts\activate.bat

pip install -U optimum-intel
pip install -U transformers==5.10.4
pip install -U "openvino>=2026.4.1"

set MODEL_NAME=gemma-4-E2B-it-assistant
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
