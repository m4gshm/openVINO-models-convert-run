python -m venv qwen3.5
if not exist ".\qwen3.5\Scripts\activate.bat" (
  echo ERROR: failed to create venv qwen3.5
  pause
  exit /b 1
)
call .\qwen3.5\Scripts\activate.bat

pip install -U optimum-intel
pip install -U "openvino>=2026.4.1"
pip install -U "transformers==5.2.0"

set MODEL_NAME=Qwen3.5-2B
set MODEL_DEVELOPER=Qwen
set MODEL_NAME_OUT=%MODEL_NAME%
set MODEL_PATH=./%MODEL_DEVELOPER%/%MODEL_NAME%
set OUTPUT_DIR=../models/%MODEL_NAME_OUT%

set WEIGHT_FORMAT=int4

optimum-cli export openvino ^
  --model %MODEL_PATH% ^
  --task image-text-to-text ^
  --weight-format %WEIGHT_FORMAT% ^
  --backup-precision int8_sym ^
  --sym ^
  --group-size-fallback ignore ^
  --trust-remote-code ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-sym

pause
