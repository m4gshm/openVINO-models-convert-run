python -m venv gemma4
if not exist ".\gemma4\Scripts\activate.bat" (
  echo ERROR: failed to create venv gemma4
  pause
  exit /b 1
)
call .\gemma4\Scripts\activate.bat

pip install "git+https://github.com/huggingface/optimum-intel.git" torchvision Pillow --extra-index-url https://download.pytorch.org/whl/cpu
pip install -U transformers==5.10.4
pip install -U "openvino>=2026.4.1"
pip install datasets

set MODEL_NAME=gemma-4-12b-it-qat-q4_0-unquantized
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
  --backup-precision int8_sym ^
  --group-size %GROUP_SIZE% ^
  --trust-remote-code ^
  --sym ^
  --dataset textvqa ^
  --awq ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-sym-g%GROUP_SIZE%-awq

pause
