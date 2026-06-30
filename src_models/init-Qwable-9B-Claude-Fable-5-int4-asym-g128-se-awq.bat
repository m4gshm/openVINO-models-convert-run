rem python -m venv venv_qwen3_5
rem .\venv_qwen3_5\Scripts\activate.bat
rem python -m pip install --upgrade "optimum-intel[openvino]@git+https://github.com/huggingface/optimum-intel.git"
rem python -m pip install --upgrade transformers==5.2.0
rem pip install pillow
rem pip install torchvision


set MODEL_NAME=Qwable-9B-Claude-Fable-5
set MODEL_DEVELOPER=empero-ai
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
  --scale-estimation ^
  --awq ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-asym-g%GROUP_SIZE%-se-awq

pause
