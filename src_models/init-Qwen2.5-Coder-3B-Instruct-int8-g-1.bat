@REM git clone --depth 1 https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct -d Qwen/Qwen2.5-Coder-3B-Instruct
set MODEL_NAME=Qwen2.5-Coder-3B-Instruct
set MODEL_NAME_OUT=%MODEL_NAME%
set MODEL_DEVELOPER=Qwen
set MODEL_PATH=./%MODEL_DEVELOPER%/%MODEL_NAME%
set OUTPUT_DIR=../models/%MODEL_NAME_OUT%

set GROUP_SIZE=-1
set WEIGHT_FORMAT=int8

optimum-cli export openvino ^
  --model %MODEL_PATH% ^
  --task text-generation-with-past ^
  --weight-format %WEIGHT_FORMAT% ^
  --sym ^
  --group-size %GROUP_SIZE% ^
  --ratio 1.0 ^
  --trust-remote-code ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-sym-g%GROUP_SIZE%-r1/1

pause
