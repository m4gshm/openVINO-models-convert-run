set MODEL_NAME=Qwen2.5-Coder-7B-Instruct
set MODEL_NAME_OUT=%MODEL_NAME%
set MODEL_DEVELOPER=Qwen
set MODEL_PATH=./%MODEL_DEVELOPER%/%MODEL_NAME%
set OUTPUT_DIR=../models/%MODEL_NAME_OUT%

set GROUP_SIZE=128
set WEIGHT_FORMAT=int4

optimum-cli export openvino ^
  --model %MODEL_PATH% ^
  --task text-generation-with-past ^
  --weight-format %WEIGHT_FORMAT% ^
  --backup-precision int8_sym ^
  --sym ^
  --group-size %GROUP_SIZE% ^
  --ratio 1.0 ^
  --trust-remote-code ^
  --dataset gsm8k ^
  --scale-estimation ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-sym-g%GROUP_SIZE%-r1-se/1

pause
