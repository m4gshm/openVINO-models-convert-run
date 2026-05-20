set MODEL_NAME=Qwen3-Embedding-0.6B
set MODEL_DEVELOPER=Qwen
set MODEL_NAME_OUT=%MODEL_NAME%
set SOURCE_MODEL=%MODEL_DEVELOPER%/%MODEL_NAME%
set MODEL_PATH=./%SOURCE_MODEL%
set OUTPUT_DIR=../models/%MODEL_NAME_OUT%

set GROUP_SIZE=128
set WEIGHT_FORMAT=int4

optimum-cli export openvino ^
  --model %MODEL_PATH% ^
  --task feature-extraction ^
  --weight-format %WEIGHT_FORMAT% ^
  --backup-precision int8_sym ^
  --sym ^
  --group-size %GROUP_SIZE% ^
  --ratio 1.0 ^
  --trust-remote-code ^
  --dataset wikitext2 ^
  --scale-estimation ^
  --awq ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-sym-g%GROUP_SIZE%-r1-se-awq/1

pause
