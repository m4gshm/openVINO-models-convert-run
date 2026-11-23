@REM Independent EAGLE3 draft head for Qwen3.6-35B-A3B (PatchyTisa, trained on caption data,
@REM aux layers {2, 19, 37}, draft vocab 32000). Same pipeline as the Dogacel one: the draft
@REM repo ships no tokenizer, borrow the target's one; needs optimum-intel git-master.
set MODEL_NAME=Qwen3.6-35B-A3B-Caption-Eagle3DraftModel
set MODEL_DEVELOPER=PatchyTisa
set MODEL_PATH=./%MODEL_DEVELOPER%/%MODEL_NAME%
set OUTPUT_DIR=../models/Qwen3.6-35B-A3B-eagle3-caption

set GROUP_SIZE=128
set WEIGHT_FORMAT=int4

python -m huggingface_hub.commands.huggingface_cli download %MODEL_DEVELOPER%/%MODEL_NAME% --local-dir %MODEL_PATH%

copy /Y ..\models\Qwen3.6-35B-A3B-int4-sym-g128-awq\tokenizer.json %MODEL_PATH%\
copy /Y ..\models\Qwen3.6-35B-A3B-int4-sym-g128-awq\tokenizer_config.json %MODEL_PATH%\

optimum-cli export openvino ^
  --model %MODEL_PATH% ^
  --task text-generation-with-past ^
  --weight-format %WEIGHT_FORMAT% ^
  --backup-precision int8_sym ^
  --group-size %GROUP_SIZE% ^
  --trust-remote-code ^
  --sym ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-sym-g%GROUP_SIZE%

python %~dp0apply_eagle3_rt_info.py %OUTPUT_DIR%-%WEIGHT_FORMAT%-sym-g%GROUP_SIZE%

python %~dp0make_eagle3_draft_embeds_native.py %OUTPUT_DIR%-%WEIGHT_FORMAT%-sym-g%GROUP_SIZE%
