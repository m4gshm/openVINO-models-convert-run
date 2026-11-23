@REM EAGLE3 draft head for Qwen3.6-35B-A3B (vocab 248320, hidden 2048, chain drafting).
@REM The draft repo ships no tokenizer: borrow the target's one so optimum-cli and
@REM the GenAI draft loader both find it. Needs optimum-intel git-master (EAGLE3 patcher).
pip install optimum-intel@git+https://github.com/huggingface/optimum-intel.git
pip install -U "optimum-intel[openvino]" "nncf>=3.4.0"

set MODEL_NAME=specdrift-qwen3.6-35b-a3b-eagle3
set MODEL_DEVELOPER=Dogacel
set MODEL_NAME_OUT=Qwen3.6-35B-A3B-eagle3
set MODEL_PATH=./%MODEL_DEVELOPER%/%MODEL_NAME%
set OUTPUT_DIR=../models/%MODEL_NAME_OUT%

set GROUP_SIZE=128
set WEIGHT_FORMAT=int4

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

@REM GenAI takes hidden_layers_list from the draft IR rt_info; without it the
@REM pipeline defaults to layers {2, n/2, n-3} and Eagle3 graft fails with
@REM "Port for tensor name last_hidden_state was not found".
python %~dp0apply_eagle3_rt_info.py %OUTPUT_DIR%-%WEIGHT_FORMAT%-sym-g%GROUP_SIZE%

@REM A VLM main model drives the draft in EMBEDDINGS mode: the runner sets
@REM "inputs_embeds" on the draft by name. Convert the tokens-style draft
@REM (input_ids -> embed_tokens Gather) to an embeds-native one, otherwise the
@REM pipeline fails with "Port inputs_embeds was not found".
python %~dp0make_eagle3_draft_embeds_native.py %OUTPUT_DIR%-%WEIGHT_FORMAT%-sym-g%GROUP_SIZE%

pause
