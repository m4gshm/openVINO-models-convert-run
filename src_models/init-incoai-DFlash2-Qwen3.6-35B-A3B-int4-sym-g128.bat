@REM DFlash2 draft head for Qwen3.6-35B-A3B (incoai, gated Markov head "dflash2").
@REM Target layers {1, 6, 11, 16, 22, 27, 32, 37}, mask-token block drafting, block_size 8,
@REM full target vocab 248320 (no d2t mapping). Needs optimum-intel git-master
@REM (pip install optimum-intel@git+https://github.com/huggingface/optimum-intel.git):
@REM the DFlash2 patcher and the gated head export do not exist in release 2.3.0.
@REM The exporter stamps the dflash rt_info (dflash_mode, mask_token_id, target_layer_ids,
@REM markov_rank, markov_head_type) onto the graph itself — no extra rt_info script needed,
@REM and the draft needs no embeds-native surgery: the GenAI dflash transforms graft the
@REM target embedding and lm_head into the draft at load time.
@REM Consumed by Qwen3.6-35B-A3B-int4-sym-g128-awq-GPU-dflash2-dev.bat (--dflash_mode on),
@REM which needs a GenAI build with gated-head support (see dflash_model_transforms.cpp).
set MODEL_NAME=Qwen3.6-35B-A3B-DFlash2
set MODEL_DEVELOPER=incoai
set MODEL_NAME_OUT=Qwen3.6-35B-A3B-dflash2
set MODEL_PATH=./%MODEL_DEVELOPER%/%MODEL_NAME%
set OUTPUT_DIR=../models/%MODEL_NAME_OUT%

set GROUP_SIZE=128
set WEIGHT_FORMAT=int4

@REM The repo ships no tokenizer: borrow the target's one so the export and the
@REM GenAI draft loader both find it. huggingface-cli is broken in this python
@REM (no huggingface_hub.commands), use the snapshot_download API.
python -c "from huggingface_hub import snapshot_download; snapshot_download(r'%MODEL_DEVELOPER%/%MODEL_NAME%', local_dir=r'%MODEL_PATH%')"

copy /Y ..\models\Qwen3.6-35B-A3B-int4-sym-g128-awq\tokenizer.json %MODEL_PATH%\
copy /Y ..\models\Qwen3.6-35B-A3B-int4-sym-g128-awq\tokenizer_config.json %MODEL_PATH%\

@REM optimum-cli resolves %MODEL_PATH% as a HF repo id, not a local folder, so export
@REM from inside the model directory with --model . — and the output path is relative
@REM to that cwd too (it lands in ..\..\..\models, NOT src_models\models).
pushd %MODEL_PATH%
optimum-cli export openvino ^
  --model . ^
  --task text-generation-with-past ^
  --weight-format %WEIGHT_FORMAT% ^
  --backup-precision int8_sym ^
  --group-size %GROUP_SIZE% ^
  --sym ^
  ..\..\..\models\%MODEL_NAME_OUT%-%WEIGHT_FORMAT%-sym-g%GROUP_SIZE%
popd

@REM Sanity check: the draft graph must carry the dflash rt_info.
findstr /C:"dflash_mode" ..\models\%MODEL_NAME_OUT%-%WEIGHT_FORMAT%-sym-g%GROUP_SIZE%\openvino_model.xml

@REM  INFO:nncf:Statistics of the bitwidth distribution:
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM +===========================+=============================+========================================+
@REM | float                     | 0% (1 / 11)                 | 0% (0 / 8)                             |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | int8_sym, per-channel     | 80% (2 / 11)                | 0% (0 / 8)                             |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | int4_sym, group size 128  | 20% (8 / 11)                | 100% (8 / 8)                           |
@REM +---------------------------+-----------------------------+----------------------------------------+
