pip install -q "git+https://github.com/huggingface/optimum-intel.git" --extra-index-url https://download.pytorch.org/whl/cpu
pip install -qU --pre "openvino-genai" "openvino" "openvino-tokenizers" "nncf" --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
@REM  pip install -U "openvino==2026.4.1"
pip install -U "transformers<=5.4.0"
set MODEL_NAME=LFM-2.5-Coder-2.6B
set MODEL_DEVELOPER=Schnuckade
set MODEL_NAME_OUT=%MODEL_NAME%
set MODEL_PATH=./%MODEL_DEVELOPER%/%MODEL_NAME%
set OUTPUT_DIR=../models/%MODEL_NAME_OUT%

set GROUP_SIZE=-1
set WEIGHT_FORMAT=nf4

optimum-cli export openvino ^
  --model %MODEL_PATH% ^
  --task text-generation-with-past ^
  --weight-format %WEIGHT_FORMAT% ^
  --backup-precision int8_sym ^
  --sym ^
  --group-size %GROUP_SIZE% ^
  --trust-remote-code ^
  --dataset gsm8k ^
  --scale-estimation ^
  --awq ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-sym-g%GROUP_SIZE%-se-awq

pause

@REM INFO:nncf:Statistics of the bitwidth distribution:
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM +===========================+=============================+========================================+
@REM | float                     | 0% (1 / 190)                | 0% (0 / 166)                           |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | int8_sym, per-channel     | 10% (23 / 190)              | 0% (0 / 166)                           |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | nf4, per-channel          | 90% (166 / 190)             | 100% (166 / 166)                       |
@REM +---------------------------+-----------------------------+----------------------------------------+
