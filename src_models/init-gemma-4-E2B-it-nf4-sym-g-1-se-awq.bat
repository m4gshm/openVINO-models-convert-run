pip install "git+https://github.com/huggingface/optimum-intel.git" --extra-index-url https://download.pytorch.org/whl/cpu
pip install transformers==5.5.4
pip install torchvision Pillow --extra-index-url https://download.pytorch.org/whl/cpu

set MODEL_NAME=gemma-4-E2B-it
set MODEL_DEVELOPER=google
set MODEL_NAME_OUT=%MODEL_NAME%
set MODEL_PATH=./%MODEL_DEVELOPER%/%MODEL_NAME%
set OUTPUT_DIR=../models/%MODEL_NAME_OUT%

set GROUP_SIZE=-1
set WEIGHT_FORMAT=nf4

optimum-cli export openvino ^
  --model %MODEL_PATH% ^
  --task image-text-to-text ^
  --weight-format %WEIGHT_FORMAT% ^
  --group-size %GROUP_SIZE% ^
  --trust-remote-code ^
  --dataset textvqa ^
  --sym ^
  --scale-estimation ^
  --awq ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-sym-g%GROUP_SIZE%-se-awq

pause

@REM  Statistics collection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 32/32 • 0:02:25 • 0:00:00
@REM  INFO:nncf:Statistics of the bitwidth distribution:
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM  +===========================+=============================+========================================+
@REM  | float                     | 0% (2 / 279)                | 0% (0 / 276)                           |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | int8_asym, per-channel    | 18% (1 / 279)               | 0% (0 / 276)                           |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | nf4, per-channel          | 82% (276 / 279)             | 100% (276 / 276)                       |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  Applying data-aware AWQ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 35/35 • 0:04:51 • 0:00:00
@REM  Applying Scale Estimation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 277/277 • 0:07:57 • 0:00:00
@REM  Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:20 • 0:00:00
@REM  INFO:nncf:Statistics of the bitwidth distribution:
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM  +===========================+=============================+========================================+
@REM  | int8_sym, per-channel     | 100% (1 / 1)                | 100% (1 / 1)                           |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:01 • 0:00:00
@REM  INFO:nncf:Statistics of the bitwidth distribution:
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM  +===========================+=============================+========================================+
@REM  | float                     | 0% (2 / 117)                | 0% (0 / 115)                           |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | int8_sym, per-channel     | 100% (115 / 117)            | 100% (115 / 115)                       |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:04 • 0:00:00
@REM  INFO:nncf:Statistics of the bitwidth distribution:
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM  +===========================+=============================+========================================+
@REM  | int8_sym, per-channel     | 100% (1 / 1)                | 100% (1 / 1)                           |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:05 • 0:00:00
@REM  INFO:nncf:Statistics of the bitwidth distribution:
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM  +===========================+=============================+========================================+
@REM  | int8_sym, per-channel     | 100% (137 / 137)            | 100% (137 / 137)                       |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:05 • 0:00:00
