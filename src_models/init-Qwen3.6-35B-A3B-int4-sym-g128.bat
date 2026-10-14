pip install -q "git+https://github.com/huggingface/optimum-intel.git" --extra-index-url https://download.pytorch.org/whl/cpu
pip install -qU --pre "openvino-genai" "openvino" "openvino-tokenizers" "nncf" --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
@REM pip install -U "git+https://github.com/huggingface/optimum-intel.git" torchvision Pillow --extra-index-url https://download.pytorch.org/whl/cpu
@REM pip install -U "openvino==2026.4.1"
pip install -U "transformers==5.2"

set MODEL_NAME=Qwen3.6-35B-A3B
set MODEL_DEVELOPER=Qwen
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
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-sym-g%GROUP_SIZE%

pause

@REM INFO:nncf:Statistics of the bitwidth distribution:
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM +===========================+=============================+========================================+
@REM | int8_sym, per-channel     | 1% (31 / 541)               | 0% (0 / 510)                           |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | int4_sym, group size 128  | 99% (510 / 541)             | 100% (510 / 510)                       |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM Applying Weight Compression ??????????????????????????????????????????????????????????????????? 100% • 0:04:06 • 0:00:00
@REM INFO:nncf:Statistics of the bitwidth distribution:
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM +===========================+=============================+========================================+
@REM | int8_sym, per-channel     | 100% (1 / 1)                | 100% (1 / 1)                           |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM Applying Weight Compression ??????????????????????????????????????????????????????????????????? 100% • 0:00:01 • 0:00:00
@REM INFO:nncf:Statistics of the bitwidth distribution:
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM +===========================+=============================+========================================+
@REM | int8_sym, per-channel     | 100% (1 / 1)                | 100% (1 / 1)                           |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM Applying Weight Compression ??????????????????????????????????????????????????????????????????? 100% • 0:00:00 • 0:00:00
@REM INFO:nncf:Statistics of the bitwidth distribution:
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM +===========================+=============================+========================================+
@REM | int8_sym, per-channel     | 100% (110 / 110)            | 100% (110 / 110)                       |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM Applying Weight Compression ??????????????????????????????????????????????????????????????????? 100% • 0:00:03 • 0:00:00
@REM INFO:nncf:Statistics of the bitwidth distribution:
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM +===========================+=============================+========================================+
@REM | int8_sym, per-channel     | 100% (1 / 1)                | 100% (1 / 1)                           |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM Applying Weight Compression ??????????????????????????????????????????????????????????????????? 100% • 0:00:00 • 0:00:00
@REM WARNING:nncf:Group-wise quantization with group size 128 can't be applied to some nodes. They will be ignored and kept with original precision.
@REM Consider changing group size value or setting group size fallback parameter to ADJUST, which enables automatic adjustment to smaller group size values.
@REM INFO:nncf:Statistics of the bitwidth distribution:
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM +===========================+=============================+========================================+
@REM | float                     | 0% (1 / 14)                 | 0% (0 / 12)                            |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | int8_sym, per-channel     | 0% (1 / 14)                 | 0% (0 / 12)                            |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | int4_sym, group size 128  | 100% (12 / 14)              | 100% (12 / 12)                         |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM Applying Weight Compression ??????????????????????????????????????????????????????????????????? 100% • 0:00:05 • 0:00:00
