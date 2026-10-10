pip install -U "git+https://github.com/huggingface/optimum-intel.git" torchvision Pillow --extra-index-url https://download.pytorch.org/whl/cpu
pip install -U "openvino>=2026.4.1"
pip install -U "transformers==5.2.0"
set MODEL_NAME=Qwen3.5-9B
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
  --sym ^
  --group-size %GROUP_SIZE% ^
  --trust-remote-code ^
  --dataset textvqa ^
  --awq ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-sym-g%GROUP_SIZE%-awq

pause

@REM Statistics collection ??????????????????????????????????????????????????????????????????? 100% 32/32 • 0:07:27 • 0:00:00
@REM INFO:nncf:Statistics of the bitwidth distribution:
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM +===========================+=============================+========================================+
@REM | int8_sym, per-channel     | 13% (25 / 273)              | 0% (0 / 248)                           |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | int4_sym, group size 128  | 87% (248 / 273)             | 100% (248 / 248)                       |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM Applying data-aware AWQ ????????????????????????????????????????????????????????????????? 100% 40/40 • 0:04:31 • 0:00:00
@REM Applying Weight Compression ??????????????????????????????????????????????????????????????????? 100% • 0:00:22 • 0:00:00
@REM INFO:nncf:Statistics of the bitwidth distribution:
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM +===========================+=============================+========================================+
@REM | int8_sym, per-channel     | 100% (1 / 1)                | 100% (1 / 1)                           |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM Applying Weight Compression ??????????????????????????????????????????????????????????????????? 100% • 0:00:02 • 0:00:00
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
@REM | float                     | 0% (1 / 9)                  | 0% (0 / 7)                             |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | int8_sym, per-channel     | 21% (1 / 9)                 | 0% (0 / 7)                             |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | int4_sym, group size 128  | 79% (7 / 9)                 | 100% (7 / 7)                           |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM Applying data-free AWQ ???????????????????????????????????????????????????????????????????? 100% 1/1 • 0:00:00 • 0:00:00
@REM Applying Weight Compression ??????????????????????????????????????????????????????????????????? 100% • 0:00:00 • 0:00:00
