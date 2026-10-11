python -m venv gemma4
if not exist ".\gemma4\Scripts\activate.bat" (
  echo ERROR: failed to create venv gemma4
  pause
  exit /b 1
)
call .\gemma4\Scripts\activate.bat

pip install -U optimum-intel
pip install -U transformers==5.10.4
pip install -U "openvino>=2026.4.1"

set MODEL_NAME=gemma-4-E2B-it-qat-q4_0-unquantized
set MODEL_DEVELOPER=google
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
  --dataset textvqa ^
  --scale-estimation ^
  --awq ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-sym-g%GROUP_SIZE%-se-awq

pause

@REM  Statistics collection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 32/32 • 0:02:44 • 0:00:00
@REM  INFO:nncf:Statistics of the bitwidth distribution:
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM  +===========================+=============================+========================================+
@REM  | float                     | 0% (2 / 279)                | 0% (0 / 276)                           |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | int8_sym, per-channel     | 18% (1 / 279)               | 0% (0 / 276)                           |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | int4_sym, group size 128  | 82% (276 / 279)             | 100% (276 / 276)                       |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  Applying data-aware AWQ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 35/35 • 0:01:42 • 0:00:00
@REM  Applying Scale Estimation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 277/277 • 0:07:33 • 0:00:00
@REM  Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:10 • 0:00:00
@REM  INFO:nncf:Statistics of the bitwidth distribution:
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM  +===========================+=============================+========================================+
@REM  | int8_sym, per-channel     | 100% (1 / 1)                | 100% (1 / 1)                           |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:00 • 0:00:00
@REM  INFO:nncf:Statistics of the bitwidth distribution:
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM  +===========================+=============================+========================================+
@REM  | float                     | 0% (2 / 117)                | 0% (0 / 115)                           |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | int8_sym, per-channel     | 100% (115 / 117)            | 100% (115 / 115)                       |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:03 • 0:00:00
@REM  INFO:nncf:Statistics of the bitwidth distribution:
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM  +===========================+=============================+========================================+
@REM  | int8_sym, per-channel     | 100% (1 / 1)                | 100% (1 / 1)                           |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:03 • 0:00:00
@REM  INFO:nncf:Statistics of the bitwidth distribution:
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM  +===========================+=============================+========================================+
@REM  | int8_sym, per-channel     | 100% (137 / 137)            | 100% (137 / 137)                       |
@REM  +---------------------------+-----------------------------+----------------------------------------+
@REM  Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:04 • 0:00:00
