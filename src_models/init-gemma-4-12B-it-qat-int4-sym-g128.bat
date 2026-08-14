@REM pip install --upgrade transformers==5.10.0
set MODEL_NAME=gemma-4-12b-it-qat-q4_0-unquantized
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
  --dataset textvqa ^
  --sym ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-sym-g%GROUP_SIZE%

pause

INFO:nncf:Statistics of the bitwidth distribution:
+---------------------------+-----------------------------+----------------------------------------+
| Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
+===========================+=============================+========================================+
| int8_sym, per-channel     | 8% (1 / 329)                | 0% (0 / 328)                           |
+---------------------------+-----------------------------+----------------------------------------+
| int4_sym, group size 128  | 92% (328 / 329)             | 100% (328 / 328)                       |
+---------------------------+-----------------------------+----------------------------------------+
Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:02:46 • 0:00:00
INFO:nncf:Statistics of the bitwidth distribution:
+---------------------------+-----------------------------+----------------------------------------+
| Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
+===========================+=============================+========================================+
| int8_sym, per-channel     | 100% (1 / 1)                | 100% (1 / 1)                           |
+---------------------------+-----------------------------+----------------------------------------+
Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:07 • 0:00:00
INFO:nncf:Statistics of the bitwidth distribution:
+---------------------------+-----------------------------+----------------------------------------+
| Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
+===========================+=============================+========================================+
| int8_sym, per-channel     | 100% (3 / 3)                | 100% (3 / 3)                           |
+---------------------------+-----------------------------+----------------------------------------+
Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:00 • 0:00:00
