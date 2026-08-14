set MODEL_NAME=gemma-4-26B-A4B-it-qat-q4_0-unquantized
set MODEL_DEVELOPER=google
set MODEL_NAME_OUT=%MODEL_NAME%
set MODEL_PATH=./%MODEL_DEVELOPER%/%MODEL_NAME%
set OUTPUT_DIR=../models/%MODEL_NAME_OUT%

set GROUP_SIZE=64
set WEIGHT_FORMAT=int4

optimum-cli export openvino ^
  --model %MODEL_PATH% ^
  --task image-text-to-text ^
  --weight-format %WEIGHT_FORMAT% ^
  --backup-precision int8_asym ^
  --group-size %GROUP_SIZE% ^
  --trust-remote-code ^
  --dataset textvqa ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-asym-g%GROUP_SIZE%

pause

INFO:nncf:Statistics of the bitwidth distribution:
+---------------------------+-----------------------------+----------------------------------------+
| Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
+===========================+=============================+========================================+
| int8_asym, per-channel    | 3% (31 / 356)               | 0% (0 / 325)                           |
+---------------------------+-----------------------------+----------------------------------------+
| int4_asym, group size 64  | 97% (325 / 356)             | 100% (325 / 325)                       |
+---------------------------+-----------------------------+----------------------------------------+
Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:03:15 • 0:00:00
INFO:nncf:Statistics of the bitwidth distribution:
+---------------------------+-----------------------------+----------------------------------------+
| Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
+===========================+=============================+========================================+
| int8_sym, per-channel     | 100% (1 / 1)                | 100% (1 / 1)                           |
+---------------------------+-----------------------------+----------------------------------------+
Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:05 • 0:00:00
INFO:nncf:Statistics of the bitwidth distribution:
+---------------------------+-----------------------------+----------------------------------------+
| Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
+===========================+=============================+========================================+
| int8_sym, per-channel     | 100% (192 / 192)            | 100% (192 / 192)                       |
+---------------------------+-----------------------------+----------------------------------------+
Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:06 • 0:00:00
INFO:nncf:Statistics of the bitwidth distribution:
+---------------------------+-----------------------------+----------------------------------------+
| Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
+===========================+=============================+========================================+
+---------------------------+-----------------------------+----------------------------------------+
