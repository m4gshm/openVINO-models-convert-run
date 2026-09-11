@REM transformers==5.7.0
set MODEL_NAME=LFM2.5-8B-A1B-Coder
set MODEL_DEVELOPER=josephmayo
set MODEL_NAME_OUT=%MODEL_NAME%
set MODEL_PATH=./%MODEL_DEVELOPER%/%MODEL_NAME%
set OUTPUT_DIR=../models/%MODEL_NAME_OUT%

set WEIGHT_FORMAT=int4

optimum-cli export openvino ^
  --model %MODEL_PATH% ^
  --task text-generation-with-past ^
  --weight-format %WEIGHT_FORMAT% ^
  --backup-precision int8_sym ^
  --group-size %GROUP_SIZE% ^
  --trust-remote-code ^
  --sym ^
  --dataset wikitext2 ^
  --scale-estimation ^
  --awq ^
  %OUTPUT_DIR%-%WEIGHT_FORMAT%-sym-g%GROUP_SIZE%-se-awq

pause

@REM Statistics collection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 128/128 • 0:08:00 • 0:00:00
@REM INFO:nncf:Statistics of the bitwidth distribution:
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
@REM +===========================+=============================+========================================+
@REM | float                     | 0% (1 / 174)                | 0% (0 / 154)                           |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | int8_sym, per-channel     | 3% (19 / 174)               | 0% (0 / 154)                           |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM | int4_sym, group size 128  | 97% (154 / 174)             | 100% (154 / 154)                       |
@REM +---------------------------+-----------------------------+----------------------------------------+
@REM Applying data-aware AWQ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 24/24 • 0:09:04 • 0:00:00
@REM Applying Scale Estimation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 173/173 • 7:10:15 • 0:00:00
@REM Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:27 • 0:00:00
@REM Traceback (most recent call last):
@REM   File "<frozen runpy>", line 203, in _run_module_as_main
@REM   File "<frozen runpy>", line 88, in _run_code
@REM   File "C:\ProgramData\scoop\apps\python\current\Scripts\optimum-cli.exe\__main__.py", line 5, in <module>
@REM     sys.exit(main())
@REM              ~~~~^^
@REM   File "C:\ProgramData\scoop\apps\python\current\Lib\site-packages\optimum\commands\optimum_cli.py", line 219, in main
@REM     service.run()
@REM     ~~~~~~~~~~~^^
@REM   File "C:\ProgramData\scoop\apps\python\current\Lib\site-packages\optimum\commands\export\openvino.py", line 486, in run
@REM     _main_quantize(
@REM     ~~~~~~~~~~~~~~^
@REM         model_name_or_path=self.args.model,
@REM         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
@REM     ...<6 lines>...
@REM         model_kwargs=self.args.model_kwargs,
@REM         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
@REM     )
@REM     ^
@REM   File "C:\ProgramData\scoop\apps\python\current\Lib\site-packages\optimum\exporters\openvino\__main__.py", line 813, in _main_quantize
@REM     model._apply_quantization(
@REM     ~~~~~~~~~~~~~~~~~~~~~~~~~^
@REM         quantization_config,
@REM         ^^^^^^^^^^^^^^^^^^^^
@REM     ...<5 lines>...
@REM         immediate_save=True,
@REM         ^^^^^^^^^^^^^^^^^^^^
@REM     )
@REM     ^
@REM   File "C:\ProgramData\scoop\apps\python\current\Lib\site-packages\optimum\intel\openvino\modeling_base.py", line 775, in _apply_quantization
@REM     quantizer.quantize(ov_config=OVConfig(quantization_config=quantization_config), **kwargs)
@REM     ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
@REM   File "C:\ProgramData\scoop\apps\python\current\Lib\site-packages\optimum\intel\openvino\quantization.py", line 1454, in quantize
@REM     self._quantize_ovbasemodel(
@REM     ~~~~~~~~~~~~~~~~~~~~~~~~~~^
@REM         ov_config,
@REM         ^^^^^^^^^^
@REM     ...<3 lines>...
@REM         **kwargs,
@REM         ^^^^^^^^^
@REM     )
@REM     ^
@REM   File "C:\ProgramData\scoop\apps\python\current\Lib\site-packages\optimum\intel\openvino\quantization.py", line 1541, in _quantize_ovbasemodel
@REM     ov_model_path.with_suffix(".bin").unlink()
@REM     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
@REM   File "C:\ProgramData\scoop\apps\python\current\Lib\pathlib\__init__.py", line 1042, in unlink
@REM     os.unlink(self)
@REM     ~~~~~~~~~^^^^^^
@REM PermissionError: [WinError 32] Процесс не может получить доступ к файлу, так как этот файл занят другим процессом: 'C:\\Users\\mfour\\AppData\\Local\\Temp\\tmpg45ljr9b\\openvino_model.bin'