@REM set EXPECTED_OPENVINO_DEV_NAME=openvino_genai_windows_2026.5.0.0.dev20261009_x86_64
set EXPECTED_OPENVINO_DEV_NAME=dist
if not defined OPENVINO_DEV_NAME (
    set "OPENVINO_DEV_NAME=%EXPECTED_OPENVINO_DEV_NAME%"
) else if "%OPENVINO_DEV_NAME%" NEQ "%EXPECTED_OPENVINO_DEV_NAME%" (
    set "OPENVINO_DEV_NAME=%EXPECTED_OPENVINO_DEV_NAME%"
)

set "INTEL_OPENVINO_DIR="
set "PYTHONPATH="
@REM setupvars.bat prepends its lib dirs to PATH on every call, so reruns in one console grow PATH past the cmd limit
@REM %SystemRoot% path guards against msys/GnuWin32 find shadowing the builtin
echo "%OPENVINO_LIB_PATHS%" | "%SystemRoot%\System32\find.exe" /i "\bin\%OPENVINO_DEV_NAME%\\" >NUL || call .\bin\%OPENVINO_DEV_NAME%\setupvars.bat
echo OPENVINO_DEV_NAME: %OPENVINO_DEV_NAME%
.venv\Scripts\python agent.py %*
