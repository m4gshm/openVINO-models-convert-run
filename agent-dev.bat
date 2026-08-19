EXPECTED_OPENVINO_DEV_NAME==openvino_genai_windows_2026.4.0.0.dev20260818_x86_64
IF not defined OPENVINO_DEV_NAME or "%OPENVINO_DEV_NAME%"=="%EXPECTED_NAME% (
    set OPENVINO_DEV_NAME
)

if not defined INTEL_OPENVINO_DIR (
    call bin/%OPENVINO_DEV_NAME%/setupvars.bat
)
echo OPENVINO_DEV_NAME: %OPENVINO_DEV_NAME%
.venv\Scripts\python agent.py %*
