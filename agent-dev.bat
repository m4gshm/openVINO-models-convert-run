set OPENVINO_DEV_PATH=openvino_genai_windows_2026.4.0.0.dev20260717_x86_64
rem set OPENVINO_DEV_PATH=odev.openvino_genai_windows_2026.3.0.0rc2_x86_64
if not defined INTEL_OPENVINO_DIR (
    call ./dev/%OPENVINO_DEV_PATH%/setupvars.bat
)
python agent.py %*