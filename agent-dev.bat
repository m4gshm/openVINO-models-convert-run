IF not defined OPENVINO_DEV_NAME (
    set OPENVINO_DEV_NAME=openvino_genai_windows_2026.4.0.0.dev20260803_x86_64
)

if not defined INTEL_OPENVINO_DIR (
    call bin/%OPENVINO_DEV_NAME%/setupvars.bat
)

.venv\Scripts\python agent.py %*