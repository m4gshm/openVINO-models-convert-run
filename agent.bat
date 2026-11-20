@REM reset dev-build vars from agent-dev.bat so the pip-installed release openvino (venv) is used
set "OPENVINO_DEV_NAME="
set "INTEL_OPENVINO_DIR="
set "PYTHONPATH="
.venv\Scripts\python agent.py %*
