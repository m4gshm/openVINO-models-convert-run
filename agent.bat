@REM reset dev-build vars from agent-dev.bat so the pip-installed release openvino (venv) is used
set "OPENVINO_DEV_NAME="
set "INTEL_OPENVINO_DIR="
set "PYTHONPATH="
@REM experiments: never clean the kernel cache (explicit --cache_cleanup_days in a script wins)
.venv\Scripts\python agent.py --cache_cleanup_days 0 %*
