if not defined INTEL_OPENVINO_DIR (
    call ./dev/openvino_genai_windows_2026.4.0.0.dev20260714_x86_64/setupvars.bat
)
python agent.py --model OmniCoder-9B-int4-sym-g128-se-awq --pip CB --cache_precision u4