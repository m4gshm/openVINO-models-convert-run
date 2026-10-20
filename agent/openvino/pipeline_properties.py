"""Pipeline properties of the OpenVINO engines, per device.

The properties are the key/value pairs handed over to the pipeline factory of
``openvino_genai``: they select the cache directory, the performance hint and
the device specific knobs.

Every builder reads the switches of the command line, so that the whole set of
the pipeline properties of a device lives in one place.
"""
import logging
import multiprocessing
from typing import Any

from agent.common.cli import enum_value
from agent.openvino.options import DeviceType

log = logging.getLogger(__name__)


def build_cpu_properties(model_cache_dir: str, args) -> dict[str, Any]:
    """Properties of the CPU pipeline."""
    return {
        "CACHE_DIR": model_cache_dir,
        "PERFORMANCE_HINT": args.performance_hint,
        "ENABLE_MMAP": "YES",
        "INFERENCE_NUM_THREADS": multiprocessing.cpu_count(),
    }


def build_gpu_properties(model_cache_dir: str, args) -> dict[str, Any]:
    """Properties of the GPU pipeline."""
    return {
        "CACHE_DIR": model_cache_dir,
        "PERFORMANCE_HINT": args.performance_hint,
        "ENABLE_MMAP": "YES",

        "GPU_ENABLE_LARGE_ALLOCATIONS": args.gpu_enable_large_allocations,
        "GPU_QUEUE_THROTTLE": args.gpu_priorities,
        "MODEL_PRIORITY": args.gpu_priorities,
        "GPU_HOST_TASK_PRIORITY": args.gpu_priorities,
        "GPU_QUEUE_PRIORITY": args.gpu_priorities,

        "COMPILATION_NUM_THREADS": multiprocessing.cpu_count(),
    }


def build_npu_properties(model_cache_dir: str, args, max_prompt_len: int | None) -> dict[str, Any]:
    """Properties of the NPU pipeline."""
    properties: dict[str, Any] = {
        "CACHE_DIR": model_cache_dir,
        "CACHE_MODE": "OPTIMIZE_SPEED",
        "PERFORMANCE_HINT": args.performance_hint,
        "ENABLE_MMAP": "YES",

        "NPU_COMPILER_TYPE": args.npu_compiler_type,
        "NPU_USE_NPUW": "YES",
        "NPU_TURBO": args.npu_turbo,

        "NPUW_LLM_GENERATE_HINT": args.npu_generate_hint,
        "NPUW_LLM_PREFILL_HINT": args.npu_prefill_hint,
        "NPUW_LLM_PREFILL_CHUNK_SIZE": args.npuw_llm_prefill_chunk_size,

        "NPUW_LLM_GENERATE_PYRAMID": args.npuw_llm_generate_pyramid,
        "NPUW_LLM_SHARED_HEAD": "YES",
        "NPUW_PARALLEL_COMPILE": "YES",
        "LOG_LEVEL": "LOG_INFO",
    }
    if max_prompt_len:
        properties["MAX_PROMPT_LEN"] = max_prompt_len
    properties.update(optional_npu_properties(args))
    return properties


def optional_npu_properties(args) -> dict[str, Any]:
    """The NPUW knobs the client asked for; the pipeline defaults stay in place otherwise."""
    optional: dict[str, Any] = {}
    if args.npuw_llm:
        optional["NPUW_LLM"] = args.npuw_llm
    if args.npuw_pa:
        optional["NPUW_PA"] = args.npuw_pa
    if args.npuw_devices:
        optional["NPUW_DEVICES"] = args.npuw_devices
    if args.npu_compilation_mode_params:
        # optimization-level=2 performance-hint-override=latency
        optional["NPU_COMPILATION_MODE_PARAMS"] = args.npu_compilation_mode_params
    if args.npuw_llm_prefill_moe_hint:
        optional["NPUW_LLM_PREFILL_MOE_HINT"] = args.npuw_llm_prefill_moe_hint
    if args.npuw_llm_generate_moe_hint:
        optional["NPUW_LLM_GENERATE_MOE_HINT"] = args.npuw_llm_generate_moe_hint
    if args.npuw_llm_prefill_attention_hint:
        optional["NPUW_LLM_PREFILL_ATTENTION_HINT"] = args.npuw_llm_prefill_attention_hint
    if args.npuw_llm_generate_attention_hint:
        optional["NPUW_LLM_GENERATE_ATTENTION_HINT"] = args.npuw_llm_generate_attention_hint
    if args.npuw_llm_propagate_slice_up:
        optional["NPUW_LLM_PROPAGATE_SLICE_UP"] = args.npuw_llm_propagate_slice_up
    if args.npuw_llm_enable_prefix_caching:
        optional["NPUW_LLM_ENABLE_PREFIX_CACHING"] = args.npuw_llm_enable_prefix_caching
    if args.npuw_llm_enable_continuous_prefill:
        optional["NPUW_LLM_ENABLE_CONTINUOUS_PREFILL"] = args.npuw_llm_enable_continuous_prefill
    if args.npuw_llm_enable_block_based_kv_cache:
        optional["NPUW_LLM_ENABLE_BLOCK_BASED_KV_CACHE"] = args.npuw_llm_enable_block_based_kv_cache
    return optional


def select_pipeline_properties(device: DeviceType, cpu_properties: dict[str, Any],
                               gpu_properties: dict[str, Any],
                               npu_properties: dict[str, Any]) -> dict[str, Any]:
    """Pick the property set of the target device.

    ``AUTO`` gets the CPU and the GPU properties together, so that the pipeline
    may fall back from the accelerator to the CPU.
    """
    if device == DeviceType.NPU:
        return npu_properties
    if device == DeviceType.CPU:
        return cpu_properties
    if device == DeviceType.AUTO:
        return cpu_properties | gpu_properties
    return gpu_properties


def apply_common_properties(properties: dict[str, Any], kv_cache_precision: str | None,
                             attention_backend: str | None) -> dict[str, Any]:
    """Add the properties that are not bound to a single device."""
    if kv_cache_precision:
        properties["KV_CACHE_PRECISION"] = kv_cache_precision
    if attention_backend:
        properties["ATTENTION_BACKEND"] = attention_backend
    return properties


def device_value(device: DeviceType) -> str:
    """The device name as the pipeline factory expects it."""
    return enum_value(device)
