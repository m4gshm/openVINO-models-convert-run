"""Command line option enums of the OpenVINO mode.

The enums mirror the values accepted on the command line, so the CLI choices and
the internal representation stay in one place.
"""
from enum import Enum


class Turn(Enum):
    """Generic on/off switch values accepted on the command line."""
    # pylint: disable=invalid-name  # members mirror CLI values
    on = 'on'
    off = 'off'


class DeviceType(Enum):
    """Inference devices the pipeline can be compiled for."""
    GPU = 'GPU'
    NPU = 'NPU'
    CPU = 'CPU'
    AUTO = 'AUTO'


class Pipe(Enum):
    """Serving pipeline kinds exposed via --pipe."""
    CB = 'CB'
    VLM = 'VLM'
    LLM = 'LLM'


class ParserType(Enum):
    """Prompt parser families selectable via --parser."""
    # pylint: disable=invalid-name  # members mirror model family names
    qwen2 = 'qwen2'
    qwen3moe = 'qwen3moe'
    gemma4 = 'gemma4'
    lfm2 = 'lfm2'


class AttentionBackend(Enum):
    """Attention kernels selectable via --attention_backend."""
    PA = 'PA'
    SDPA = 'SDPA'


class KvCachePrecision(Enum):
    """KV cache data types accepted via --kv_cache_precision."""
    # pylint: disable=invalid-name  # members mirror KV cache dtypes
    u8 = 'u8'
    u4 = 'u4'
    f16 = 'f16'


class NpuCompilerType(Enum):
    """NPU compiler selection strategy."""
    DRIVER = 'DRIVER'
    PLUGIN = 'PLUGIN'
    PREFER_PLUGIN = 'PREFER_PLUGIN'


class YesNo(Enum):
    """Generic yes/no values accepted on the command line."""
    YES = 'YES'
    NO = 'NO'


class Level(Enum):
    """Priority levels for queue/model priorities."""
    HIGH = 'HIGH'
    MEDIUM = 'MEDIUM'


class NpuGenerateHint(Enum):
    """NPUW generate hint values."""
    BEST_PERF = 'BEST_PERF'
    FAST_COMPILE = 'FAST_COMPILE'


class PrefillHint(Enum):
    """NPUW prefill scheduling hint values."""
    DYNAMIC = 'DYNAMIC'
    STATIC = 'STATIC'


class AttentionHint(Enum):
    """Attention shape hints for NPUW LLM prefill/generate."""
    DYNAMIC = 'DYNAMIC'
    STATIC = 'STATIC'
    PYRAMID = 'PYRAMID'
    HFA = 'HFA'


class PerformanceHint(Enum):
    """OpenVINO performance hint values."""
    LATENCY = 'LATENCY'
    THROUGHPUT = 'THROUGHPUT'
    CUMULATIVE_THROUGHPUT = 'CUMULATIVE_THROUGHPUT'


class MoEHint(Enum):
    """Mixture-of-experts routing hint values."""
    DENSE = 'DENSE'
    HOST_ROUTED = 'HOST_ROUTED'
    DEVICE_ROUTED = 'DEVICE_ROUTED'
