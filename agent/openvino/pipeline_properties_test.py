"""Tests of the pipeline properties of the OpenVINO engines."""
import unittest
from argparse import Namespace

from agent.openvino.options import DeviceType
from agent.openvino.pipeline_properties import apply_common_properties, build_cpu_properties, \
    build_gpu_properties, build_npu_properties, device_value, optional_npu_properties, \
    select_pipeline_properties


def new_args(**kwargs) -> Namespace:
    """Arguments of the pipeline properties with sane defaults."""
    defaults = {
        "performance_hint": "LATENCY",
        "gpu_enable_large_allocations": "YES",
        "gpu_priorities": "HIGH",
        "npu_compiler_type": "PREFER_PLUGIN",
        "npu_turbo": "YES",
        "npu_generate_hint": "BEST_PERF",
        "npu_prefill_hint": "DYNAMIC",
        "npuw_llm_generate_pyramid": "YES",
        "npuw_llm_prefill_chunk_size": 1024,
        "npuw_llm": None,
        "npuw_pa": None,
        "npuw_devices": None,
        "npu_compilation_mode_params": None,
        "npuw_llm_prefill_moe_hint": None,
        "npuw_llm_generate_moe_hint": None,
        "npuw_llm_prefill_attention_hint": None,
        "npuw_llm_generate_attention_hint": None,
        "npuw_llm_propagate_slice_up": None,
        "npuw_llm_enable_prefix_caching": None,
        "npuw_llm_enable_continuous_prefill": None,
        "npuw_llm_enable_block_based_kv_cache": None,
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


class CpuPropertiesCase(unittest.TestCase):
    """Tests of the CPU pipeline properties."""

    def test_cache_dir_and_hint_are_set(self):
        """The cache directory and the performance hint reach the properties."""
        properties = build_cpu_properties("./cache", new_args())
        self.assertEqual("./cache", properties["CACHE_DIR"])
        self.assertEqual("LATENCY", properties["PERFORMANCE_HINT"])
        self.assertEqual("YES", properties["ENABLE_MMAP"])
        self.assertIn("INFERENCE_NUM_THREADS", properties)


class GpuPropertiesCase(unittest.TestCase):
    """Tests of the GPU pipeline properties."""

    def test_queue_priority_is_mapped(self):
        """The GPU priorities reach the queue and the model priority."""
        properties = build_gpu_properties("./cache", new_args())
        self.assertEqual("HIGH", properties["GPU_QUEUE_PRIORITY"])
        self.assertEqual("HIGH", properties["MODEL_PRIORITY"])
        self.assertEqual("YES", properties["GPU_ENABLE_LARGE_ALLOCATIONS"])


class NpuPropertiesCase(unittest.TestCase):
    """Tests of the NPU pipeline properties."""

    def test_npuw_knobs_are_present(self):
        """The base NPUW properties are always set."""
        properties = build_npu_properties("./cache", new_args(), 4096)
        self.assertEqual("PREFER_PLUGIN", properties["NPU_COMPILER_TYPE"])
        self.assertEqual("YES", properties["NPU_USE_NPUW"])
        self.assertEqual("BEST_PERF", properties["NPUW_LLM_GENERATE_HINT"])
        self.assertEqual(4096, properties["MAX_PROMPT_LEN"])

    def test_max_prompt_len_is_omitted_when_unset(self):
        """An unset context length does not reach the properties."""
        properties = build_npu_properties("./cache", new_args(), None)
        self.assertNotIn("MAX_PROMPT_LEN", properties)

    def test_optional_knobs_are_added_when_asked_for(self):
        """An optional knob reaches the properties only when it is set."""
        properties = build_npu_properties("./cache", new_args(npuw_llm_enable_prefix_caching="YES"), None)
        self.assertEqual("YES", properties["NPUW_LLM_ENABLE_PREFIX_CACHING"])


class OptionalNpuPropertiesCase(unittest.TestCase):
    """Tests of the optional NPUW knobs."""

    def test_unset_knobs_are_omitted(self):
        """A knob that was not asked for does not reach the properties."""
        self.assertEqual({}, optional_npu_properties(new_args()))

    def test_asked_knobs_are_mapped(self):
        """A knob that was asked for reaches the properties."""
        optional = optional_npu_properties(new_args(npuw_llm="YES", npuw_devices="NPU"))
        self.assertEqual("YES", optional["NPUW_LLM"])
        self.assertEqual("NPU", optional["NPUW_DEVICES"])


class SelectPipelinePropertiesCase(unittest.TestCase):
    """Tests of the selection of the properties of the device."""

    def test_npu_selects_the_npu_properties(self):
        """The NPU device uses the NPU properties."""
        selected = select_pipeline_properties(DeviceType.NPU, {"a": 1}, {"b": 2}, {"c": 3})
        self.assertEqual({"c": 3}, selected)

    def test_cpu_selects_the_cpu_properties(self):
        """The CPU device uses the CPU properties."""
        selected = select_pipeline_properties(DeviceType.CPU, {"a": 1}, {"b": 2}, {"c": 3})
        self.assertEqual({"a": 1}, selected)

    def test_gpu_selects_the_gpu_properties(self):
        """The GPU device uses the GPU properties."""
        selected = select_pipeline_properties(DeviceType.GPU, {"a": 1}, {"b": 2}, {"c": 3})
        self.assertEqual({"b": 2}, selected)

    def test_auto_merges_the_cpu_and_the_gpu_properties(self):
        """The AUTO device may fall back from the GPU to the CPU."""
        selected = select_pipeline_properties(DeviceType.AUTO, {"a": 1}, {"b": 2}, {"c": 3})
        self.assertEqual({"a": 1, "b": 2}, selected)


class CommonPropertiesCase(unittest.TestCase):
    """Tests of the device independent properties."""

    def test_kv_cache_precision_and_attention_backend_are_added(self):
        """The device independent knobs reach the properties."""
        properties = apply_common_properties({}, "u8", "PA")
        self.assertEqual("u8", properties["KV_CACHE_PRECISION"])
        self.assertEqual("PA", properties["ATTENTION_BACKEND"])

    def test_unset_knobs_are_omitted(self):
        """An unset knob does not reach the properties."""
        properties = apply_common_properties({}, None, None)
        self.assertNotIn("KV_CACHE_PRECISION", properties)
        self.assertNotIn("ATTENTION_BACKEND", properties)

    def test_device_value_is_the_raw_name(self):
        """The device reaches the pipeline factory as its raw name."""
        self.assertEqual("NPU", device_value(DeviceType.NPU))


if __name__ == '__main__':
    unittest.main()
