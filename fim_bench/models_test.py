"""Tests of the FIM bench model registry and prompt building."""

import unittest

from fim_bench.models import MELLUM_STYLE, QWEN_STYLE, build_prompt, model_info


class ModelInfoCase(unittest.TestCase):
    """Tests of the model registry."""

    def test_known_model_is_returned_as_is(self):
        info = model_info("Qwen2.5-Coder-3B-nf4-sym-g-1-se-awq")
        self.assertEqual(QWEN_STYLE, info.style)

    def test_mellum_family_guess(self):
        self.assertEqual(MELLUM_STYLE, model_info("Mellum-4b-base").style)

    def test_unknown_family_falls_back_to_qwen(self):
        self.assertEqual(QWEN_STYLE, model_info("Some-New-Coder").style)


class BuildPromptCase(unittest.TestCase):
    """Tests of the FIM prompt assembly."""

    def test_qwen_style_prompt(self):
        info = model_info("Qwen2.5-Coder-3B-nf4-sym-g-1-se-awq")
        prompt = build_prompt(info, "PRE", "SUF")
        self.assertEqual("<|fim_prefix|>PRE<|fim_suffix|>SUF<|fim_middle|>", prompt)

    def test_mellum_style_prompt(self):
        prompt = build_prompt(model_info("Mellum-4b-base"), "PRE", "SUF")
        self.assertEqual("<fim_prefix>PRE<fim_suffix>SUF<fim_middle>", prompt)


if __name__ == "__main__":
    unittest.main()
