"""Tests of the OpenVINO mode wiring: model path, options and engine setup."""
import unittest
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from openvino_genai import Tokenizer

from agent.multimodal.model_capabilities import MediaCapabilities
from agent.openvino.options import ParserType, Pipe
from agent.server_openvino import add_openvino_args, new_model_parser, resolve_attention_backend, \
    resolve_max_prompt_len, resolve_model_path, resolve_pipe_and_parser
from agent.openai import GenerateOpts


def new_args() -> ArgumentParser:
    """An argument parser with the arguments of the OpenVINO mode."""
    parser = ArgumentParser()
    add_openvino_args(parser)
    return parser


class ResolveModelPathCase(unittest.TestCase):
    """Tests of the resolution of the path of the model."""

    def test_relative_name_is_looked_up_in_the_models_dir(self):
        """A relative model name is looked up in the models directory."""
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "m"
            model_dir.mkdir()
            path, name = resolve_model_path("m", tmp)
        self.assertEqual(model_dir, path)
        self.assertEqual("m", name)

    def test_absolute_path_is_used_as_it_is(self):
        """An absolute model path is taken as it is."""
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "m"
            model_dir.mkdir()
            path, name = resolve_model_path(str(model_dir), "./models")
        self.assertEqual(model_dir, path)
        self.assertEqual("m", name)

    def test_file_path_drops_the_extension(self):
        """The name of a model file drops its extension."""
        with TemporaryDirectory() as tmp:
            model_file = Path(tmp) / "m.gguf"
            model_file.write_text("x", encoding="utf-8")
            path, name = resolve_model_path(str(model_file), "./models")
        self.assertEqual(model_file, path)
        self.assertEqual("m", name)


class ResolveMaxPromptLenCase(unittest.TestCase):
    """Tests of the resolution of the context length of the server."""

    def test_command_line_wins(self):
        """The context length of the command line wins."""
        self.assertEqual(1024, resolve_max_prompt_len(1024, GenerateOpts(max_prompt_tokens=4096), 8192))

    def test_options_file_is_used_next(self):
        """The context length of the options file is used when the command line has none."""
        self.assertEqual(4096, resolve_max_prompt_len(None, GenerateOpts(max_prompt_tokens=4096), 8192))

    def test_model_context_length_is_the_last_resort(self):
        """The context length of the model is used when nothing else sets it."""
        self.assertEqual(8192, resolve_max_prompt_len(None, GenerateOpts(), 8192))


class ResolvePipeAndParserCase(unittest.TestCase):
    """Tests of the selection of the pipeline and the parser of the model."""

    @staticmethod
    def parse(*cli_args: str) -> Namespace:
        """Parsed arguments of the OpenVINO mode."""
        return new_args().parse_args(["--model", "m", *cli_args])

    def test_explicit_pipe_wins(self):
        """An explicitly requested pipeline is not overridden."""
        args = self.parse("--pipe", "CB")
        pipe, _parser_type = resolve_pipe_and_parser(args, {"Qwen2ForCausalLM"}, _text_only())
        self.assertEqual(Pipe.CB, pipe)

    def test_parser_is_detected_by_the_architecture(self):
        """The parser of the model is detected by its architecture."""
        args = self.parse()
        _pipe, parser_type = resolve_pipe_and_parser(args, {"Qwen3MoeForCausalLM"}, _text_only())
        self.assertEqual(ParserType.qwen3moe, parser_type)

    def test_explicit_parser_is_respected(self):
        """An explicitly requested parser is used instead of the detected one."""
        args = self.parse("--parser", "qwen2")
        _pipe, parser_type = resolve_pipe_and_parser(args, {"Qwen3MoeForCausalLM"}, _text_only())
        self.assertEqual(ParserType.qwen2, parser_type)

    def test_multimodal_model_is_served_as_a_vlm(self):
        """A model with media encoders is served as a VLM."""
        args = self.parse()
        pipe, _parser_type = resolve_pipe_and_parser(args, set(), _with_media())
        self.assertEqual(Pipe.VLM, pipe)


class NewModelParserCase(unittest.TestCase):
    """Tests of the instantiation of the parser of a model family."""

    def test_qwen3moe_parser(self):
        """The qwen3 parser is instantiated for the qwen3 family."""
        self.assertEqual("Qwen3MoeParser", type(new_model_parser(ParserType.qwen3moe)).__name__)

    def test_gemma4_parser(self):
        """The gemma4 parser is instantiated for the gemma4 family."""
        self.assertEqual("Gemma4ChannelParser", type(new_model_parser(ParserType.gemma4)).__name__)

    def test_qwen2_parser(self):
        """The qwen2 parser is instantiated for the qwen2 family."""
        self.assertEqual("Qwen2Parser", type(new_model_parser(ParserType.qwen2)).__name__)

    def test_lfm2_parser(self):
        """The lfm2 parser is instantiated for the lfm2 family."""
        self.assertEqual("Lfm2Parser", type(new_model_parser(ParserType.lfm2)).__name__)

    def test_unknown_family_gets_the_base_parser(self):
        """An unknown family falls back to the base parser."""
        self.assertEqual("Parser", type(new_model_parser(None)).__name__)


class ResolveAttentionBackendCase(unittest.TestCase):
    """Tests of the selection of the attention kernel of the pipeline."""

    def test_pa_is_forced_for_the_plain_pipeline(self):
        """The plain CPU and GPU pipelines use the PA kernel."""
        self.assertEqual("PA", resolve_attention_backend(None, Pipe.LLM, False, None))

    def test_pa_is_not_forced_for_the_continuous_batching(self):
        """The continuous batching pipeline does not get the PA kernel."""
        self.assertIsNone(resolve_attention_backend(None, Pipe.CB, False, None))

    def test_pa_is_not_forced_for_the_npu(self):
        """The NPU pipeline does not get the PA kernel."""
        self.assertIsNone(resolve_attention_backend(None, Pipe.LLM, True, None))

    def test_pa_is_not_forced_for_gemma4(self):
        """The gemma4 pipeline does not get the PA kernel."""
        self.assertIsNone(resolve_attention_backend(None, Pipe.LLM, False, ParserType.gemma4))

    def test_explicit_backend_is_kept(self):
        """An explicitly requested kernel is used as it is."""
        self.assertEqual("SDPA", resolve_attention_backend("SDPA", Pipe.LLM, False, None))


def _text_only() -> MediaCapabilities:
    """Media capabilities of a model without media encoders."""
    return MediaCapabilities.text_only()


def _with_media() -> MediaCapabilities:
    """Media capabilities of a model with a vision encoder."""
    return MediaCapabilities(image=True)


if __name__ == '__main__':
    unittest.main()
