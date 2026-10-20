"""Tests of the model configuration detection: architectures, parser and pipeline."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.openvino.model_config import default_pipe_of, detect_parser_type, or_default_pipe, \
    parser_type_of, read_model_config
from agent.openvino.options import ParserType, Pipe


def write_config(folder: Path, config: dict) -> Path:
    """Write a config.json into the folder and return the folder."""
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return folder


class ReadModelConfigCase(unittest.TestCase):
    """Tests of the reading of the model config.json."""

    def test_architectures_are_read(self):
        """The architecture names of the config land in the model config."""
        with TemporaryDirectory() as tmp:
            model_dir = write_config(Path(tmp), {"architectures": ["Qwen3MoeForCausalLM"]})
            config = read_model_config(model_dir)
        self.assertEqual({"Qwen3MoeForCausalLM"}, config.architectures)

    def test_nested_text_config_provides_the_context_length(self):
        """A multimodal config nests the context length under text_config."""
        with TemporaryDirectory() as tmp:
            model_dir = write_config(Path(tmp), {"text_config": {"max_position_embeddings": 32768}})
            config = read_model_config(model_dir)
        self.assertEqual(32768, config.max_position_embeddings)

    def test_flat_context_length_is_read(self):
        """An old config keeps the context length at the top level."""
        with TemporaryDirectory() as tmp:
            model_dir = write_config(Path(tmp), {"max_position_embeddings": 4096})
            config = read_model_config(model_dir)
        self.assertEqual(4096, config.max_position_embeddings)

    def test_missing_config_yields_an_empty_configuration(self):
        """A folder without a config.json is not an error."""
        with TemporaryDirectory() as tmp:
            config = read_model_config(Path(tmp))
        self.assertEqual(set(), config.architectures)
        self.assertIsNone(config.max_position_embeddings)

    def test_malformed_config_does_not_raise(self):
        """A broken config.json must not abort the startup."""
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text("{not json", encoding="utf-8")
            config = read_model_config(model_dir)
        self.assertEqual(set(), config.architectures)


class DetectParserTypeCase(unittest.TestCase):
    """Tests of the parser selection by the architecture of the model."""

    def test_gemma4_is_detected(self):
        """A gemma4 export uses the channel parser."""
        self.assertEqual(ParserType.gemma4, detect_parser_type({"Gemma4ForConditionalGeneration"}))

    def test_qwen3_moe_is_detected(self):
        """A qwen3 moe export uses the qwen3 parser."""
        self.assertEqual(ParserType.qwen3moe, detect_parser_type({"Qwen3MoeForCausalLM"}))

    def test_qwen3_5_is_detected(self):
        """A qwen3.5 export uses the qwen3 parser."""
        self.assertEqual(ParserType.qwen3moe, detect_parser_type({"Qwen3_5ForConditionalGeneration"}))

    def test_qwen2_is_detected(self):
        """A qwen2 export uses the qwen2 parser."""
        self.assertEqual(ParserType.qwen2, detect_parser_type({"Qwen2ForCausalLM"}))

    def test_qwen3_is_detected(self):
        """A qwen3 export uses the qwen2 parser."""
        self.assertEqual(ParserType.qwen2, detect_parser_type({"Qwen3ForCausalLM"}))

    def test_lfm2_is_detected(self):
        """A lfm2 export uses the lfm2 parser."""
        self.assertEqual(ParserType.lfm2, detect_parser_type({"Lfm2ForCausalLM"}))

    def test_unknown_architecture_has_no_parser(self):
        """An unknown architecture leaves the parser selection open."""
        self.assertIsNone(detect_parser_type({"SomethingElseForCausalLM"}))


class ParserTypeOfCase(unittest.TestCase):
    """Tests of the conversion of the CLI parser name."""

    def test_known_name_is_converted(self):
        """A known CLI value becomes the matching parser type."""
        self.assertEqual(ParserType.qwen2, parser_type_of("qwen2"))

    def test_unknown_name_is_rejected(self):
        """An unknown CLI value leaves the parser selection open."""
        self.assertIsNone(parser_type_of("nonexistent"))

    def test_empty_name_is_rejected(self):
        """An empty CLI value leaves the parser selection open."""
        self.assertIsNone(parser_type_of(""))


class DefaultPipeCase(unittest.TestCase):
    """Tests of the default pipeline of a parser family."""

    def test_gemma4_defaults_to_vlm(self):
        """A gemma4 export is multimodal and is served as a VLM."""
        self.assertEqual(Pipe.VLM, default_pipe_of(ParserType.gemma4, {"Gemma4ForConditionalGeneration"}))

    def test_qwen3_5_defaults_to_vlm(self):
        """A qwen3.5 export is multimodal and is served as a VLM."""
        self.assertEqual(Pipe.VLM,
                         default_pipe_of(ParserType.qwen3moe, {"Qwen3_5ForConditionalGeneration"}))

    def test_text_only_family_defaults_to_llm(self):
        """A text-only export is served as a plain LLM pipeline."""
        self.assertEqual(Pipe.LLM, default_pipe_of(ParserType.qwen2, {"Qwen2ForCausalLM"}))

    def test_or_default_pipe_keeps_the_set_pipe(self):
        """An explicitly requested pipe is not overridden."""
        self.assertEqual(Pipe.CB, or_default_pipe(Pipe.CB, Pipe.LLM))

    def test_or_default_pipe_fills_the_missing_pipe(self):
        """A missing pipe falls back to the default one."""
        self.assertEqual(Pipe.LLM, or_default_pipe(None, Pipe.LLM))


if __name__ == '__main__':
    unittest.main()
