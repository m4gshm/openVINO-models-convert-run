import contextlib
import json
import tempfile
import unittest
from pathlib import Path

from agent.multimodal.model_capabilities import (MediaCapabilities, read_media_capabilities,
                                                 resolve_modalities)
from agent.multimodal.modality import Modality


@contextlib.contextmanager
def tempfile_directory():
    """Empty temporary directory as a model folder stand-in."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def make_dir(root: Path, files: dict[str, object]) -> Path:
    """Create model folder files; dict value is file content (dict is dumped as JSON)."""
    for name, content in files.items():
        text = json.dumps(content) if isinstance(content, (dict, list)) else str(content)
        (root / name).write_text(text, encoding="utf-8")
    return root


VLM_FILES = {
    "openvino_tokenizer.xml": "x", "openvino_detokenizer.xml": "x",
    "openvino_language_model.xml": "x", "openvino_text_embeddings_model.xml": "x",
    "openvino_vision_embeddings_model.xml": "x", "openvino_vision_embeddings_model.bin": "x",
    "openvino_vision_embeddings_merger_model.xml": "x",
    "config.json": {"architectures": ["Gemma4UnifiedForConditionalGeneration"],
                    "video_token_id": 100, "vision_config": {"patch_size": 16}},
}

AUDIO_PROCESSOR = {"feature_extractor": {"feature_extractor_type": "AudioExtractor",
                                         "sampling_rate": 24000},
                   "image_processor": {"image_processor_type": "ImageProcessor"}}


class ReadMediaCapabilitiesCase(unittest.TestCase):
    def test_vision_and_audio_ir_files(self):
        with tempfile_directory() as root:
            make_dir(root, VLM_FILES | {"openvino_audio_embeddings_model.xml": "x"})
            capabilities = read_media_capabilities(root)
            self.assertTrue(capabilities.image)
            self.assertTrue(capabilities.audio)
            self.assertTrue(capabilities.video)

    def test_vision_only_export(self):
        with tempfile_directory() as root:
            make_dir(root, {"openvino_language_model.xml": "x",
                            "openvino_vision_embeddings_pos_model.xml": "x"})
            capabilities = read_media_capabilities(root)
            self.assertTrue(capabilities.image)
            self.assertFalse(capabilities.audio)

    def test_text_only_export(self):
        with tempfile_directory() as root:
            make_dir(root, {"openvino_model.xml": "x", "openvino_tokenizer.xml": "x",
                            "config.json": {"architectures": ["Qwen2ForCausalLM"]}})
            capabilities = read_media_capabilities(root)
            self.assertFalse(capabilities.has_media)
            self.assertEqual("text only", capabilities.describe())

    def test_config_json_fallback_without_ir(self):
        with tempfile_directory() as root:
            make_dir(root, {"config.json": {"architectures": ["Qwen2AudioForConditionalGeneration"],
                                            "audio_config": {"d_model": 1280},
                                            "image_token_id": 151655}})
            capabilities = read_media_capabilities(root)
            self.assertTrue(capabilities.image)
            self.assertTrue(capabilities.audio)

    def test_nested_thinker_config_audio(self):
        with tempfile_directory() as root:
            make_dir(root, {"config.json": {"thinker_config": {"audio_config": {"d_model": 1280}}}})
            self.assertTrue(read_media_capabilities(root).audio)

    def test_processor_config_sample_rate(self):
        with tempfile_directory() as root:
            make_dir(root, {"processor_config.json": AUDIO_PROCESSOR})
            capabilities = read_media_capabilities(root)
            self.assertTrue(capabilities.image)
            self.assertTrue(capabilities.audio)
            self.assertEqual(24000, capabilities.audio_sample_rate)

    def test_single_model_file_is_not_scanned(self):
        with tempfile_directory() as root:
            make_dir(root, VLM_FILES)
            gguf = root / "model.gguf"
            gguf.write_bytes(b"GGUF")
            self.assertFalse(read_media_capabilities(gguf).has_media)

    def test_missing_folder(self):
        with tempfile_directory() as root:
            self.assertFalse(read_media_capabilities(root / "absent").has_media)

    def test_broken_config_json_is_ignored(self):
        with tempfile_directory() as root:
            make_dir(root, {"config.json": "{ not json"})
            self.assertFalse(read_media_capabilities(root).has_media)


class ToModalitiesCase(unittest.TestCase):
    capabilities = MediaCapabilities(image=True, audio=True, video=True, audio_sample_rate=16000)

    def test_vlm_pipe_takes_image_and_audio(self):
        self.assertEqual({"audio", "image", "text"}, set(self.capabilities.to_modalities("VLM").names()))

    def test_llm_pipe_takes_text_only(self):
        self.assertEqual(["text"], self.capabilities.to_modalities("LLM").names())

    def test_cb_pipe_has_no_audio(self):
        self.assertEqual({"image", "text"}, set(self.capabilities.to_modalities("CB").names()))

    def test_video_is_never_advertised(self):
        self.assertFalse(self.capabilities.to_modalities("VLM").supports(Modality.VIDEO))


class ResolveModalitiesCase(unittest.TestCase):
    def test_model_capabilities_are_preferred_over_architecture_name(self):
        # architecture says audio, exported folder has no audio encoder
        capabilities = MediaCapabilities(image=True, audio=False)
        resolved = resolve_modalities(capabilities, {"Qwen3OmniForConditionalGeneration"}, "VLM")
        self.assertFalse(resolved.supports(Modality.AUDIO))
        self.assertTrue(resolved.supports(Modality.IMAGE))

    def test_architecture_fallback_without_media_files(self):
        resolved = resolve_modalities(MediaCapabilities.text_only(), {"Qwen2VLForConditionalGeneration"}, "VLM")
        self.assertTrue(resolved.supports(Modality.IMAGE))

    def test_llm_pipe_narrows_capabilities_to_text(self):
        resolved = resolve_modalities(MediaCapabilities(image=True, audio=True), set(), "LLM")
        self.assertEqual(["text"], resolved.names())


if __name__ == '__main__':
    unittest.main()
