"""Tests of modality detection from architectures and pipeline kind."""
import unittest

from agent.multimodal.media_error import MediaError
from agent.multimodal.modality import Modalities, Modality, detect_modalities


class DetectModalitiesCase(unittest.TestCase):
    """Tests of detect_modalities fallback heuristics."""

    def test_text_only_llm(self):
        """LLM pipe accepts text only."""
        detected = detect_modalities({"Qwen2ForCausalLM"}, "LLM")
        self.assertEqual(["text"], detected.names())

    def test_vlm_pipe_enables_images(self):
        """VLM pipe enables images but not video."""
        detected = detect_modalities({"Qwen3_5ForConditionalGeneration"}, "VLM")
        self.assertTrue(detected.supports(Modality.IMAGE))
        self.assertFalse(detected.supports(Modality.VIDEO))

    def test_vision_architecture_enables_images_without_pipe(self):
        """Vision architectures enable images even without a pipe."""
        self.assertTrue(detect_modalities({"Qwen2VLForConditionalGeneration"}).supports(Modality.IMAGE))

    def test_omni_architecture_enables_audio(self):
        """Omni architectures enable audio (and images)."""
        detected = detect_modalities({"Qwen3OmniMoeForConditionalGeneration"}, "VLM")
        self.assertTrue(detected.supports(Modality.AUDIO))
        self.assertTrue(detected.supports(Modality.IMAGE))

    def test_continuous_batching_has_no_audio(self):
        """CB pipe has no audio inputs, images stay enabled."""
        detected = detect_modalities({"Qwen3OmniMoeForConditionalGeneration"}, "CB")
        self.assertFalse(detected.supports(Modality.AUDIO))
        self.assertTrue(detected.supports(Modality.IMAGE))

    def test_empty_architectures(self):
        """Empty architectures yield text only."""
        self.assertEqual(["text"], detect_modalities(set(), None).names())


class ModalitiesCase(unittest.TestCase):
    """Tests of the Modalities container."""

    def test_of_always_includes_text(self):
        """of() always includes text alongside media."""
        modalities = Modalities.of(Modality.IMAGE, "audio")
        self.assertEqual({"audio", "image", "text"}, set(modalities.names()))

    def test_media_names_excludes_text(self):
        """media_names() excludes text."""
        self.assertEqual(["audio"], Modalities.of("text", "audio").media_names())


class MediaErrorCase(unittest.TestCase):
    """Tests of the MediaError payload."""

    def test_error_keeps_modality(self):
        """MediaError keeps its modality attribute."""
        error = MediaError("boom", modality=Modality.IMAGE.value)
        self.assertEqual("boom", error.message)
        self.assertEqual("image", error.modality)


if __name__ == '__main__':
    unittest.main()
