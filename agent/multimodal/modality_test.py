import unittest

from agent.multimodal.media_error import MediaError
from agent.multimodal.modality import Modalities, Modality, detect_modalities


class DetectModalitiesCase(unittest.TestCase):
    def test_text_only_llm(self):
        detected = detect_modalities({"Qwen2ForCausalLM"}, "LLM")
        self.assertEqual(["text"], detected.names())

    def test_vlm_pipe_enables_images(self):
        detected = detect_modalities({"Qwen3_5ForConditionalGeneration"}, "VLM")
        self.assertTrue(detected.supports(Modality.IMAGE))
        self.assertFalse(detected.supports(Modality.VIDEO))

    def test_vision_architecture_enables_images_without_pipe(self):
        self.assertTrue(detect_modalities({"Qwen2VLForConditionalGeneration"}).supports(Modality.IMAGE))

    def test_omni_architecture_enables_audio(self):
        detected = detect_modalities({"Qwen3OmniMoeForConditionalGeneration"}, "VLM")
        self.assertTrue(detected.supports(Modality.AUDIO))
        self.assertTrue(detected.supports(Modality.IMAGE))

    def test_continuous_batching_has_no_audio(self):
        detected = detect_modalities({"Qwen3OmniMoeForConditionalGeneration"}, "CB")
        self.assertFalse(detected.supports(Modality.AUDIO))
        self.assertTrue(detected.supports(Modality.IMAGE))

    def test_empty_architectures(self):
        self.assertEqual(["text"], detect_modalities(set(), None).names())


class ModalitiesCase(unittest.TestCase):
    def test_of_always_includes_text(self):
        modalities = Modalities.of(Modality.IMAGE, "audio")
        self.assertEqual({"audio", "image", "text"}, set(modalities.names()))

    def test_media_names_excludes_text(self):
        self.assertEqual(["audio"], Modalities.of("text", "audio").media_names())


class MediaErrorCase(unittest.TestCase):
    def test_error_keeps_modality(self):
        error = MediaError("boom", modality=Modality.IMAGE.value)
        self.assertEqual("boom", error.message)
        self.assertEqual("image", error.modality)


if __name__ == '__main__':
    unittest.main()
