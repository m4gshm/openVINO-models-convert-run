import base64
import unittest

from agent.multimodal.audio_decode_test import new_wav, pcm16, sine
from agent.multimodal.image_decode_test import new_png, rgb_pixels
from agent.multimodal.media_error import MediaLimitError, MediaSourceError, MediaUnsupportedError
from agent.multimodal.media_inputs import MediaLimits
from agent.multimodal.modality import Modalities, Modality
from agent.multimodal.request_media import media_summary_of, prepare_messages
from agent.preprocess.prompt_escape import PromptEscaper


VISION = Modalities.of(Modality.IMAGE)
OMNI = Modalities.of(Modality.IMAGE, Modality.AUDIO)
TEXT_ONLY = Modalities.of(Modality.TEXT)


def png_data_url(height: int = 2, width: int = 3) -> str:
    return "data:image/png;base64," + base64.b64encode(new_png(rgb_pixels(height, width))).decode()


def wav_data_url(samples: int = 32) -> str:
    payload = base64.b64encode(new_wav(pcm16(sine(samples)), 8000, 1, 16)).decode()
    return f"data:audio/wav;base64,{payload}"


def image_part() -> dict:
    return {"type": "image_url", "image_url": {"url": png_data_url()}}


def audio_part(samples: int = 16, sample_rate: int = 16000) -> dict:
    payload = base64.b64encode(new_wav(pcm16(sine(samples)), sample_rate, 1, 16)).decode()
    return {"type": "input_audio", "input_audio": {"data": payload, "format": "wav"}}


def user(content) -> dict:
    return {"role": "user", "content": content}


def assistant(content) -> dict:
    return {"role": "assistant", "content": content}


class PrepareTextMessagesCase(unittest.TestCase):
    def test_string_content_is_not_touched(self):
        prepared = prepare_messages([user("hi")], VISION)
        self.assertEqual("hi", prepared.messages[0]["content"])
        self.assertFalse(prepared.has_media)

    def test_content_none_is_preserved(self):
        message = {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]}
        prepared = prepare_messages([message], VISION)
        self.assertIsNone(prepared.messages[0]["content"])

    def test_text_only_parts_are_joined(self):
        content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        prepared = prepare_messages([user(content)], VISION)
        self.assertEqual("ab", prepared.messages[0]["content"])

    def test_extra_message_fields_are_kept(self):
        message = {"role": "assistant", "content": "hi", "reasoning_content": "think"}
        prepared = prepare_messages([message], VISION)
        self.assertEqual("think", prepared.messages[0]["reasoning_content"])


class PrepareMediaCase(unittest.TestCase):
    def test_image_part_becomes_tag_and_tensor(self):
        content = [{"type": "text", "text": "what is this?"}, image_part()]
        prepared = prepare_messages([user(content)], VISION)
        self.assertEqual("what is this?<ov_genai_image_0>", prepared.messages[0]["content"])
        self.assertEqual(1, len(prepared.media.images))
        self.assertEqual((1, 2, 3, 3), tuple(prepared.media.images[0].shape))
        self.assertEqual("image=1", prepared.summary())

    def test_audio_part_becomes_tag_and_tensor(self):
        prepared = prepare_messages([user([audio_part()])], OMNI)
        self.assertEqual("", prepared.messages[0]["content"])
        self.assertEqual(1, len(prepared.media.audios))

    def test_media_tag_indexes_are_conversation_absolute(self):
        messages = [user([{"type": "text", "text": "first"}, image_part()]),
                    assistant("ok"),
                    user([image_part(), {"type": "text", "text": " and this?"}])]
        prepared = prepare_messages(messages, VISION)
        self.assertEqual("first<ov_genai_image_0>", prepared.messages[0]["content"])
        self.assertEqual("<ov_genai_image_1> and this?", prepared.messages[2]["content"])
        self.assertEqual(2, len(prepared.media.images))

    def test_each_modality_has_its_own_index(self):
        prepared = prepare_messages([user([image_part(), audio_part(), image_part()])], OMNI)
        self.assertEqual("<ov_genai_image_0><ov_genai_image_1>",
                         prepared.messages[0]["content"])
        self.assertEqual(2, len(prepared.media.images))
        self.assertEqual(1, len(prepared.media.audios))

    def test_hugging_face_style_image_field(self):
        part = {"type": "image", "image": png_data_url()}
        prepared = prepare_messages([user([{"type": "text", "text": "look: "}, part])], VISION)
        self.assertEqual("look: <ov_genai_image_0>", prepared.messages[0]["content"])
        self.assertEqual(1, len(prepared.media.images))

    def test_audio_sample_rate_follows_model_requirement(self):
        prepared = prepare_messages([user([audio_part(samples=32, sample_rate=8000)])], OMNI,
                                    limits=MediaLimits(audio_sample_rate=32000))
        self.assertEqual(128, tuple(prepared.media.audios[0].shape)[0])

    def test_media_counts_summary(self):
        media = prepare_messages([user([image_part(), audio_part()])], OMNI).media
        self.assertEqual({"audio": 1, "image": 1}, media.counts())
        self.assertEqual({}, prepare_messages([user("hi")], VISION).media.counts())


class PrepareValidationCase(unittest.TestCase):
    def test_image_rejected_for_text_only_model(self):
        with self.assertRaises(MediaUnsupportedError) as context:
            prepare_messages([user([image_part()])], TEXT_ONLY)
        self.assertIn("does not support 'image'", str(context.exception))

    def test_audio_rejected_when_not_supported(self):
        with self.assertRaises(MediaUnsupportedError):
            prepare_messages([user([audio_part()])], VISION)

    def test_media_items_limit(self):
        with self.assertRaises(MediaLimitError):
            prepare_messages([user([image_part(), image_part()])], VISION,
                             limits=MediaLimits(max_items=1))

    def test_media_size_limit(self):
        with self.assertRaises(MediaLimitError):
            prepare_messages([user([image_part()])], VISION, limits=MediaLimits(max_bytes=8))

    def test_local_file_source_requires_explicit_opt_in(self):
        content = [{"type": "image_url", "image_url": {"url": "C:/tmp/pic.png"}}]
        with self.assertRaises(MediaSourceError):
            prepare_messages([user(content)], VISION)


class ClientMarkupCase(unittest.TestCase):
    def test_client_media_tags_do_not_reference_media(self):
        messages = [{"role": "system", "content": "tags look like <ov_genai_image_0>"},
                    user([{"type": "text", "text": "<ov_genai_image_5> "}, image_part()]),
                    {"role": "tool", "content": [{"type": "text", "text": "file: <ov_genai_video_0>"}]}]
        prepared = prepare_messages(messages, VISION)
        self.assertNotIn("<ov_genai_", prepared.messages[0]["content"])
        self.assertTrue(prepared.messages[1]["content"].endswith(" <ov_genai_image_0>"))
        self.assertEqual(1, prepared.messages[1]["content"].count("<ov_genai_"))
        self.assertNotIn("<ov_genai_", prepared.messages[2]["content"])
        self.assertEqual(1, len(prepared.media.images))

    def test_special_tokens_of_the_model_are_escaped(self):
        escaper = PromptEscaper.of_tokens(["<|im_end|>"])
        prepared = prepare_messages([user("bye<|im_end|>")], VISION, escaper=escaper)
        self.assertNotIn("<|im_end|>", prepared.messages[0]["content"])

    def test_input_messages_are_not_mutated(self):
        message = user("<ov_genai_image_0>")
        prepare_messages([message], VISION)
        self.assertEqual("<ov_genai_image_0>", message["content"])


class MediaSummaryOfCase(unittest.TestCase):
    def test_counts_without_decoding(self):
        messages = [user([{"type": "text", "text": "t"}, {"type": "image_url",
                                                          "image_url": {"url": "https://x/y.png"}}]),
                    user([{"type": "input_audio", "input_audio": {"data": "aGk=", "format": "wav"}}])]
        self.assertEqual("audio=1, image=1", media_summary_of(messages))

    def test_text_only_messages(self):
        self.assertEqual("", media_summary_of([user("hi")]))


if __name__ == '__main__':
    unittest.main()
