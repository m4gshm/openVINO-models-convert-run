"""Tests of chat content parts parsing into text and media segments."""
import unittest

from pydantic import TypeAdapter

from agent.multimodal.content_parts import (content_text, has_media, part_media_ref, segments_of)
from agent.multimodal.media_error import MediaPartError
from agent.multimodal.modality import Modality
from agent.openai.chat_completions_api import ChatCompletionMessageParam

PNG_DATA_URL = "data:image/png;base64,aGVsbG8="


class SegmentsOfCase(unittest.TestCase):
    """Tests of segments_of/content_text/has_media over content part shapes."""

    def test_string_content_is_one_text_segment(self):
        """A plain string content is a single text segment without media."""
        self.assertEqual("hello", content_text("hello"))
        self.assertFalse(has_media("hello"))

    def test_empty_content_has_no_segments(self):
        """Missing or empty content yields no segments."""
        self.assertEqual([], segments_of(None))
        self.assertEqual([], segments_of(""))

    def test_text_parts_are_joined(self):
        """Multiple text parts are joined with a space."""
        content = [{"type": "text", "text": "a "}, {"type": "text", "text": "b"}]
        self.assertEqual("a b", content_text(content))

    def test_refusal_part_is_kept_as_text(self):
        """Refusal parts are treated as text."""
        content = [{"type": "refusal", "refusal": "no"}]
        self.assertEqual("no", content_text(content))

    def test_image_url_part(self):
        """OpenAI image_url parts produce an image media reference."""
        content = [{"type": "image_url", "image_url": {"url": PNG_DATA_URL, "detail": "high"}}]
        segments = segments_of(content)
        self.assertTrue(has_media(content))
        ref = segments[0].media
        self.assertEqual(Modality.IMAGE, ref.modality)
        self.assertEqual(PNG_DATA_URL, ref.source)

    def test_input_audio_part(self):
        """OpenAI input_audio parts produce an audio media reference."""
        content = [{"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "wav"}}]
        ref = segments_of(content)[0].media
        self.assertEqual(Modality.AUDIO, ref.modality)
        self.assertEqual("aGVsbG8=", ref.data)
        self.assertEqual("wav", ref.fmt)

    def test_hugging_face_style_image_part(self):
        """HuggingFace-style image parts produce an image media reference."""
        content = [{"type": "image", "image": "https://example.com/pic.png"}]
        self.assertEqual("https://example.com/pic.png", part_media_ref(content[0]).source)

    def test_video_part(self):
        """video_url parts produce a video media reference."""
        content = [{"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}}]
        self.assertEqual(Modality.VIDEO, segments_of(content)[0].media.modality)

    def test_text_and_media_order_is_preserved(self):
        """The order of text and media parts is preserved."""
        content = [{"type": "text", "text": "before "},
                   {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
                   {"type": "text", "text": " after"}]
        kinds = [("media" if segment.is_media else "text") for segment in segments_of(content)]
        self.assertEqual(["text", "media", "text"], kinds)
        self.assertEqual("before  after", content_text(content))

    def test_unknown_part_type_is_rejected(self):
        """Unknown part types raise MediaPartError."""
        with self.assertRaises(MediaPartError):
            segments_of([{"type": "file", "file": {"url": "https://example.com/doc.pdf"}}])

    def test_media_part_without_source_is_rejected(self):
        """Media parts without any source raise MediaPartError."""
        with self.assertRaises(MediaPartError):
            segments_of([{"type": "image_url", "image_url": {}}])

    def test_unexpected_content_type_is_rejected(self):
        """Non-list, non-string content raises MediaPartError."""
        with self.assertRaises(MediaPartError):
            segments_of(42)


class RequestMessageModelCase(unittest.TestCase):
    """Tests of request message validation keeping media parts."""

    adapter = TypeAdapter(list[ChatCompletionMessageParam])

    def test_media_parts_survive_request_validation(self):
        """Media parts survive ChatCompletionMessageParam validation and dumps."""
        messages = self.adapter.validate_json(
            '[{"role": "user", "content": ['
            '{"type": "text", "text": "what is on the picture?"},'
            '{"type": "image_url", "image_url": {"url": "' + PNG_DATA_URL + '"}},'
            '{"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "wav"}}]}]')
        dumped = messages[0].model_dump()
        self.assertTrue(has_media(dumped["content"]))
        self.assertEqual("what is on the picture?", content_text(dumped["content"]))

    def test_plain_text_message_is_not_affected(self):
        """Plain text messages pass through validation unchanged."""
        messages = self.adapter.validate_json('[{"role": "user", "content": "hi"}]')
        self.assertEqual("hi", messages[0].model_dump()["content"])

    def test_describe_hides_payload(self):
        """describe() never exposes the media payload."""
        ref = part_media_ref({"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "wav"}})
        self.assertNotIn("aGVsbG8=", ref.describe())
        self.assertIn("audio", ref.describe())


if __name__ == '__main__':
    unittest.main()
