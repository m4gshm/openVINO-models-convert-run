import base64
import tempfile
import unittest
from pathlib import Path

from agent.multimodal.content_parts import MediaRef
from agent.multimodal.media_error import MediaLimitError, MediaSourceError
from agent.multimodal.media_sources import (decode_base64, format_of_media_type, looks_like_base64, resolve_bytes,
                                            sniff_format)
from agent.multimodal.modality import Modality

PAYLOAD = bytes(range(0, 128, 4))
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
WAV_BYTES = b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 16


def media_ref(source: str | None = None, data: str | None = None,
              media_type: str | None = None, fmt: str | None = None) -> MediaRef:
    return MediaRef(modality=Modality.IMAGE, source=source, data=data, media_type=media_type, fmt=fmt)


class ResolveBytesCase(unittest.TestCase):
    def test_data_url_with_base64(self):
        encoded = base64.b64encode(PAYLOAD).decode()
        declared, data = resolve_bytes(media_ref(source=f"data:image/png;base64,{encoded}"))
        self.assertEqual(PAYLOAD, data)
        self.assertEqual("image/png", declared)

    def test_data_url_without_base64_is_percent_decoded(self):
        declared, data = resolve_bytes(media_ref(source="data:text/plain,hello%20world"))
        self.assertEqual(b"hello world", data)
        self.assertEqual("text/plain", declared)

    def test_inline_base64_data(self):
        _, data = resolve_bytes(media_ref(data=base64.b64encode(PAYLOAD).decode()))
        self.assertEqual(PAYLOAD, data)

    def test_data_url_size_limit(self):
        encoded = base64.b64encode(PAYLOAD).decode()
        with self.assertRaises(MediaLimitError):
            resolve_bytes(media_ref(source=f"data:image/png;base64,{encoded}"), max_bytes=8)

    def test_local_file_is_rejected_by_default(self):
        with self.assertRaises(MediaSourceError):
            resolve_bytes(media_ref(source="C:/tmp/pic.png"))

    def test_local_file_when_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "pic.png"
            file_path.write_bytes(PNG_BYTES)
            _, data = resolve_bytes(media_ref(source=str(file_path)), allow_local_files=True)
            self.assertEqual(PNG_BYTES, data)

    def test_missing_local_file(self):
        with self.assertRaises(MediaSourceError):
            resolve_bytes(media_ref(source="C:/definitely/not/there.png"), allow_local_files=True)

    def test_unsupported_scheme(self):
        with self.assertRaises(MediaSourceError):
            resolve_bytes(media_ref(source="ftp://example.com/pic.png"))

    def test_unreachable_url_is_reported(self):
        with self.assertRaises(MediaSourceError):
            resolve_bytes(media_ref(source="http://127.0.0.1:1/pic.png"), http_timeout=1.0)

    def test_reference_without_any_data(self):
        with self.assertRaises(MediaSourceError):
            resolve_bytes(media_ref())

    def test_malformed_data_url(self):
        with self.assertRaises(MediaSourceError):
            resolve_bytes(media_ref(source="data:image/png"))


class FormatDetectionCase(unittest.TestCase):
    def test_sniff_png(self):
        self.assertEqual("png", sniff_format(PNG_BYTES))

    def test_sniff_wav(self):
        self.assertEqual("wav", sniff_format(WAV_BYTES))

    def test_sniff_unknown(self):
        self.assertIsNone(sniff_format(b"no signature here"))

    def test_media_type_mapping(self):
        self.assertEqual("wav", format_of_media_type("audio/wav; codecs=0"))
        self.assertEqual("jpeg", format_of_media_type("image/jpeg"))
        self.assertIsNone(format_of_media_type(None))

    def test_base64_detection(self):
        self.assertTrue(looks_like_base64(base64.b64encode(PAYLOAD).decode()))
        self.assertFalse(looks_like_base64("https://example.com/a.png"))

    def test_decode_base64_without_padding(self):
        self.assertEqual(b"Man", decode_base64(base64_without_padding()))


def base64_without_padding() -> str:
    return base64.b64encode(b"Man").decode().rstrip("=")


if __name__ == '__main__':
    unittest.main()
