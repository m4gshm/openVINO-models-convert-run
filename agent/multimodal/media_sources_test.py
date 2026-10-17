"""Tests of media source resolution: data urls, base64, local files."""
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
    """Build an image MediaRef with the given fields."""
    return MediaRef(modality=Modality.IMAGE, source=source, data=data, media_type=media_type, fmt=fmt)


class ResolveBytesCase(unittest.TestCase):
    """Tests of resolve_bytes source handling and limits."""

    def test_data_url_with_base64(self):
        """A base64 data url resolves to decoded payload and media type."""
        encoded = base64.b64encode(PAYLOAD).decode()
        declared, data = resolve_bytes(media_ref(source=f"data:image/png;base64,{encoded}"))
        self.assertEqual(PAYLOAD, data)
        self.assertEqual("image/png", declared)

    def test_data_url_without_base64_is_percent_decoded(self):
        """A non-base64 data url is percent-decoded."""
        declared, data = resolve_bytes(media_ref(source="data:text/plain,hello%20world"))
        self.assertEqual(b"hello world", data)
        self.assertEqual("text/plain", declared)

    def test_inline_base64_data(self):
        """Inline base64 data resolves to the decoded payload."""
        _, data = resolve_bytes(media_ref(data=base64.b64encode(PAYLOAD).decode()))
        self.assertEqual(PAYLOAD, data)

    def test_data_url_size_limit(self):
        """Payloads over the byte limit raise MediaLimitError."""
        encoded = base64.b64encode(PAYLOAD).decode()
        with self.assertRaises(MediaLimitError):
            resolve_bytes(media_ref(source=f"data:image/png;base64,{encoded}"), max_bytes=8)

    def test_local_file_is_rejected_by_default(self):
        """Local paths are rejected unless explicitly allowed."""
        with self.assertRaises(MediaSourceError):
            resolve_bytes(media_ref(source="C:/tmp/pic.png"))

    def test_local_file_when_allowed(self):
        """Local files load when allowed."""
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "pic.png"
            file_path.write_bytes(PNG_BYTES)
            _, data = resolve_bytes(media_ref(source=str(file_path)), allow_local_files=True)
            self.assertEqual(PNG_BYTES, data)

    def test_missing_local_file(self):
        """Missing local files raise MediaSourceError."""
        with self.assertRaises(MediaSourceError):
            resolve_bytes(media_ref(source="C:/definitely/not/there.png"), allow_local_files=True)

    def test_unsupported_scheme(self):
        """Unsupported url schemes raise MediaSourceError."""
        with self.assertRaises(MediaSourceError):
            resolve_bytes(media_ref(source="ftp://example.com/pic.png"))

    def test_unreachable_url_is_reported(self):
        """Unreachable urls raise MediaSourceError."""
        with self.assertRaises(MediaSourceError):
            resolve_bytes(media_ref(source="http://127.0.0.1:1/pic.png"), http_timeout=1.0)

    def test_reference_without_any_data(self):
        """References without source or data raise MediaSourceError."""
        with self.assertRaises(MediaSourceError):
            resolve_bytes(media_ref())

    def test_malformed_data_url(self):
        """Malformed data urls raise MediaSourceError."""
        with self.assertRaises(MediaSourceError):
            resolve_bytes(media_ref(source="data:image/png"))


class FormatDetectionCase(unittest.TestCase):
    """Tests of format sniffing and base64 helpers."""

    def test_sniff_png(self):
        """PNG bytes sniff as png."""
        self.assertEqual("png", sniff_format(PNG_BYTES))

    def test_sniff_wav(self):
        """WAV bytes sniff as wav."""
        self.assertEqual("wav", sniff_format(WAV_BYTES))

    def test_sniff_unknown(self):
        """Unknown payloads sniff as None."""
        self.assertIsNone(sniff_format(b"no signature here"))

    def test_media_type_mapping(self):
        """Media types map to formats, parameters ignored."""
        self.assertEqual("wav", format_of_media_type("audio/wav; codecs=0"))
        self.assertEqual("jpeg", format_of_media_type("image/jpeg"))
        self.assertIsNone(format_of_media_type(None))

    def test_base64_detection(self):
        """looks_like_base64 distinguishes base64 from urls."""
        self.assertTrue(looks_like_base64(base64.b64encode(PAYLOAD).decode()))
        self.assertFalse(looks_like_base64("https://example.com/a.png"))

    def test_decode_base64_without_padding(self):
        """decode_base64 tolerates missing padding."""
        self.assertEqual(b"Man", decode_base64(base64_without_padding()))


def base64_without_padding() -> str:
    """Return base64 of 'Man' with padding stripped."""
    return base64.b64encode(b"Man").decode().rstrip("=")


if __name__ == '__main__':
    unittest.main()
