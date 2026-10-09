"""Resolution of media references into raw bytes.

A content part may carry media as a base64 data URL, as inline base64 payload, as an
``http(s)`` URL or as a local file path (the latter only when explicitly allowed).
"""

import base64
import binascii
import logging
import re
from pathlib import Path
from urllib.parse import unquote, unquote_to_bytes, urlparse

import httpx

from agent.multimodal.content_parts import MediaRef
from agent.multimodal.media_error import MediaLimitError, MediaSourceError

log = logging.getLogger(__name__)

DATA_URL_PREFIX = "data:"
DEFAULT_HTTP_TIMEOUT = 30.0

_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")
# A reference is treated as a URL only when it has a scheme with an authority: "http://", "file://".
_URL_SCHEME_RE = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]{1,15})://")

_MEDIA_TYPE_FORMATS = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/flac": "flac",
    "audio/ogg": "ogg",
    "audio/opus": "ogg",
    "audio/webm": "webm",
    "audio/aac": "aac",
    "audio/basic": "au",
    "audio/x-pcm": "pcm",
}

_SIGNATURE_FORMATS = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
    (b"II*\x00", "tiff"),
    (b"MM\x00*", "tiff"),
    (b"OggS", "ogg"),
    (b"fLaC", "flac"),
    (b"ID3", "mp3"),
    (b"\xff\xfb", "mp3"),
    (b"\xff\xf3", "mp3"),
    (b"\xff\xf2", "mp3"),
    (b"\x00\x00\x01\x00", "ico"),
    (b"RIFF", "wav"),
)


def format_of_media_type(media_type: str | None) -> str | None:
    """Map a MIME type onto a short format name."""
    if not media_type:
        return None
    return _MEDIA_TYPE_FORMATS.get(media_type.split(";")[0].strip().lower())


def sniff_format(data: bytes) -> str | None:
    """Detect a media container/codec by its signature bytes."""
    if len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if len(data) >= 12 and data[4:12] in (b"ftypM4A", b"ftypisom", b"ftypmp42"):
        return "m4a"
    for signature, media_format in _SIGNATURE_FORMATS:
        if data.startswith(signature):
            return media_format
    return None


def looks_like_base64(text: str) -> bool:
    """Whether a string is a raw base64 payload without any URL scheme."""
    stripped = text.strip()
    if len(stripped) < 16 or len(stripped) % 4 != 0:
        return False
    return bool(_BASE64_RE.match(stripped))


def decode_base64(payload: str) -> bytes:
    """Decode base64 text tolerating whitespace and missing padding."""
    cleaned = "".join(payload.split())
    padding = -len(cleaned) % 4
    try:
        return base64.b64decode(cleaned + "=" * padding)
    except (binascii.Error, ValueError) as e:
        raise MediaSourceError(f"base64 payload is not decodable: {e}") from e


def _parse_data_url(url: str) -> tuple[str | None, bytes]:
    media_type, separator, payload = url[len(DATA_URL_PREFIX):].partition(",")
    if not separator:
        raise MediaSourceError("malformed data URL: missing ',' separator")
    attributes = [attribute.strip().lower() for attribute in media_type.split(";")]
    is_base64 = "base64" in attributes
    declared_type = attributes[0] if attributes and attributes[0] not in ("", "base64") else None
    data = decode_base64(payload) if is_base64 else unquote_to_bytes(payload)
    return declared_type, data


def _fetch_url(url: str, timeout: float, max_bytes: int) -> bytes:
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            if response.status_code >= 400:
                raise MediaSourceError(f"media download failed: HTTP {response.status_code} for {url}")
            data = response.content
    except httpx.HTTPError as e:
        raise MediaSourceError(f"media download failed for {url}: {e}") from e
    if len(data) > max_bytes:
        raise MediaLimitError(f"media size {len(data)} bytes exceeds limit {max_bytes} bytes: {url}")
    return data


def _read_file(path: str, max_bytes: int) -> bytes:
    file_path = Path(path)
    if not file_path.is_file():
        raise MediaSourceError(f"media file does not exist: {path}")
    size = file_path.stat().st_size
    if size > max_bytes:
        raise MediaLimitError(f"media file size {size} bytes exceeds limit {max_bytes} bytes: {path}")
    try:
        return file_path.read_bytes()
    except OSError as e:
        raise MediaSourceError(f"cannot read media file {path}: {e}") from e


def resolve_bytes(ref: MediaRef, *, allow_local_files: bool = False,
                  max_bytes: int = 20 * 1024 * 1024,
                  http_timeout: float = DEFAULT_HTTP_TIMEOUT) -> tuple[str | None, bytes]:
    """Resolve a media reference into its declared media type (maybe None) and raw bytes.

    Raises:
        MediaSourceError: when the reference cannot be located or decoded.
        MediaLimitError: when the resolved payload exceeds ``max_bytes``.
    """
    payload = ref.data
    source = ref.source
    if payload:
        data = decode_base64(payload)
        if len(data) > max_bytes:
            raise MediaLimitError(f"media size {len(data)} bytes exceeds limit {max_bytes} bytes")
        return ref.media_type, data

    if not source:
        raise MediaSourceError(f"{ref.describe()} has no data")

    if source.startswith(DATA_URL_PREFIX):
        declared_type, data = _parse_data_url(source)
        if len(data) > max_bytes:
            raise MediaLimitError(f"media size {len(data)} bytes exceeds limit {max_bytes} bytes")
        return (ref.media_type or declared_type), data

    parsed = _URL_SCHEME_RE.match(source)
    if parsed:
        scheme = parsed.group("scheme").lower()
        if scheme in ("http", "https"):
            return ref.media_type, _fetch_url(source, http_timeout, max_bytes)
        if scheme == "file":
            if not allow_local_files:
                raise MediaSourceError("local file media is disabled, run the server with --allow_local_media_files")
            return ref.media_type, _read_file(unquote(urllib_path(source)), max_bytes)
        raise MediaSourceError(f"unsupported media URL scheme '{scheme}': {source}")
    if _is_bare_base64(source):
        data = decode_base64(source)
        if len(data) > max_bytes:
            raise MediaLimitError(f"media size {len(data)} bytes exceeds limit {max_bytes} bytes")
        return ref.media_type, data
    if not allow_local_files:
        raise MediaSourceError("local file media is disabled, run the server with --allow_local_media_files; "
                               f"got source '{source}'")
    return ref.media_type, _read_file(source, max_bytes)


def urllib_path(source: str) -> str:
    """Path component of a file:// URL."""
    return urlparse(source).path.lstrip("/")


def _is_bare_base64(source: str) -> bool:
    """Whether a source string is base64 payload without any URL or file marker."""
    if "/" in source or "\\" in source or "." in source:
        return False
    return looks_like_base64(source)
