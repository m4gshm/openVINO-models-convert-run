"""Normalization of chat message content into text and media segments.

OpenAI compatible clients send ``content`` either as a plain string or as a list of parts.
Part shapes differ between vendors, so this module accepts the common ones and reduces them
to :class:`Segment` items: text to keep in the prompt and media references to decode.
"""

import logging
from dataclasses import dataclass
from typing import Any

from agent.multimodal.media_error import MediaPartError
from agent.multimodal.modality import Modality

log = logging.getLogger(__name__)

TYPE_KEY = "type"
TEXT = "text"
REFUSAL = "refusal"
IMAGE = "image"
IMAGE_URL = "image_url"
AUDIO = "audio"
AUDIO_URL = "audio_url"
INPUT_AUDIO = "input_audio"
VIDEO = "video"
VIDEO_URL = "video_url"


@dataclass(frozen=True)
class MediaRef:
    """Reference to a media item attached to a message."""

    modality: Modality
    source: str | None = None
    data: str | None = None
    media_type: str | None = None
    fmt: str | None = None

    def describe(self) -> str:
        """Short human readable description without media payload."""
        location = "inline base64" if self.data else (self.source or "unknown source")
        return f"{self.modality.value}({location}, format={self.fmt or self.media_type or 'auto'})"


@dataclass(frozen=True)
class Segment:
    """A piece of message content: either text or a media reference."""

    text: str | None = None
    media: MediaRef | None = None

    @staticmethod
    def of_text(text: str) -> "Segment":
        """Text segment."""
        return Segment(text=text)

    @staticmethod
    def of_media(media: MediaRef) -> "Segment":
        """Media segment."""
        return Segment(media=media)

    @property
    def is_media(self) -> bool:
        """Whether the segment carries media."""
        return self.media is not None


def as_dict(value: Any) -> dict:
    """Convert pydantic models and mappings into a plain dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    return {}


def _field_str(part: dict, *names: str) -> str | None:
    for name in names:
        value = part.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def _nested_field(part: dict, key: str, *names: str) -> str | None:
    nested = as_dict(part.get(key))
    return _field_str(nested, *names) if nested else None


def _media_ref(modality: Modality, source: str | None, data: str | None,
               media_type: str | None = None, fmt: str | None = None) -> MediaRef:
    if not source and not data:
        raise MediaPartError(f"{modality.value} content part has no source: "
                             f"expected base64 data URL, URL, file path or inline base64 data")
    return MediaRef(modality=modality, source=source, data=data, media_type=media_type, fmt=fmt)


def part_media_ref(part: dict) -> MediaRef:
    """Build a MediaRef out of a single content part dict."""
    part_type = str(part.get(TYPE_KEY) or "").lower()
    if part_type in (IMAGE_URL, IMAGE):
        return _media_ref(Modality.IMAGE,
                          source=_field_str(part, "image", "url") or _nested_field(part, "image_url", "url"),
                          data=_nested_field(part, IMAGE, "data"),
                          media_type=_nested_field(part, "image_url", "media_type"))
    if part_type in (INPUT_AUDIO, AUDIO, AUDIO_URL):
        audio = as_dict(part.get(INPUT_AUDIO))
        return _media_ref(Modality.AUDIO,
                          source=_field_str(part, "audio", "url") or _nested_field(part, AUDIO_URL, "url"),
                          data=_field_str(audio, "data") or _nested_field(part, INPUT_AUDIO, "data"),
                          media_type=_field_str(audio, "media_type") or _nested_field(part, INPUT_AUDIO,
                                                                                     "media_type"),
                          fmt=_field_str(audio, "format", "codec") or _nested_field(part, INPUT_AUDIO, "format"))
    if part_type in (VIDEO, VIDEO_URL):
        return _media_ref(Modality.VIDEO,
                          source=_field_str(part, "video", "url") or _nested_field(part, VIDEO_URL, "url"),
                          data=_nested_field(part, VIDEO, "data"),
                          media_type=_nested_field(part, VIDEO_URL, "media_type"))
    raise MediaPartError(f"unsupported content part type '{part_type}', "
                         f"supported: {TEXT}, {IMAGE_URL}, {INPUT_AUDIO}, {VIDEO_URL}")


def segments_of(content: Any) -> list[Segment]:
    """Split message content (string or content parts list) into text and media segments."""
    if content is None:
        return []
    if isinstance(content, str):
        return [Segment.of_text(content)] if content else []
    if isinstance(content, (list, tuple)):
        segments: list[Segment] = []
        for raw_part in content:
            part = as_dict(raw_part)
            part_type = str(part.get(TYPE_KEY) or "").lower()
            if part_type in (TEXT, ""):
                text = _field_str(part, TEXT)
                if text:
                    segments.append(Segment.of_text(text))
            elif part_type == REFUSAL:
                text = _field_str(part, REFUSAL)
                if text:
                    segments.append(Segment.of_text(text))
            elif part_type == "file":
                raise MediaPartError("content part 'file' is not supported, send it as 'image_url' or "
                                     "'input_audio' with base64 data")
            else:
                segments.append(Segment.of_media(part_media_ref(part)))
        return segments
    raise MediaPartError(f"unexpected message content type {type(content)}")


def content_text(content: Any) -> str:
    """Text of a message content, ignoring media parts. Safe for both str and parts list."""
    return "".join(segment.text for segment in segments_of(content) if segment.text)


def has_media(content: Any) -> bool:
    """Whether the message content carries at least one media part."""
    return any(segment.is_media for segment in segments_of(content))
