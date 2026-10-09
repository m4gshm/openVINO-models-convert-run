"""Decoded media inputs of a single generation call and the loader producing them.

OpenVINO GenAI pipelines receive media as ``ov.Tensor`` lists passed next to the prompt
(``images=``, ``videos=``, ``audios=``), while the prompt itself refers to them by index.
"""

import logging
from dataclasses import dataclass, field

import openvino as ov

from agent.multimodal.audio_decode import decode_audio
from agent.multimodal.content_parts import MediaRef
from agent.multimodal.image_decode import decode_image
from agent.multimodal.media_error import MediaDecodeError, MediaLimitError
from agent.multimodal.media_sources import format_of_media_type, resolve_bytes, sniff_format
from agent.multimodal.modality import Modality

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MediaLimits:
    """Limits and target formats applied while resolving and decoding media of one request."""

    max_bytes: int = 20 * 1024 * 1024
    max_items: int = 16
    allow_local_files: bool = False
    http_timeout: float = 30.0
    audio_sample_rate: int = 16000


@dataclass
class MediaInputs:
    """Decoded media tensors to pass into a pipeline together with the prompt."""

    images: list[ov.Tensor] = field(default_factory=list)
    videos: list[ov.Tensor] = field(default_factory=list)
    audios: list[ov.Tensor] = field(default_factory=list)

    @staticmethod
    def empty() -> "MediaInputs":
        """No media inputs, the plain text generation path."""
        return MediaInputs()

    def is_empty(self) -> bool:
        """Whether nothing but text was attached."""
        return not (self.images or self.videos or self.audios)

    def of(self, modality: Modality) -> list[ov.Tensor]:
        """Tensor list of the given modality."""
        if modality == Modality.IMAGE:
            return self.images
        if modality == Modality.AUDIO:
            return self.audios
        if modality == Modality.VIDEO:
            return self.videos
        return []

    def generate_kwargs(self) -> dict[str, list[ov.Tensor]]:
        """Pipeline generation keyword arguments for the non-empty media modalities."""
        kwargs: dict[str, list[ov.Tensor]] = {}
        if self.images:
            kwargs["images"] = self.images
        if self.videos:
            kwargs["videos"] = self.videos
        if self.audios:
            kwargs["audios"] = self.audios
        return kwargs

    def counts(self) -> dict[str, int]:
        """Number of items per modality, empty modalities omitted."""
        counts = {Modality.IMAGE.value: len(self.images), Modality.AUDIO.value: len(self.audios),
                  Modality.VIDEO.value: len(self.videos)}
        return {name: count for name, count in counts.items() if count}

    def summary(self) -> str:
        """Short description for logs and API responses."""
        counts = self.counts()
        return ", ".join(f"{name}={count}" for name, count in sorted(counts.items())) or "no media"


class MediaLoader:
    """Resolves and decodes media references, enforcing the request limits."""

    def __init__(self, limits: MediaLimits):
        self.limits = limits
        self._loaded = 0

    def _next_index(self, media: MediaInputs, modality: Modality) -> int:
        self._loaded += 1
        if self._loaded > self.limits.max_items:
            raise MediaLimitError(f"too many media items in one request: limit is "
                                  f"{self.limits.max_items}, modality '{modality.value}'")
        return len(media.of(modality))

    def load(self, ref: MediaRef, media: MediaInputs) -> int:
        """Decode one media reference into a tensor, return its index within its modality list.

        Raises:
            MediaDecodeError, MediaSourceError, MediaLimitError: see :mod:`agent.multimodal.media_error`.
        """
        index = self._next_index(media, ref.modality)
        _, raw = _resolve(ref, self.limits)
        detected = sniff_format(raw)
        media_format = ref.fmt or detected or format_of_media_type(ref.media_type)
        log.debug("decoding %s item #%d, %d bytes, format=%s", ref.modality.value, index, len(raw), media_format)
        if ref.modality == Modality.IMAGE:
            media.images.append(decode_image(raw, media_format))
        elif ref.modality == Modality.AUDIO:
            media.audios.append(decode_audio(raw, media_format, self.limits.audio_sample_rate))
        elif ref.modality == Modality.VIDEO:
            raise MediaDecodeError(f"video input is not supported yet (format '{media_format or 'unknown'}'), "
                                   "attach separate frames as images instead")
        else:
            raise MediaDecodeError(f"media modality '{ref.modality.value}' cannot be attached to a prompt")
        return index


def _resolve(ref: MediaRef, limits: MediaLimits):
    """Resolve a reference into (media type, bytes) using the request limits."""
    return resolve_bytes(ref, allow_local_files=limits.allow_local_files,
                         max_bytes=limits.max_bytes, http_timeout=limits.http_timeout)
