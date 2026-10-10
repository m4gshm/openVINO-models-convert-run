"""Preparation of chat messages that carry images or audio for OpenVINO GenAI generation.

OpenVINO GenAI expands universal media tags placed in the prompt
(``<ov_genai_image_0>``, ``<ov_genai_audio_1>``, ...), so a multimodal OpenAI request is
turned into two artifacts:

* messages with ``content`` flattened to a plain string containing the tags,
* :class:`~agent.multimodal.media_inputs.MediaInputs` tensors in the very same order.

Tag indexes are conversation absolute: every request carries all media of the history again
(the OpenAI protocol is stateless), so the index equals the position in the modality list.
"""

import logging
from dataclasses import dataclass, field

from agent.multimodal.content_parts import as_dict, segments_of
from agent.multimodal.media_error import MediaError, MediaUnsupportedError
from agent.multimodal.media_inputs import MediaInputs, MediaLimits, MediaLoader
from agent.multimodal.modality import Modalities, Modality
from agent.preprocess.prompt_escape import MEDIA_TAGS_ONLY, PromptEscaper


log = logging.getLogger(__name__)

CONTENT_KEY = "content"

MEDIA_TAG_TEMPLATES = {
    Modality.IMAGE: "<ov_genai_image_{}>",
    Modality.VIDEO: "<ov_genai_video_{}>",
    Modality.AUDIO: "<ov_genai_audio_{}>",
}


@dataclass
class PreparedMessages:
    """Messages ready for a chat history plus the decoded media of the request."""

    messages: list[dict] = field(default_factory=list)
    media: MediaInputs = field(default_factory=MediaInputs.empty)

    @property
    def has_media(self) -> bool:
        """Whether the request carries any media item."""
        return not self.media.is_empty()

    def summary(self) -> str:
        """Short description for logs."""
        return self.media.summary()


def unsupported_message(modality: Modality, modalities: Modalities) -> str:
    """Client facing explanation of a rejected modality."""
    supported = ", ".join(modalities.names())
    return (f"model does not support '{modality.value}' input, supported modalities: {supported}. "
            f"Run the server on a model that carries the matching encoder with --pipe VLM.")


def prepare_messages(messages: list[dict], modalities: Modalities,
                     limits: MediaLimits | None = None,
                     escaper: PromptEscaper = MEDIA_TAGS_ONLY) -> PreparedMessages:
    """Flatten multimodal content parts into prompt text with media tags and decode the media.

    Args:
        messages: OpenAI chat messages as dicts (``model_dump()`` of request models).
        modalities: modalities accepted by the running model.
        limits: media size/count limits of a single request.
        escaper: neutralizes media tags and special tokens written by the client, applied to
            every string of a message before the server inserts tags of the real attachments.

    Returns:
        PreparedMessages with ChatHistory compatible dicts and decoded media tensors.

    Raises:
        MediaError: unsupported modality, malformed content part, source or decode failure.
    """
    media_limits = limits or MediaLimits()
    loader = MediaLoader(media_limits)
    media = MediaInputs()
    prepared: list[dict] = []
    for message in messages:
        message_dict = escaper.escape_in(as_dict(message))
        content = message_dict.get(CONTENT_KEY)
        if isinstance(content, list):
            message_dict[CONTENT_KEY] = _flatten_content(content, modalities, loader, media)
        prepared.append(message_dict)
    if media.counts():
        log.info("request media prepared: %s", media.summary())
    return PreparedMessages(messages=prepared, media=media)


def _flatten_content(parts: list, modalities: Modalities, loader: MediaLoader,
                     media: MediaInputs) -> str:
    """Render content parts as text, replacing media parts with universal OpenVINO tags."""
    segments = segments_of(parts)
    pieces: list[str] = []
    for segment in segments:
        if not segment.is_media:
            pieces.append(segment.text or "")
            continue
        ref = segment.media
        if not modalities.supports(ref.modality):
            raise MediaUnsupportedError(unsupported_message(ref.modality, modalities),
                                        modality=ref.modality.value)
        index = loader.load(ref, media)
        pieces.append(MEDIA_TAG_TEMPLATES[ref.modality].format(index))
    return "".join(pieces)


def media_summary_of(messages: list[dict]) -> str:
    """Count media parts of raw messages without decoding them."""
    counts: dict[str, int] = {}
    for message in messages:
        content = as_dict(message).get(CONTENT_KEY)
        if not isinstance(content, list):
            continue
        for segment in segments_of(content):
            if segment.is_media:
                name = segment.media.modality.value
                counts[name] = counts.get(name, 0) + 1
    return ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))


__all__ = ["MediaError", "PreparedMessages", "prepare_messages", "unsupported_message",
           "media_summary_of", "MEDIA_TAG_TEMPLATES"]
