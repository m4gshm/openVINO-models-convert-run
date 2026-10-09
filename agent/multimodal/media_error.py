"""Errors raised while handling multimodal inputs of a chat request.

All of them are subtypes of :class:`MediaError`, which carries a message that can be
returned to the client as-is: it explains what exactly is unsupported and how to fix it.
"""


class MediaError(Exception):
    """Base error for multimodal input problems reported to the client."""

    def __init__(self, message: str, *, modality: str | None = None):
        super().__init__(message)
        self.message = message
        self.modality = modality


class MediaUnsupportedError(MediaError):
    """Requested modality is not supported by the model or by the pipeline in use."""


class MediaPartError(MediaError):
    """Message content part is malformed or of an unknown type."""


class MediaSourceError(MediaError):
    """Media reference cannot be resolved into bytes (bad URL, missing file, too large)."""


class MediaDecodeError(MediaError):
    """Media bytes cannot be decoded into a tensor with the available decoders."""


class MediaLimitError(MediaError):
    """Media count or size exceeds the configured request limits."""
