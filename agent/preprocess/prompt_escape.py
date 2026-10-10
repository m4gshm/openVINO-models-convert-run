"""Neutralization of control markup that a client may put into prompt text.

The rendered prompt is a single string, so text written by a client (a system prompt quoting
docs, a file returned by a tool, a pasted README) cannot be told apart from markup that the
server or the chat template put there on purpose:

* OpenVINO GenAI universal media tags ``<ov_genai_image_N>`` / ``<ov_genai_video_N>`` are
  regex-parsed from the whole prompt; a tag without a matching tensor fails generation with
  ``Missing image/video with index N``, a tag with a matching index moves the picture;
* model special tokens (``<|im_start|>``, ``<|image_pad|>``, ``<turn|>``, ...) are tokenized into
  their control ids, so client text could close a turn, fake a role or add media placeholders.

Escaping inserts a zero width space after the first character of such markup: the model still
reads the same text while neither the pipeline nor the tokenizer recognize it any more.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

log = logging.getLogger(__name__)

ZERO_WIDTH_SPACE = "\u200b"

# Only vision tags are parsed by OpenVINO GenAI, audio is listed so that the escaped prompt does
# not depend on the pipeline version.
MEDIA_TAG_PATTERN = r"<ov_genai_(?:image|video|audio)_\d+>"

# Special tokens are markup-like: '<|im_start|>', '<turn|>', '<bos>', '[INST]'.
_TOKEN_OPEN = "<["
_TOKEN_CLOSE = ">]"
_DECODE_BATCH = 4096


class SpecialTokenSource(Protocol):
    """Part of ``openvino_genai.Tokenizer`` used to discover special tokens."""

    def get_vocab(self) -> dict: ...

    def decode(self, tokens, skip_special_tokens: bool = True): ...


def _token_text(token: bytes | str) -> str | None:
    if isinstance(token, str):
        return token
    try:
        return token.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _is_markup(text: str | None) -> bool:
    return bool(text) and len(text) > 2 and text[0] in _TOKEN_OPEN and text[-1] in _TOKEN_CLOSE


def read_special_tokens(tokenizer: SpecialTokenSource) -> frozenset[str]:
    """Markup-like special tokens of a model tokenizer.

    ``openvino_genai.Tokenizer`` exposes no special-token list, so a vocabulary entry counts as
    special when it looks like markup and disappears on ``decode(skip_special_tokens=True)``
    while decoding back to itself otherwise. Plain added tokens that the model prints as text
    (Qwen ``<think>``, ``<tool_call>``) are deliberately kept out: they decode visibly.
    """
    candidates: list[tuple[str, int]] = []
    for token, token_id in tokenizer.get_vocab().items():
        text = _token_text(token)
        if _is_markup(text):
            candidates.append((text, int(token_id)))
    special: set[str] = set()
    for start in range(0, len(candidates), _DECODE_BATCH):
        batch = candidates[start:start + _DECODE_BATCH]
        ids = [[token_id] for _, token_id in batch]
        skipped = tokenizer.decode(ids, skip_special_tokens=True)
        kept = tokenizer.decode(ids, skip_special_tokens=False)
        for (text, _), without, with_special in zip(batch, skipped, kept):
            if not without and with_special == text:
                special.add(text)
    return frozenset(special)


@dataclass(frozen=True)
class PromptEscaper:
    """Escapes media tags and the given special tokens in client supplied text."""

    special_tokens: frozenset[str] = frozenset()
    _pattern: re.Pattern = field(init=False, repr=False, compare=False)
    _triggers: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        # Longest first, so '<|image_pad|>' wins over a hypothetical '<|image|>' prefix match.
        tokens = sorted(self.special_tokens, key=len, reverse=True)
        alternatives = [MEDIA_TAG_PATTERN] + [re.escape(token) for token in tokens]
        object.__setattr__(self, "_pattern", re.compile("|".join(alternatives)))
        object.__setattr__(self, "_triggers", frozenset("<") | {token[0] for token in tokens})

    @staticmethod
    def of_tokens(tokens: Iterable[str]) -> "PromptEscaper":
        """Escaper for the media tags and the given special tokens."""
        return PromptEscaper(frozenset(token for token in tokens if len(token) > 1))

    @staticmethod
    def of_tokenizer(tokenizer: SpecialTokenSource | None) -> "PromptEscaper":
        """Escaper for the media tags and the special tokens of a model tokenizer.

        Falls back to media tags only when the tokenizer cannot be inspected.
        """
        if tokenizer is None:
            return PromptEscaper()
        try:
            tokens = read_special_tokens(tokenizer)
        except Exception as e:  # pylint: disable=broad-exception-caught
            log.warning("cannot read special tokens of the tokenizer, escaping media tags only: %s", e)
            return PromptEscaper()
        log.info("prompt escaping: %d special tokens of the model", len(tokens))
        return PromptEscaper.of_tokens(tokens)

    def escape(self, text: str) -> str:
        """Text with every media tag and special token broken by a zero width space."""
        if not any(trigger in text for trigger in self._triggers):
            return text
        return self._pattern.sub(_break_markup, text)

    def escape_in(self, value: Any) -> Any:
        """Copy of a nested str/list/dict structure with every string escaped."""
        if isinstance(value, str):
            return self.escape(value)
        if isinstance(value, dict):
            return {key: self.escape_in(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.escape_in(item) for item in value]
        return value


def _break_markup(match: re.Match) -> str:
    text = match.group(0)
    return text[0] + ZERO_WIDTH_SPACE + text[1:]


MEDIA_TAGS_ONLY = PromptEscaper()

__all__ = ["PromptEscaper", "MEDIA_TAGS_ONLY", "ZERO_WIDTH_SPACE", "read_special_tokens"]
