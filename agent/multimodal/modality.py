"""Input modalities of a model: which media types a request may carry.

Capabilities come from the model itself (see :mod:`agent.multimodal.model_capabilities`); the
pipeline kind only bounds what can be delivered, and architecture names are a fallback.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger(__name__)

PIPE_VLM = "VLM"
PIPE_CB = "CB"
PIPE_LLM = "LLM"


class Modality(str, Enum):
    """Content modality of a chat message part."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


# Architecture name hints of models with a vision encoder.
VISION_ARCH_HINTS = (
    "vision",
    "vl",
    "llava",
    "minicpm",
    "internvl",
    "molmo",
    "pixtral",
    "gemma4",
    "gemma_4",
    "phi3vision",
    "phi3_vision",
    "phi4multimodal",
    "phi4_multimodal",
    "conditionalgeneration",
)

# Architecture name hints of models accepting audio inputs.
AUDIO_ARCH_HINTS = (
    "omni",
    "audio",
    "speech",
    "whisper",
    "minicpmo",
    "minicpm_o",
    "phi4multimodal",
    "phi4_multimodal",
)


@dataclass(frozen=True)
class Modalities:
    """Set of modalities an engine accepts for chat requests."""

    values: frozenset[Modality] = field(default_factory=lambda: frozenset({Modality.TEXT}))

    @staticmethod
    def of(*modalities: Modality | str) -> "Modalities":
        """Build Modalities from Modality members or their names."""
        values = {m if isinstance(m, Modality) else Modality(str(m).lower()) for m in modalities}
        return Modalities(values=frozenset(values | {Modality.TEXT}))

    def supports(self, modality: Modality) -> bool:
        """Whether the modality is accepted."""
        return modality in self.values

    def names(self) -> list[str]:
        """Modality names sorted for a stable API response."""
        return sorted(m.value for m in self.values)

    def media_names(self) -> list[str]:
        """Names of non-text modalities only."""
        return [name for name in self.names() if name != Modality.TEXT.value]


def _matches(architectures: set[str], hints: tuple[str, ...]) -> bool:
    lowered = [arch.lower() for arch in architectures]
    return any(hint in arch for arch in lowered for hint in hints)


def detect_modalities(model_architectures: set[str],
                      pipe: str | None = None) -> Modalities:
    """Fallback modality detection from architecture names, used when the model folder exposes
    no media encoders (GGUF file, HuggingFace checkout). Prefer
    :func:`agent.multimodal.model_capabilities.resolve_modalities`, which reads the model folder.

    Args:
        model_architectures: architectures read from the model ``config.json``.
        pipe: pipeline kind name, one of ``VLM``, ``CB``, ``LLM``.

    Returns:
        Modalities accepted by the engine.
    """
    detected = {Modality.TEXT}
    pipe_name = (pipe or "").upper()
    is_vlm = pipe_name == PIPE_VLM or pipe_name == PIPE_CB
    if is_vlm or _matches(model_architectures, VISION_ARCH_HINTS):
        detected.add(Modality.IMAGE)
    # Continuous batching pipeline has no audio inputs in the OpenVINO GenAI API.
    if pipe_name != PIPE_CB and _matches(model_architectures, AUDIO_ARCH_HINTS):
        detected.add(Modality.AUDIO)
    return Modalities(frozenset(detected))
