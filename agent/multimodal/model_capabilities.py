"""Input modality capabilities read from the model folder itself.

The exported OpenVINO GenAI model directory declares its media encoders by file presence, which is
what the pipeline actually loads:

* ``openvino_vision_embeddings_model.xml`` (plus ``..._merger``, ``..._pos``, ``openvino_resampler``)
  -> the model consumes images and video frames;
* ``openvino_audio_embeddings_model.xml`` -> the model consumes audio.

File presence is more accurate than the architecture name: the same architecture exported twice can
have (gemma-4-12b) or lack (gemma-4-26b-a4b) an audio encoder. When the folder has no OpenVINO IR at
all (GGUF file, HF-only checkout) the detector falls back to ``config.json`` and
``processor_config.json`` / ``preprocessor_config.json`` keys.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from agent.multimodal.modality import Modality, Modalities, PIPE_CB, PIPE_LLM, detect_modalities

log = logging.getLogger(__name__)

IR_SUFFIXES = (".xml", ".bin")
VISION_FILE_PREFIXES = ("openvino_vision_embeddings", "openvino_vision_encoder", "openvino_resampler")
AUDIO_FILE_PREFIXES = ("openvino_audio_embeddings", "openvino_audio_encoder")

IMAGE_CONFIG_KEYS = ("vision_config", "image_token_id", "image_token")
AUDIO_CONFIG_KEYS = ("audio_config", "audio_token_id", "audio_token")
VIDEO_CONFIG_KEYS = ("video_config", "video_token_id", "video_token")

IMAGE_PROCESSOR_KEYS = ("image_processor", "image_processor_type")
VIDEO_PROCESSOR_KEYS = ("video_processor", "video_processor_type")
AUDIO_PROCESSOR_KEYS = ("feature_extractor", "audio_processor", "sampling_rate")

CONFIG_JSON = "config.json"
PROCESSOR_CONFIGS = ("processor_config.json", "preprocessor_config.json")

DEFAULT_AUDIO_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class MediaCapabilities:
    """Media modalities the model folder declares, and the audio sample rate it expects."""

    image: bool = False
    audio: bool = False
    video: bool = False
    audio_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE

    @staticmethod
    def text_only() -> "MediaCapabilities":
        """No media encoders found."""
        return MediaCapabilities()

    @property
    def has_media(self) -> bool:
        """Whether any media encoder was detected."""
        return self.image or self.audio or self.video

    def describe(self) -> str:
        """Human readable detection result for logs."""
        found = [name for name, present in (("image", self.image), ("audio", self.audio),
                                            ("video", self.video)) if present]
        if not found:
            return "text only"
        return f"{', '.join(found)}, audio_sample_rate={self.audio_sample_rate}"

    def to_modalities(self, pipe: str | None = None) -> Modalities:
        """Intersect model capabilities with what the selected pipeline can carry.

        A text pipeline (``LLM``) has no media inputs at all, and continuous batching (``CB``)
        accepts images and video frames but no audio. Video is never advertised: decoding a video
        container needs an external demuxer, so clients send frames as images.
        """
        pipe_name = (pipe or "").upper()
        if pipe_name == PIPE_LLM:
            return Modalities.of(Modality.TEXT)
        pipeline_media = {Modality.IMAGE, Modality.VIDEO} if pipe_name == PIPE_CB else {
            Modality.IMAGE, Modality.AUDIO, Modality.VIDEO}
        accepted = {Modality.TEXT}
        if self.image and Modality.IMAGE in pipeline_media:
            accepted.add(Modality.IMAGE)
        if self.audio and Modality.AUDIO in pipeline_media:
            accepted.add(Modality.AUDIO)
        return Modalities(frozenset(accepted))


def _load_json(path: Path) -> dict:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        log.warning("cannot read %s: %s", path, e)
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _has_ir_file(model_dir: Path, prefixes: tuple[str, ...]) -> bool:
    """Whether the folder contains an OpenVINO IR file with one of the given name prefixes."""
    for entry in model_dir.iterdir():
        name = entry.name.lower()
        if name.endswith(IR_SUFFIXES) and name.startswith(prefixes):
            return True
    return False


def _has_keys(config: dict, keys: tuple[str, ...], depth: int = 3) -> bool:
    """Search config keys, descending into nested sub-configs (thinker_config, audio_config, ...)."""
    if depth < 0:
        return False
    for key, value in config.items():
        if key in keys and value not in (None, {}, [], False):
            return True
        if isinstance(value, dict) and _has_keys(value, keys, depth - 1):
            return True
    return False


def _audio_sample_rate(processors: list[dict]) -> int:
    """Sample rate expected by the model audio front-end."""
    for config in processors:
        for value in config.values():
            if isinstance(value, dict):
                rate = value.get("sampling_rate")
                if isinstance(rate, int) and rate > 0:
                    return rate
        rate = config.get("sampling_rate")
        if isinstance(rate, int) and rate > 0:
            return rate
    return DEFAULT_AUDIO_SAMPLE_RATE


def read_media_capabilities(model_path: Path | str) -> MediaCapabilities:
    """Detect which media the model can take, from the model folder contents.

    Args:
        model_path: exported model directory. A single model file (GGUF) carries no media
            descriptors, so it resolves to text only.

    Returns:
        MediaCapabilities of the model; text only when nothing was recognized.
    """
    path = Path(model_path)
    if not path.is_dir():
        return MediaCapabilities.text_only()
    model_dir = path

    image = _has_ir_file(model_dir, VISION_FILE_PREFIXES)
    audio = _has_ir_file(model_dir, AUDIO_FILE_PREFIXES)
    video = False
    if image or audio:
        # Media encoders are known from the IR files: only refine video from the configs.
        video = _has_keys(_load_json(model_dir / CONFIG_JSON), VIDEO_CONFIG_KEYS)
        processors = [_load_json(model_dir / name) for name in PROCESSOR_CONFIGS]
        return MediaCapabilities(image=image, audio=audio, video=video,
                                 audio_sample_rate=_audio_sample_rate(processors))

    config = _load_json(model_dir / CONFIG_JSON)
    processors = [_load_json(model_dir / name) for name in PROCESSOR_CONFIGS]
    processor_image = any(_has_keys(item, IMAGE_PROCESSOR_KEYS, depth=1) for item in processors)
    processor_audio = any(_has_keys(item, AUDIO_PROCESSOR_KEYS, depth=1) for item in processors)
    processor_video = any(_has_keys(item, VIDEO_PROCESSOR_KEYS, depth=1) for item in processors)
    image = processor_image or _has_keys(config, IMAGE_CONFIG_KEYS)
    audio = processor_audio or _has_keys(config, AUDIO_CONFIG_KEYS)
    video = processor_video or _has_keys(config, VIDEO_CONFIG_KEYS)
    if not (image or audio or video):
        return MediaCapabilities.text_only()
    return MediaCapabilities(image=image, audio=audio, video=video,
                             audio_sample_rate=_audio_sample_rate(processors))


def resolve_modalities(capabilities: MediaCapabilities,
                       model_architectures: set[str],
                       pipe: str | None = None) -> Modalities:
    """Accepted modalities of an engine: model capabilities first, name heuristics as fallback.

    Args:
        capabilities: what was read from the model folder by :func:`read_media_capabilities`.
        model_architectures: architectures from ``config.json``, used when the folder is silent.
        pipe: pipeline kind name (``VLM``, ``CB``, ``LLM``), limits what can be delivered at all.

    Returns:
        Modalities to accept in chat requests.
    """
    if capabilities.has_media:
        return capabilities.to_modalities(pipe)
    return detect_modalities(model_architectures, pipe)
