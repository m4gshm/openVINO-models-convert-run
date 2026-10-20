"""Detection of the model architecture and of the matching parser and pipeline.

The exported model folder describes itself in ``config.json``: the architecture
names select the prompt parser, and the context length caps the prompt size.
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from agent.openvino.options import ParserType, Pipe

log = logging.getLogger(__name__)

CONFIG_JSON = "config.json"


@dataclass
class ModelConfig:
    """What the model folder says about itself."""

    architectures: set[str] = field(default_factory=set)
    max_position_embeddings: int | None = None


def read_model_config(model_path: Path) -> ModelConfig:
    """Read the architectures and the context length from ``config.json``.

    A malformed or missing ``config.json`` yields an empty configuration: the
    caller then falls back to the CLI switches.
    """
    model_config = ModelConfig()
    config_json = model_path / CONFIG_JSON
    if not config_json.is_file():
        return model_config
    try:
        config = json.loads(config_json.read_text(encoding="utf-8"))
    except Exception as e:  # pylint: disable=broad-exception-caught  # malformed config.json must not abort startup
        log.error("error on read %s: %s", config_json, e)
        return model_config
    architectures = config.get("architectures")
    if isinstance(architectures, list):
        model_config.architectures = set(architectures)
    model_config.max_position_embeddings = _max_position_embeddings(config)
    return model_config


def _max_position_embeddings(config: dict) -> int | None:
    """Context length of the model: nested for multimodal configs, flat for the old ones."""
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return text_config.get("max_position_embeddings")
    # old models like Qwen2.5
    return config.get("max_position_embeddings")


def parser_type_of(value: str | None) -> ParserType | None:
    """Convert a CLI parser name into a parser type, None for an unknown name."""
    if not value:
        return None
    try:
        return ParserType(value)
    except ValueError:
        log.error("unknown parser type '%s'", value)
        return None


def detect_parser_type(model_architectures: set[str]) -> ParserType | None:
    """Select the prompt parser family by the architecture names of the model."""
    architectures = [arch.lower() for arch in model_architectures]

    def has(fragment: str) -> bool:
        """Whether any architecture name contains the fragment."""
        return any(fragment in arch for arch in architectures)

    if has("gemma4"):
        return ParserType.gemma4
    if has("qwen3_5"):
        return ParserType.qwen3moe
    if has("qwen3moe"):
        return ParserType.qwen3moe
    if has("qwen3") or has("qwen2"):
        return ParserType.qwen2
    if has("lfm2"):
        return ParserType.lfm2
    return None


def default_pipe_of(parser_type: ParserType, model_architectures: set[str]) -> Pipe:
    """Pipeline a parser family is normally served with.

    A qwen3.5 export is multimodal and is served as a VLM; the other text-only
    families are served as plain LLM pipelines.
    """
    if parser_type == ParserType.gemma4:
        return Pipe.VLM
    if parser_type == ParserType.qwen3moe and _is_qwen3_5(model_architectures):
        return Pipe.VLM
    return Pipe.LLM


def _is_qwen3_5(model_architectures: set[str]) -> bool:
    """Whether the model is a qwen3.5 export."""
    return any("qwen3_5" in arch.lower() for arch in model_architectures)


def or_default_pipe(pipe: Pipe | None, default: Pipe) -> Pipe:
    """Return pipe if it is set, otherwise the default one."""
    return pipe if pipe else default
