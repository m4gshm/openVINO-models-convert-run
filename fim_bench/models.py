"""Registry of FIM token styles and the models under test.

Different model families were pretrained with different fill-in-the-middle
control tokens. A wrong style yields empty or degenerate completions, so every
model entry must declare the style it was trained with.

To add a model: append an entry to ``MODELS`` (or rely on the family guess of
:func:`model_info`) and make sure its directory sits in ``models/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FimStyle:
    """The FIM control tokens of a model family."""

    prefix: str
    suffix: str
    middle: str
    stop: tuple[str, ...] = field(default=())


QWEN_STYLE = FimStyle(
    prefix="<|fim_prefix|>",
    suffix="<|fim_suffix|>",
    middle="<|fim_middle|>",
    stop=("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|endoftext|>", "<|file_sep|>"),
)

LFM_STYLE = FimStyle(
    prefix="<|fim_pre|>",
    suffix="<|fim_suf|>",
    middle="<|fim_mid|>",
    stop=("<|fim_pre|>", "<|fim_suf|>", "<|fim_mid|>", "<|endoftext|>"),
)

MELLUM_STYLE = FimStyle(
    prefix="<fim_prefix>",
    suffix="<fim_suffix>",
    middle="<fim_middle>",
    stop=("<|endoftext|>", "<fim_prefix>"),
)


@dataclass(frozen=True)
class ModelInfo:
    """A model under benchmark with its FIM style."""

    name: str
    style: FimStyle
    note: str = ""


MODELS: dict[str, ModelInfo] = {
    "Qwen2.5-Coder-3B-nf4-sym-g-1-se-awq": ModelInfo(
        "Qwen2.5-Coder-3B-nf4-sym-g-1-se-awq", QWEN_STYLE, "reference FIM coder, qwen2 arch"),
    "Qwen3.5-2B-int4-sym": ModelInfo(
        "Qwen3.5-2B-int4-sym", QWEN_STYLE, "general model, FIM tokens inherited from the qwen tokenizer"),
    "LFM-2.5-Coder-2.6B-nf4-sym-g-1-se-awq": ModelInfo(
        "LFM-2.5-Coder-2.6B-nf4-sym-g-1-se-awq", LFM_STYLE, "hybrid conv+attention, needs NPUW STATIC prefill"),
    "Mellum-4b-dpo-all-nf4-sym-g-1-se-awq": ModelInfo(
        "Mellum-4b-dpo-all-nf4-sym-g-1-se-awq", MELLUM_STYLE, "JetBrains completion model, DPO variant"),
    "Mellum-4b-base-nf4-sym-g-1-se-awq": ModelInfo(
        "Mellum-4b-base-nf4-sym-g-1-se-awq", MELLUM_STYLE, "JetBrains completion model, pretrained base (raw FIM)"),
}


def model_info(name: str) -> ModelInfo:
    """The info of a known model, or a best-effort guess by the family name."""
    known = MODELS.get(name)
    if known:
        return known
    lowered = name.lower()
    if "lfm" in lowered:
        return ModelInfo(name, LFM_STYLE, "guessed by family name")
    if "mellum" in lowered:
        return ModelInfo(name, MELLUM_STYLE, "guessed by family name")
    return ModelInfo(name, QWEN_STYLE, "guessed by family name")


def build_prompt(info: ModelInfo, prefix: str, suffix: str) -> str:
    """Wrap a prefix/suffix pair into the FIM prompt of the model style."""
    return f"{info.style.prefix}{prefix}{info.style.suffix}{suffix}{info.style.middle}"
