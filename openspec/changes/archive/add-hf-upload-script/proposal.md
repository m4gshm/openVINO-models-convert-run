# Proposal

## Why

Models converted to OpenVINO IR in `models/` need to be published to the HuggingFace org https://huggingface.co/m4gshm. Uploading is currently manual; a repeatable script makes publishing idempotent (re-running syncs updates) and keeps README cards consistent across all published models, following the pattern of the 3 models already uploaded.

## What Changes

- Add a publish script (Python, `huggingface_hub` based) that uploads listed model folders from `models/` to `m4gshm/<model-name>` on HuggingFace.
- The script contains an explicit, user-extensible list of models to publish (seeded with the 9 Qwen2.5-Coder variants listed in the request).
- Re-running the script is safe: existing repos are updated/synced (files re-uploaded only when changed), not duplicated or deleted.
- Before each upload, generate/check a model card (`README.md` with YAML metadata: `license`, `base_model`, `tags`, `language`) from a shared template, prepared on the pattern of the already-published models.
- First upload of a repo creates it (public) with `.gitattributes` suitable for large files; model card is written into the local model folder and uploaded.

## Capabilities

### New Capabilities

- `hf-model-publishing`: Publishing OpenVINO IR model folders from `models/` to the `m4gshm` HuggingFace org: model list, upload/sync behavior, and model card generation.

### Modified Capabilities

(none)

## Impact

- New script (e.g. `publish_models.py` at repo root, consistent with existing root-level scripts) plus a README template.
- New optional dependency: `huggingface_hub` (listed in `agent/pyproject.toml` and root `pyproject.toml` if added there).
- Requires `HF_TOKEN` (or `huggingface-cli login`) with write access to the `m4gshm` org/user.
- No changes to the agent runtime code; `models/` folders gain a `README.md` after first run.
