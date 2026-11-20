# Tasks

## 1. Setup

- [x] 1.1 Add `huggingface_hub` to root `pyproject.toml` and `agent/pyproject.toml`; run `uv sync` and verify import works (`python -c "import huggingface_hub"`)

## 2. Publish script

- [x] 2.1 Create `publish_models.py` with the `MODELS` list containing the 9 Qwen2.5-Coder folders and verify the list matches existing directories under `models/`
- [x] 2.2 Reuse `models/gen_readmes.py` for card generation: fix its `<org>` placeholder to `m4gshm` and drop the misleading trailing file list from the "IR weights" line; import its logic into the publish script for folders lacking a README (it already skips existing ones); verify generated card for `Qwen2.5-Coder-0.5B-int8-sym` matches the style of the published cards
- [x] 2.3 Implement auth check (HF_TOKEN env or cached login) with a clear fail-fast message; verify the no-credentials error path
- [x] 2.4 Implement upload loop: `create_repo(exist_ok=True)` + `upload_folder` per model, continue on missing folder / upload error, non-zero exit if any model failed; verify by mocking `huggingface_hub` in tests

## 3. Tests

- [x] 3.1 Create `publish_models_test.py` covering: card generation (new README), existing README preserved, missing folder skipped, quantization suffix extraction; run `uv run python -m pytest publish_models_test.py -v`

## 4. Verification

- [x] 4.1 Run `task lint` and `task typecheck` — no new violations
- [x] 4.2 Dry-run one model against HF (e.g. `Qwen2.5-Coder-0.5B-int8-sym`), confirm repo appears at `m4gshm/` with README; re-run and confirm "up to date" (no new commits)
- [x] 4.3 All 9 models published to HF (verified: 9 repos, each 14 files with README.md); `models/` has 17 folders total, 9 published + 8 gemma-4/OmniCoder/etc. awaiting conversion
