# Proposal

## Why

The agent project's test suite (`agent/**/*_test.py`) covers parsers, token handling, and engines at the unit level with fakes, but nothing verifies that a full stack — real OpenVINO model on disk, HTTP server up, OpenAI-compatible endpoints — actually answers end to end. A real-model smoke e2e run catches regressions (model loading, chat template, streaming, FIM/interrupt path) that unit tests cannot.

## What Changes

- Add a pytest-marked e2e test suite (opt-in, not run by plain `task test`) that:
  - Ensures the target model `OpenVINO/Qwen3.5-2B-int4-ov` is present under `models/` (fetch via HF snapshot download with `hf` CLI or `huggingface_hub.snapshot_download`, cached so the download happens only once).
  - Boots the agent server (OpenVINO mode) on a fixed free port as a subprocess or in-process, and waits until the HTTP endpoint reports it is serving (via the template chat-prefill generation).
  - Sends `POST /v1/chat/completions` with `stream=false` and with `stream=true` and verifies the externally observable response contract (roles, content resulting in visible text, SSE frame order in streaming, `finish_reason`, last chunk marked as final).
  - Sends a FIM request to `POST /v1/completions` using Qwen3.5 FIM special tokens (`<|fim_prefix|>` / `<|fim_suffix|>` / `<|fim_middle|>`) and asserts the completion contains the reconstructed middle part of the code.
  - Cleans up the server after the suite.

## Capabilities

### New Capabilities
- `e2e-tests`: end-to-end smoke verification of the agent server with the real Qwen3.5 model — model fetching, server startup, OpenAI chat completions (stream and non-stream), and FIM completions.

### Modified Capabilities
<!-- none: no production code behavior changes, this change only adds tests -->

## Impact

- New test module(s) under `agent/` (e.g. `agent/e2e_test.py` plus a small `agent/e2e.py` helper for model fetch and server bootstrap), co-located `_test.py` naming per project rules.
- Requires network access and disk space on first run (model ~2 GB); environment/jenkins-safe because the model dir is gitignored and cached.
- No production code changes expected; if e2e reveals needed server-side fixes, those are separate changes.
- Dependency: `huggingface_hub` — evaluate whether it is already installed transitively; if not, use the installed `hf`/`huggingface-cli` executables via `subprocess` to avoid a new runtime dependency.
