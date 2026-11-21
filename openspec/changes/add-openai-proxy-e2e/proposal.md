# Proposal

## Why

The existing e2e test suite (`add-e2e-tests`) covers only native OpenVINO mode — it downloads a model, starts a local server, and verifies chat/FIM responses. The OpenAI proxy mode (`server_openai.py`) is designed to forward requests to a remote OpenAI-compatible API, but for local development and testing we need it to connect to a **local OpenVINO Model Server (OVMS)** instance serving the same model. This gives us end-to-end verification of the proxy → OVMS path without requiring external API access.

## What Changes

- Extend `agent/e2e.py` with an **OVMS harness** that starts OVMS as a separate process serving `Qwen3.5-2B-int4-ov`, and a **proxy harness** that starts the agent in proxy mode pointing to the local OVMS.
- Add **proxy-mode e2e tests** in `agent/e2e_proxy_test.py` (new marker `e2e-proxy`) that exercise the same chat completion and FIM flows through the proxy → OVMS path.
- Register the `e2e-proxy` marker in pytest config so `task test` stays unaffected; `task e2e-proxy` runs the proxy suite.
- No changes to production code — only test infrastructure and helper functions.

## Capabilities

### New Capabilities
- `openai-proxy-e2e`: end-to-end smoke verification of the OpenAI proxy mode connecting to a local OVMS instance — OVMS server bootstrap, proxy server bootstrap, chat completions (stream & non-stream), and FIM completions through the proxy → OVMS path.

## Impact

- New test module `agent/e2e_proxy_test.py` (co-located with `agent/e2e_test.py`).
- New helper functions in `agent/e2e.py` for OVMS and proxy server harnesses.
- Requires `OVMS_PORT` and `PROXY_PORT` environment variables (or auto-assigned free ports).
- No production code changes.
- Extends the existing `add-e2e-tests` change's approach; shares `agent/e2e.py` helpers.