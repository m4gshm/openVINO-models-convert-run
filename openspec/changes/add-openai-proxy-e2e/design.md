# Design

## Context

The existing `agent/e2e.py` provides `ServerHarness` that starts the agent in native OpenVINO mode (`--model <path> --device <dev>`). The model must be downloaded (~2 GB) and loaded into OpenVINO, taking up to 300 s on NPU/GPU.

OpenVINO Model Server (OVMS) is Intel's standalone model serving component. It can serve OpenVINO models via HTTP/gRPC endpoints. The OpenAI proxy mode (`agent/server_openai.py`) forwards requests to a remote API; for local testing we need it to connect to a local OVMS instance instead.

The e2e tests for proxy mode involve **two subprocesses**:
1. OVMS — serves the model on a local port
2. Agent in proxy mode — connects to OVMS and exposes OpenAI-compatible endpoints

## Goals / Non-Goals

**Goals:**
- Add `OvmssHarness` class to `agent/e2e.py` that starts OVMS as a separate process serving `Qwen3.5-2B-int4-ov`.
- Add `ProxyServerHarness` class to `agent/e2e.py` that starts the agent in proxy mode pointing to the local OVMS.
- Add a new test module `agent/e2e_proxy_test.py` with `@pytest.mark.e2e-proxy` marker that exercises chat (stream/non-stream) and FIM through the proxy → OVMS path.
- Share HTTP helpers (`chat_completion`, `text_completion_fim`, `parse_sse_response`) from `agent/e2e.py` — no duplication.
- Skip cleanly when OVMS binary is not available.

**Non-Goals:**
- No changes to production code (server, controllers, parsers).
- No coverage of different model formats — uses the same Qwen3.5-2B-int4-ov as native e2e.
- No multimodal testing for proxy mode — same as native mode.
- No authentication or credential management — local OVMS requires no API key.

## Decisions

- **Separate test module (`agent/e2e_proxy_test.py`)** rather than mixing proxy tests into `agent/e2e_test.py`. Rationale: the two suites have different fixtures, different skip conditions, and different subprocess management. Keeping them separate avoids conditional logic clutter.

- **`OvmssHarness` in `agent/e2e.py`** that starts OVMS as a subprocess. OVMS is started with `--model_path <models/Qwen3.5-2B-int4-ov> --port <free_port> --grpc_port <free_port>`. The harness picks free ports, waits for OVMS to become ready (poll `GET http://localhost:<port>/health`), and tracks the process for cleanup.

- **`ProxyServerHarness` in `agent/e2e.py`** that starts the agent in proxy mode (`--openai_base_url http://localhost:<ovms_port>`). The harness reads the OVMS port from the `OvmssHarness` instance. No API key needed for local OVMS.

- **Two-phase startup**: first start OVMS, wait for ready, then start proxy pointing to OVMS. The proxy readiness poll is fast (5 s) since it's just an HTTP forwarder.

- **Environment variables**: `OVMS_BINARY` (path to ovms binary, default: look for `ovms_server` or `ovms_server.exe` in PATH), `OVMS_MODEL_DIR` (path to model directory, default: `models/Qwen3.5-2B-int4-ov`). Both optional — if OVMS binary is not found, the suite skips.

- **Expected phrases**: use the same "What is 2+2?" question. The assertion is "contains the digit 4" (same as native mode). FIM prompts use the same control tokens.

- **Marker registration**: add `e2e-proxy` to `[tool.pytest.ini_options].markers` in `pyproject.toml`. The existing `e2e` marker is already registered. `task test` stays unaffected (both markers are opt-in).

- **Taskfile entry**: add `task e2e-proxy` running `uv run pytest agent/e2e_proxy_test.py -m e2e-proxy -v`.

## Risks / Trade-offs

- [OVMS binary availability] → tests skip when OVMS binary is not found. No OVMS = no test.
- [OVMS startup time] → OVMS loads the model, takes ~60-120 s. The harness uses a configurable timeout (default 120 s).
- [Two subprocess management] → the harness must track both OVMS and proxy processes, ensuring both are cleaned up on failure/interruption.
- [Port conflicts] → free ports are auto-assigned for both OVMS and proxy; the harness coordinates port allocation.
- [Different model behavior] → OVMS may return slightly different formatting than the native agent server. Expected phrases are soft ("contains 4"), not exact-match.
- [OVMS version compatibility] → the e2e tests assume a recent OVMS version that supports the model format. Version mismatch would cause startup failure.