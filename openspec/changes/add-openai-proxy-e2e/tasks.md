# Tasks

## 1. OVMS server harness in e2e.py

- [x] 1.1 Add `OvmssHarness` class to `agent/e2e.py` that starts OVMS as a subprocess. Reads `OVMS_BINARY` (env, default: look for `ovms_server`/`ovms_server.exe` in PATH) and `OVMS_MODEL_DIR` (env, default: `models/Qwen3.5-2B-int4-ov`). Picks free ports for HTTP and gRPC. Verifies: `uv run python -c "from agent.e2e import OvmssHarness"` imports cleanly.
- [x] 1.2 Add `ovms_binary_path()` helper — returns path to OVMS binary or `None` if not found. Verify: returns `None` when OVMS is not in PATH, returns valid path when it is.
- [x] 1.3 Add `ovms_http_port()` and `ovms_grpc_port()` helpers — pick free localhost ports. Verify: ports are distinct and free.

## 2. Proxy server harness in e2e.py

- [x] 2.1 Add `ProxyServerHarness` class to `agent/e2e.py` that starts the agent in proxy mode (`--openai_base_url http://localhost:<ovms_http_port>`). Takes the OVMS harness as a dependency (reads its port). Verifies: `uv run python -c "from agent.e2e import ProxyServerHarness"` imports cleanly.
- [x] 2.2 Add `proxy_ready_timeout_s()` helper (default 10 s, vs 300 s for native). Verify: returns 10.0 by default, configurable via `OPENAI_PROXY_READY_TIMEOUT_S`.

## 3. Proxy-mode e2e test module

- [x] 3.1 Create `agent/e2e_proxy_test.py` with `pytestmark = pytest.mark.e2e_proxy`. Session-scoped fixtures that check for OVMS binary availability — skips with message if not found. Verifies: on a machine without OVMS, `uv run pytest agent/e2e_proxy_test.py -m e2e_proxy` **skips** (not fails).
- [x] 3.2 Test `test_proxy_chat_non_streaming`: `stream=false`, `temperature=0`, bounded `max_tokens`. Assert one choice, `role == "assistant"`, content contains "4", `finish_reason in {stop, length}`. Verifies: `uv run pytest agent/e2e_proxy_test.py -m e2e_proxy -k chat_non_streaming` passes against local OVMS.
- [x] 3.3 Test `test_proxy_chat_streaming`: `stream=true`. Assert SSE frames ending with `finish_reason` then `data: [DONE]`, concatenated deltas contain "4". Verifies: `uv run pytest agent/e2e_proxy_test.py -m e2e_proxy -k streaming` passes.
- [x] 3.4 Test `test_proxy_fim_completion`: FIM prompt with known missing middle line. Assert completion contains the middle line and no FIM control tokens leak. Verifies: `uv run pytest agent/e2e_proxy_test.py -m e2e_proxy -k fim` passes.
- [x] 3.5 Test `test_proxy_fim_max_tokens_bound`: Same FIM with small `max_tokens`. Assert `finish_reason == length`. Verifies: `uv run pytest agent/e2e_proxy_test.py -m e2e_proxy -k fim_max_tokens` passes.

## 4. Test infrastructure wiring

- [x] 4.1 Register `e2e_proxy` marker in `pyproject.toml` `[tool.pytest.ini_options].markers`. Verify: `task test` output contains no e2e-proxy sessions and passes unchanged.
- [x] 4.2 Add `task e2e-proxy` to `Taskfile.yml` running `uv run pytest agent/e2e_proxy_test.py -m e2e_proxy -v`. Verify: `task e2e-proxy` runs the suite in one command.
- [x] 4.3 Document the new suite in AGENTS.md (OVMS binary, task command, skip conditions). Verify: AGENTS.md Testing section mentions `e2e-proxy` and OVMS-related env vars.

## 5. Quality gates

- [x] 5.1 `task lint` — zero pylint violations in `agent/e2e.py` (new `OvmssHarness`, `ProxyServerHarness`) and `agent/e2e_proxy_test.py`.
- [x] 5.2 `task typecheck` — zero mypy errors for new files.
- [x] 5.3 Full `task e2e-proxy` run against a local OVMS instance passes all tests.

## 6. Bug fixes discovered during implementation

- [x] 6.1 `parse_sse_response` now handles chunks without `choices` (usage/heartbeat chunks) — skips them instead of crashing with IndexError. Verified: `task e2e` passes all 4 tests.

## 7. Automatic OVMS download

- [x] 7.1 `ovms_binary_path()` resolves `OVMS_BINARY` → PATH (`ovms`/`ovms.exe`/`ovms_server`/`ovms_server.exe`) → previously unpacked `bin/ovms-<version>/` → downloads `ovms_windows_<version>_python_off.zip` from the OpenVINO Model Server GitHub release and unpacks it. Windows-only; returns `None` elsewhere.
- [x] 7.2 `OVMS_VERSION` env var selects the release (default `2026.4.1`); `bin/` is already git-ignored.
- [x] 7.3 Unit test `test_ovms_binary_path_honours_configured_path` covers the env override without network.
- [x] 7.4 OVMS CLI corrected to the real 2026.4 flags: `--model_name`, `--rest_port` (REST) and `--port` (gRPC); readiness probed on `/v2/health/ready`.
- [x] 7.5 `ProxyServerHarness` passes `--openai_api_key` (OVMS runs without auth) and targets OVMS `/v3`; `OvmssHarness.start()` is idempotent.
- [x] 7.6 FIM tests skip with an explicit message when the remote is a VLM servable that rejects `/completions`; chat tests run for real. Verified: `task e2e-proxy` → 2 passed, 2 skipped.

## 8. Proxy FIM without skips (scope change: production code)

- [x] 8.1 Root cause: the proxy controller's `completions()` was a stub (`pass`), so `/v1/completions` always returned an empty body — independent of OVMS. Confirmed by calling OVMS directly: a text-only model serves `/v3/completions` fine.
- [x] 8.2 Implement `OpenAiController.completions()` in `agent/openai/engine_rest_openai.py`: forwards `prompt`/`max_tokens`/`temperature`/`stop` to the remote text-completions API (streamed) and re-shapes completion chunks into chat chunks for the shared responder. Verified: `task e2e-proxy` → 4 passed, 0 skipped.
- [x] 8.3 FIM tests use a second OVMS + proxy pair serving `models/Mellum-4b-base-int4-sym-g128-se-awq` (text-only, FIM-specialized, `<fim_prefix>` token style); they skip only when that model is not fetched locally. `text_completion_fim()` gained a `fim=(model, begin, end, middle)` parameter (default unchanged).
- [x] 8.4 Quality gates: `task e2e` 4 passed, `task e2e-proxy` 4 passed, plain `task test` 454 passed, pylint 10.00/10, mypy clean for touched files.