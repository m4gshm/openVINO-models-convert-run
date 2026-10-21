# Tasks

## 1. e2e helpers module

- [ ] 1.1 Create `agent/e2e.py` with constants: model repo id `OpenVINO/Qwen3.5-2B-int4-ov`, models dir `models/`, expected phrases and FIM prompt parts (prefix/suffix/missing-middle line). Verify: `uv run python -c "from agent import e2e"` imports cleanly.
- [ ] 1.2 Implement `ensure_model()` — idempotent snapshot download into `models/Qwen3.5-2B-int4-ov` via `huggingface_hub.snapshot_download`, completeness check on key IR files, honoring `OPENVINO_E2E_SKIP_FETCH=1`; raises a skip-able exception when fetch fails. Verify: `uv run pytest agent/e2e_test.py -m e2e -k fetch` passes.
- [ ] 1.3 Implement server harness: free-port picker, subprocess launch of `.venv/Scripts/python agent.py --model Qwen3.5-2B-int4-ov --device <env device> --port <port>` with stdout/stderr to `logs/e2e-<port>.log`, readiness poll (readiness timeout from `OPENVINO_E2E_READY_TIMEOUT_S`, default 300 s) and guaranteed shutdown (`finally` + Windows taskkill of the process tree). Verify: starting and stopping the harness leaves no python server process (`Get-Process` check) and the log file exists.
- [ ] 1.4 Implement HTTP helpers: `httpx` client factory, `chat_completion(stream=False|True, ...)`, `text_completion_fim(...)` and minimal SSE frame parser (accumulates deltas, returns finish reason + `[DONE]` sentinel flag). Verify: unit tests of the SSE parser with canned frames in `agent/e2e_test.py` pass via `uv run pytest agent/e2e_test.py` (non-e2e marked).

## 2. e2e tests

- [ ] 2.1 Create `agent/e2e_test.py` with pytest module-level `pytestmark = pytest.mark.e2e`, session-scoped fixtures that run `ensure_model()` + server bootstrap and skip when the model cannot be fetched. Verify: on an offline machine `uv run pytest agent/e2e_test.py -m e2e` **skips** rather than errors.
- [ ] 2.2 Register the `e2e` marker in pytest configuration (`[tool.pytest.ini_options]`/`pytest.ini` `markers`) and confirm `task test` (no `-m e2e`) does not try to download or start the server. Verify: `task test` output contains no e2e sessions and passes unchanged.
- [ ] 2.3 Test `test_chat_non_streaming`: `stream=false`, `temperature=0`, bounded `max_tokens`; assert one choice, `role == assistant`, content contains `EXPECTED_CHAT_PHRASE`, `finish_reason in {stop, length}`, model name matches; failure message shows expected vs actual. Verify: `uv run pytest agent/e2e_test.py -m e2e -k chat_non_streaming` passes against the real model.
- [ ] 2.4 Test `test_chat_streaming`: `stream=true`; assert zero or more `data:` frames ending with `finish_reason` chunk then `data: [DONE]`, no frame after `[DONE]`, concatenated deltas contain `EXPECTED_CHAT_PHRASE`. Verify: `uv run pytest agent/e2e_test.py -m e2e -k streaming` passes.
- [ ] 2.5 Test `test_fim_completion`: prompt `<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}` to `/v1/completions`; assert the completion contains the missing middle line (whitespace-normalized) and no literal `<|fim_prefix|>`/`<|fim_suffix|>`/`<|fim_middle|>` tokens leak. Verify: `uv run pytest agent/e2e_test.py -m e2e -k fim` passes.
- [ ] 2.6 Test `test_fim_max_tokens_bound`: same FIM request with small `max_tokens`; assert `finish_reason == length` and the request completes within the readiness/global timeout (no hang). Verify: `uv run pytest agent/e2e_test.py -m e2e -k fim_max_tokens` passes.

## 3. Runner integration and cleanup

- [ ] 3.1 Ensure session teardown kills the server even when a test fails or the run is interrupted (subprocess started via fixture `yield`/`finally`); verify by interrupting a run with Ctrl+C or a deliberately failing assertion and checking no leftover server process and `logs/e2e-*.log` records the shutdown.
- [ ] 3.2 Add a `task e2e` recipe (`Taskfile.yml`) running `uv run pytest agent/e2e_test.py -m e2e -v` and document usage + env vars (`OPENVINO_E2E_DEVICE`, `OPENVINO_E2E_SKIP_FETCH`, `OPENVINO_E2E_READY_TIMEOUT_S`) in AGENTS.md Testing section. Verify: `task e2e` runs the whole suite in one command and passes.

## 4. Quality gates (pre-commit checklist)

- [ ] 4.1 `task lint` — zero pylint violations in new files (`agent/e2e.py`, `agent/e2e_test.py`).
- [ ] 4.2 `task typecheck` — zero mypy errors for new files (do not add them to the `ignore_errors` ratchet).
- [ ] 4.3 `task test` — all pre-existing unit tests still pass; e2e suite excluded by default.
- [ ] 4.4 Full `task e2e` run against the real downloaded model passes all tests end to end.
