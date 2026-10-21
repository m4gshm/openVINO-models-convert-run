# Design

## Context

The agent package already serves an OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) from `agent/server_openvino.py` in OpenVINO mode; entry points `agent-dev.bat`, `agent.bat` / `agent.py` wrap the CLI (`--model`, `--device`, `--port`). The FIM path exists in the parser (`agent/parser/qwen_base.py`, FIM_MIDDLE `<|fim_middle|>`, plus the Qwen3.5 FIM prefix/suffix special tokens) and is exercised only in unit tests with fakes. Model folders live under `./models` (gitignored), and HF CLI tools (`hf.exe`, `huggingface-cli.exe`) plus the `huggingface_hub` python package (v1.21.0) are already available. Existing tests are pytest modules named `<module>_test.py` co-located with the code.

## Goals / Non-Goals

**Goals:**
- One pytest session command that: ensures the model, starts the server, and asserts chat (stream & non-stream) and FIM answers against expected phrases.
- Opt-in via pytest marker (`e2e`) so `task test` stays fast and offline-safe; skips cleanly without the model.
- Diagnosable failures: log server output, compare expected vs actual text in assertion messages.

**Non-Goals:**
- No changes to production code (server, parsers, CLI).
- CI matrix, GPU/NPU device coverage beyond CPU/GPU-default local run.
- Multimodal, Ollama endpoints, interrupt/autocomplete mode — covered elsewhere or later.
- Exact token-level determinism: model output is compared by "contains expected phrase," not byte equality.

## Decisions

- **Model fetch**: use `huggingface_hub.snapshot_download` (`HF_HUB_ENABLE_HF_TRANSFER` off by default) with `repo_id="OpenVINO/Qwen3.5-2B-int4-ov"` and `local_dir="models/Qwen3.5-2B-int4-ov"`. Rationale: already installed, resumable, no new dependency. Alternative considered: shelling out to `hf download` — kept as fallback if `huggingface_hub` import fails in some env (guard with `pytest.importorskip`).
- **Pre-fetch completeness check**: treat the directory as complete when it contains the key IR files (`openvino_model.xml`/`openvino_model.bin`, tokenizer files). If absent or partial, re-download.
- **Server harness**: start the server as a **subprocess** (`sys.executable agent.py --model Qwen3.5-2B-int4-ov --port <free> --device CPU|GPU`) rather than in-process. Rationale: mirrors real deployment, keeps OpenVINO runtime out of the pytest process, stream redirection into a log file gives diagnosability. Alternative considered: in-process `init_sequential_engine` via threads — cheaper but entangles test process with OpenVINO init state.
- **Readiness**: a session fixture (module scope) polls `GET /health` / `GET /v1/models` until 200 with a generous timeout (configurable via env, default 300 s since NPU/GPU compile is slow); on timeout, fail with the captured server log tail.

  *Note on the user's mention of "requesting a template chat-prefill generation" (кеш chat-шаблона):* if the server exposes an explicit warm-up endpoint or readiness state, use it; otherwise readiness is the first successful `/v1/models` poll followed by sending the first chat request in a fixture so per-test timings measure only generation.
- **Port**: pick a random free port with `socket.bind(("127.0.0.1", 0))` to avoid clashing with a dev server on 8888.

- **Chat assertions**: request a deterministic factual answer (e.g. "What is 2+2?") with `temperature=0`, `max_tokens` bounded (e.g. 200) so `finish_reason` is `stop`.
  - `stream=false`: one JSON response payload; assert `choices[0].message.role == "assistant"`, content phrase, finish reason.
  - `stream=true`: parse `text/event-stream` with standard `httpx` manual SSE accumulation; assert delta concatenation contains the phrase, last non-`[DONE]` frame has `finish_reason` in `{stop, length}`, and the stream ends with `data: [DONE]`.
- **FIM prompt shape**: build the raw-completion prompt with Qwen3.5 special tokens:
  `<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}` and assert the completion contains the missing middle line and no literal FIM tokens. Bounded `max_tokens` keeps the test fast.
  The completion reuse of `<|fim_middle|>` is server/parser territory and is asserted only as "the missing middle line is rendered" (i.e. no raw `<|fim_middle|>` leaks).
- **Expected values table**: expected phrases and FIM prompt parts live as small module constants (`EXPECTED_*`) in the test file so output text and expectation stay side by side.
- **Cleanup**: session-scoped fixture with `yield`/`finally` terminates the subprocess and/or `taskkill /F /T /PID` on Windows to ensure no orphaned server.

## Risks / Trade-offs

- [Model download ~2 GB on first run] → model dir is gitignored and cached; skip tests rather than download when explicitly offline (`OPENVINO_E2E_SKIP_FETCH=1`).
- [Device-specific behavior: NPU compile slow] → the suite targets CPU or GPU by default (`--device` from env `OPENVINO_E2E_DEVICE`, default `GPU` like the local bat scripts), with a long readiness timeout; NPU left out of scope.
- [Model nondeterminism despite `temperature=0`] → assert "contains expected phrase" and a photo-safe stop reason rather than exact text; deterministic numeric prompts minimize variance.
- [SSE parsing drift with server chunk framing] → client parsing is standard (`data:` lines + `[DONE]`) and mirrored from the format documented in AGENTS.md (NDJSON for Ollama but SSE for OpenAI); covered by asserting the final `[DONE]` sentinel.
- [Subprocess port race] → bind-free port is passed as the CLI arg, server output monitored; readiness poll retries until set.
- [Server failures leak log noise into pytest output] → server stdout/stderr captured to `logs/e2e-<port>.log`; log tail printed only on failure or via `--verbose`.
