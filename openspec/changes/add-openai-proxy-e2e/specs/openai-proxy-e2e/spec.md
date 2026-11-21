# Spec Delta

## Purpose

End-to-end smoke verification of the OpenAI proxy mode connecting to a local OpenVINO Model Server (OVMS) instance serving the same model (Qwen3.5-2B-int4-ov). Tests verify the full proxy → OVMS path: OVMS server bootstrap, proxy server bootstrap, chat completions (stream & non-stream), and FIM completions.

## ADDED Requirements

### Requirement: OVMS server starts as separate process

The e2e suite SHALL start OVMS (OpenVINO Model Server) as a separate subprocess serving `Qwen3.5-2B-int4-ov` on a free local port. OVMS is started with the model directory and configured to listen on HTTP/gRPC endpoints.

#### Scenario: OVMS starts on free port
- **WHEN** the e2e suite starts OVMS with the model directory
- **THEN** OVMS binds to a free localhost port and serves the model within the startup budget (default 120 s)

#### Scenario: OVMS startup failure skips the suite
- **WHEN** OVMS cannot start (model missing, port conflict, binary not found)
- **THEN** the e2e tests are skipped with an explanatory message rather than failing

### Requirement: Proxy server connects to local OVMS

The e2e suite SHALL start the agent in OpenAI proxy mode (`--openai_base_url http://localhost:<ovms_port>`) on a free local port. The proxy forwards requests to the local OVMS instance. No API key is needed for local OVMS.

#### Scenario: Proxy server starts and connects to OVMS
- **WHEN** the e2e suite starts the agent with `--openai_base_url` pointing to the local OVMS
- **THEN** the proxy binds to a free localhost port and forwards requests to OVMS

#### Scenario: Proxy server ready
- **WHEN** the proxy server starts
- **THEN** it becomes ready as soon as the HTTP listener binds (no model loading)

### Requirement: Non-streaming chat completions via proxy → OVMS

The proxy server SHALL answer `POST /v1/chat/completions` with `stream=false` by forwarding to the local OVMS instance. The response follows the OpenAI chat-completion contract.

#### Scenario: Single-shot chat question
- **WHEN** a client sends a chat completion with `stream=false` asking a simple question
- **THEN** the response contains one choice with `role == "assistant"`, non-empty content, and `finish_reason` in `{stop, length}`

#### Scenario: Response contains expected phrase
- **WHEN** the OVMS returns a valid answer
- **THEN** the test asserts the content contains the expected phrase (e.g., "4" for "What is 2+2?")

### Requirement: Streaming chat completions via proxy → OVMS

The proxy server SHALL answer `POST /v1/chat/completions` with `stream=true` by streaming chunks from OVMS. The client receives SSE frames ending with `data: [DONE]`.

#### Scenario: Streaming chat question
- **WHEN** a client sends a chat completion with `stream=true`
- **THEN** the client receives one or more `data:` SSE frames ending with `finish_reason` then `data: [DONE]`, and concatenated deltas contain the expected phrase

#### Scenario: Stream ends cleanly
- **WHEN** OVMS finishes generation
- **THEN** no frames arrive after `data: [DONE]` and the last non-[DONE] frame carries `finish_reason`

### Requirement: FIM completion via proxy → OVMS

The proxy server SHALL answer `POST /v1/completions` for a fill-in-the-middle prompt by forwarding to OVMS. The completion contains the missing middle part of the code.

#### Scenario: FIM completion returns middle line
- **WHEN** a client sends a FIM prompt with known missing middle line
- **THEN** the completion text contains the missing middle line and does not leak FIM control tokens

#### Scenario: FIM respects max_tokens
- **WHEN** a FIM request sets a small `max_tokens`
- **THEN** the response terminates with `finish_reason == length` and completes within the request timeout

### Requirement: Test isolation and cleanup

Each proxy e2e test session SHALL clean up after itself: terminate both the proxy and OVMS subprocesses and release their ports. Failures or interruptions must not leave orphaned processes.

#### Scenario: Cleanup on success
- **WHEN** all tests pass
- **THEN** both the proxy and OVMS processes are terminated and no orphaned processes remain

#### Scenario: Cleanup on failure
- **WHEN** a test fails or is interrupted
- **THEN** both the proxy and OVMS processes are still terminated before the session ends