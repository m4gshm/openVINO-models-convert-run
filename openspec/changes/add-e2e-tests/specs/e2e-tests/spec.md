# Spec Delta

## Purpose

End-to-end smoke verification of the OpenAI-compatible agent server with a real OpenVINO model: fetch the model, start the server, and confirm that chat completions (streaming and non-streaming) and FIM text completions return contract-correct answers.

## ADDED Requirements

### Requirement: Model fetch into models directory
The e2e suite SHALL ensure the reference model `OpenVINO/Qwen3.5-2B-int4-ov` from Hugging Face is present in the local `models/` directory before any e2e test runs. Fetching MUST be idempotent: a complete local copy is reused and no download is repeated. A fetch failure or absence of the model SHALL cause the e2e suite to be skipped with an explanatory message rather than fail unrelated tests.

#### Scenario: Model already present
- **WHEN** the e2e suite starts and `models/Qwen3.5-2B-int4-ov` already contains a complete OpenVINO model copy
- **THEN** no network fetch runs and the tests proceed

#### Scenario: Model missing
- **WHEN** the e2e suite starts and the model directory is absent or incomplete
- **THEN** the model is downloaded from `OpenVINO/Qwen3.5-2B-int4-ov` into `models/`, and the tests proceed once the download finishes

#### Scenario: Fetch impossible
- **WHEN** the model cannot be fetched (no network, HF unreachable)
- **THEN** the e2e tests are skipped, not failed, with a message naming the missing model

### Requirement: Server starts on the fetched model
The e2e suite SHALL start the agent server in OpenVINO mode serving the fetched model on a free local port and wait until the server accepts OpenAI-compatible requests before sending any test request. The default `Qwen3.5-2B-int4-ov` model batch script shall remain the reference for CLI arguments; the suite MUST NOT modify server startup behavior.

#### Scenario: Server ready
- **WHEN** the server subprocess is started with the fetched model directory and reaches the point where the template chat prefill has been generated (i.e. the endpoint reports it is serving)
- **THEN** subsequent e2e requests are sent to `http://127.0.0.1:<port>` endpoints

#### Scenario: Server fails to start
- **WHEN** the server exits or does not become ready within the allowed time budget
- **THEN** the e2e test fails with the captured server output included in the error

### Requirement: Non-streaming chat completions contract
The server serving the real model SHALL answer `POST /v1/chat/completions` with `stream=false` with a single JSON OpenAI chat-completion response having an assistant message with non-empty text, `finish_reason` `stop` (or `length` when `max_tokens` is exhausted), and a `model` value matching the served model.

#### Scenario: Single-shot chat question
- **WHEN** a client sends a chat completion with `stream=false` asking a question whose expected answer phrase is known
- **THEN** the response contains exactly one choice whose message role is `assistant`, whose content contains the expected phrase, and whose finish reason is `stop` or `length`

### Requirement: Streaming chat completions contract
The server SHALL answer `POST /v1/chat/completions` with `stream=true` by streaming Server-Sent Events: frames with `data: ` objects and terminal frames including a chunk with `finish_reason` set (`stop` or `length`) and a final `data: [DONE]` frame. Concatenated chunk deltas SHALL form content equal to or equally valid in meaning as the expected answer, and no chunk may arrive after `[DONE]`.

#### Scenario: Streaming chat question
- **WHEN** a client sends a chat completion with `stream=true`
- **THEN** the client receives one or more `data:` SSE frames ending with a `finish_reason` frame followed by `data: [DONE]`, and the concatenated deltas contain the expected phrase

### Requirement: FIM completion via v1/completions
The server SHALL answer `POST /v1/completions` for a fill-in-the-middle prompt built from Qwen3.5 FIM special tokens (`<|fim_prefix|>` + prefix code + `<|fim_suffix|>` + suffix code) with a completion that contains the missing middle part of the code, without echoing the FIM control tokens as literal separator text.

#### Scenario: Fill-in-the-middle known code snippet
- **WHEN** a client sends a completion request whose prompt contains a code prefix, a known missing single line (e.g. a function return or assignment line), and a code suffix
- **THEN** the completion text contains the missing middle line (modulo leading/trailing whitespace and formatting) and does not contain literal `<|fim_prefix|>`/`<|fim_suffix|>` markers

#### Scenario: FIM request bound by max_tokens
- **WHEN** a FIM completion request sets a small `max_tokens`
- **THEN** the response terminates with `finish_reason` `length` and does not hang

### Requirement: Result comparison against expectation
Each e2e request SHALL compare the received result against a defined expected value and assert on it; an assertion failure MUST include the actual assistant text / completion text and the expected phrase in the error report so the failure is diagnosable from the pytest output alone.

#### Scenario: Mismatch is reported
- **WHEN** the model's answer does not contain the expected phrase
- **THEN** the test fails with a message showing both the expected phrase and the full actual output

### Requirement: Preparation and cleanup are transparent
The e2e suite SHALL shut down the server it started after tests finish and include per-test timing as pytest output. Preparation steps (model fetch, server start, template chat prefill) MUST report progress with log/pytest messages so a human can see what stage an e2e run is in.

#### Scenario: Suite completes
- **WHEN** all e2e tests finish (pass, skip, or fail)
- **THEN** the server process is terminated, the temp resources are released, and none of the suite's processes remain

#### Scenario: Suite fails midway
- **WHEN** one test fails or the suite is interrupted
- **THEN** the server process is still shut down before the session ends
