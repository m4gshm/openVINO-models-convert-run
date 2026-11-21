# Design

## Context

The agent already supports Veai IDE integration through `agent/client/veai/`. The Veai detection checks for "You are Veai Agent" in the first message, extracts user context (OS, workdir), and applies tool-call argument fixing. OpenCode IDE uses a different system prompt format ("You are an AI agent running in OpenCode") with different tool schemas (edit, glob, grep, read, shell, skill, subagent, webfetch, websearch, write, execute).

The loop detection mechanism already exists in `engine_rest_common.py` via `validate_messages()` and `PreprocessToolCall.check_loop_tool_calls()`. It needs to be extended to support user choices (continue/interrupt) for cycled tool calls.

## Goals / Non-Goals

**Goals:**
- Mirror the Veai module structure for OpenCode agent support
- Detect OpenCode agent sessions by system prompt marker
- Extract user context from OpenCode system prompts
- Apply OpenCode-specific tool-call argument fixing
- Wire detection into `engine_rest_common.py` alongside Veai
- Extend loop detection to offer user choices
- Add thinking loop detection to catch repeated reasoning without progress

**Non-Goals:**
- Modifying the existing Veai implementation
- Adding new IDE integrations beyond OpenCode
- Changing the OpenVINO GenAI inference pipeline

## Decisions

### Decision 1: Mirror Veai module structure

**Choice:** Create `agent/client/opencode/` mirroring `agent/client/veai/` structure.

**Rationale:** The Veai module provides a proven pattern for agent detection, context extraction, and tool-call fixing. Mirroring it reduces cognitive load and maintenance burden.

**Alternatives considered:**
- Inline all OpenCode logic in `engine_rest_common.py` — would duplicate Veai's pattern and increase coupling.
- Single unified agent module — would blur the distinction between IDE-specific tool schemas.

### Decision 2: Detection via system prompt marker

**Choice:** Check the first message content for "You are an AI agent running in OpenCode" string.

**Rationale:** This is the same approach Veai uses ("You are Veai Agent"). It's simple, reliable, and doesn't require API changes.

**Alternatives considered:**
- HTTP header detection — requires client changes.
- Separate endpoint — breaks the unified API contract.

### Decision 3: Tool-call fixing per OpenCode schema

**Choice:** Create per-tool fixers in `agent/client/opencode/tool_call_fixer.py` that handle OpenCode's tool schema (edit, glob, grep, read, shell, skill, subagent, webfetch, websearch, write, execute).

**Rationale:** OpenCode tools have different field names and requirements than Veai tools. Each tool needs its own fixer to normalize field names and validate required parameters.

**Alternatives considered:**
- Generic fixer — would miss tool-specific validation needs.
- No fixing — would cause silent failures on malformed tool calls.

### Decision 4: Wire into engine_rest_common.py

**Choice:** Add OpenCode detection alongside existing Veai detection in `engine_rest_common.py`, using the same pattern (is_opencode_agent → get_opencode_context → tool_fixer).

**Rationale:** Minimal changes to existing code. The Veai pattern is already wired and tested; extending it is safer than creating a new pipeline.

**Alternatives considered:**
- New engine class — adds complexity for a detection difference.
- Plugin system — over-engineered for two IDE integrations.

### Decision 5: Extend loop detection with user choices

**Choice:** When a cycled tool call is detected, offer the user "continue" or "interrupt" options via a tool call question.

**Rationale:** The existing `validate_messages()` already detects loops. Extending it to offer choices is a natural progression that improves UX without new infrastructure.

**Alternatives considered:**
- Auto-interrupt — too aggressive, may lose user work.
- Configurable threshold — adds complexity for marginal benefit.

### Decision 6: Detect thinking loops

**Choice:** Monitor the assistant's thinking blocks for repeated reasoning patterns. When the same or similar reasoning is repeated more than three times without advancing toward a solution, interrupt generation.

**Rationale:** The user reported a concrete case where the model was stuck in a thinking loop, repeating the same reasoning without making progress. Detecting and interrupting these loops saves tokens and improves UX.

**Alternatives considered:**
- No detection — wastes tokens on unproductive reasoning.
- Fixed token budget for thinking — arbitrary limit that may cut valid long reasoning short.

## Risks / Trade-offs

| Risk | Mitigation |
|------|-----------|
| OpenCode tool schema changes | Per-tool fixers can be updated independently |
| Detection marker changes in OpenCode | Marker is a simple string check; easy to update |
| Performance overhead from context extraction | Context extraction is O(n) over messages; negligible |
| Loop detection false positives | Existing heuristic already handles this; new choice UI is opt-in |

## Migration Plan

1. Create `agent/client/opencode/__init__.py` with `is_opencode_agent()` and `get_opencode_context()`.
2. Create `agent/client/opencode/tool_call_fixer.py` with per-tool fixers.
3. Wire detection into `engine_rest_common.py` alongside Veai.
4. Extend loop detection to offer user choices.
5. Add thinking loop detection to catch repeated reasoning without progress.
6. Add tests mirroring Veai test structure.

No rollback needed — changes are additive only.

## Open Questions

- Should OpenCode context extraction also handle the `AGENTS.md` file format (like the current system prompt)?
- What is the exact tool schema for OpenCode's `subagent` and `skill` tools? (Need to verify against actual logs.)