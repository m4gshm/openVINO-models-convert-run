# Proposal

## Why

The agent already supports Veai IDE integration with agent detection via system prompt markers and tool-call argument fixing. OpenCode IDE uses a different system prompt format ("You are an AI agent running in OpenCode") and requires similar support: agent detection, tool-call argument repair, and loop detection. Without this, OpenCode sessions cannot benefit from the same tool-call infrastructure that Veai enjoys.

Additionally, the loop detection mechanism (cycled tool call detection) exists but needs to be extended to support a "looping action" mode where the agent can be instructed to continue or interrupt a repeated tool call sequence.

## What Changes

- Add `is_opencode_agent()` function in `agent/client/opencode/__init__.py` that detects OpenCode agent sessions by checking for "You are an AI agent running in OpenCode" marker in the first message.
- Add `get_opencode_context()` function that extracts user context (OS, workdir) from the system prompt, mirroring the Veai implementation.
- Add OpenCode-specific tool call fixers in `agent/client/opencode/tool_call_fixer.py` that handle OpenCode's tool schema (edit, glob, grep, read, shell, skill, subagent, webfetch, websearch, write, execute).
- Wire the OpenCode detection into `engine_rest_common.py` alongside the existing Veai detection, so both agents get the same treatment.
- Add loop detection support for "looping actions" — when a tool call is detected as cycled, offer the user a choice to continue or interrupt.
- Add thinking loop detection — when the model repeats the same reasoning in thinking blocks without making progress, interrupt generation before tokens are wasted.

## Capabilities

### New Capabilities

- `opencode-agent-support`: Agent detection, context extraction, and tool-call fixing for OpenCode IDE integration.

## Impact

- `agent/client/opencode/` — new module (mirrors `agent/client/veai/`)
- `agent/openai/engine_rest_common.py` — wire OpenCode detection alongside Veai
- `agent/client/__init__.py` — export OpenCode helpers
- No breaking changes to existing Veai support