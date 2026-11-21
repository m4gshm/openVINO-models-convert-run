# Tasks

## 1. OpenCode Agent Detection Module

- [x] 1.1 Create `agent/client/opencode/__init__.py` with `is_opencode_agent()` and `get_opencode_context()` functions, verify by importing the module and testing with sample messages
- [x] 1.2 Add unit tests for `is_opencode_agent()` in `agent/client/opencode/__init___test.py`, verify tests pass with `pytest`
- [x] 1.3 Add unit tests for `get_opencode_context()` in `agent/client/opencode/user_context_test.py`, verify tests pass with `pytest`

## 2. OpenCode Tool-Call Fixers

- [x] 2.1 Create `agent/client/opencode/tool_call_args.py` with shared argument utilities (mirroring Veai's `tool_call_args.py`), verify imports work correctly
- [x] 2.2 Create `agent/client/opencode/tool_call_fixer.py` with per-tool fixers for edit, glob, grep, read, shell, skill, subagent, webfetch, websearch, write, execute, verify each fixer handles its tool's field names
- [x] 2.3 Create `agent/client/opencode/tool_call_fixer_test.py` with tests for each fixer, verify tests pass with `pytest`

## 3. Engine Wiring

- [ ] 3.1 Wire OpenCode detection into `agent/openai/engine_rest_common.py` alongside Veai detection, verify both Veai and OpenCode agents trigger the correct pipeline
- [ ] 3.2 Export OpenCode helpers from `agent/client/__init__.py`, verify imports work from consumer modules
- [ ] 3.3 Run full test suite (`task test`), verify all tests pass

## 4. Loop Detection Extension

- [ ] 4.1 Extend `validate_messages()` in `engine_rest_common.py` to offer user choices (continue/interrupt) when a cycled tool call is detected, verify the choice UI appears in the response
- [ ] 4.2 Handle user's choice response (continue/interrupt) in the message processing pipeline, verify the correct action is taken based on the choice
- [ ] 4.3 Add tests for loop detection with user choices, verify tests pass with `pytest`

## 5. Thinking Loop Detection

- [ ] 5.1 Create `agent/preprocess/thinking_loop_detector.py` with logic to detect repeated reasoning in thinking blocks, verify it catches the pattern from the user's example (same reasoning repeated 3+ times)
- [ ] 5.2 Integrate thinking loop detection into the message processing pipeline, verify it interrupts generation before tokens are wasted
- [ ] 5.3 Add tests for thinking loop detection, verify tests pass with `pytest`

## 6. Integration Verification

- [ ] 6.1 Run e2e tests with OpenCode agent session, verify the agent is correctly detected and tool calls are fixed
- [ ] 6.2 Verify loop detection works with actual OpenCode tool calls, verify the choice UI appears and user choices are respected
- [ ] 6.3 Run static analysis (`task lint`, `task typecheck`), verify zero pylint violations and zero mypy errors