# Task 03: Conditional App Initialization

**Type:** Code Modification

## Goal

Restructure app initialization to conditionally use either OpenAI engine or OpenVINO engines.

## What to Do

- Replace the existing conditional logic that chooses between `init_sequential_engine` and `init_continuous_batching_engine`
- Add a new branch for OpenAI mode that calls `init_openai_engine()`
- Ensure the three initialization paths are mutually exclusive based on `--openai` flag

## Files/Areas

- `agent/__main__.py` — Lines 510-530 (app initialization block)

## Key Points

- Current logic: `if is_not_cb` → `init_sequential_engine`, else → `init_continuous_batching_engine`
- New logic: if `--openai` → `init_openai_engine`, elif `is_not_cb` → `init_sequential_engine`, else → `init_continuous_batching_engine`
- Must preserve existing functionality for non-OpenAI modes
- `init_openai_engine` signature from `agent/server.py` line 129

## Done When

- [ ] App initialization handles all three paths correctly
- [ ] OpenAI mode uses `init_openai_engine()` with correct parameters
- [ ] Sequential and continuous batching paths remain unchanged
- [ ] All initialization paths pass the same parameter types