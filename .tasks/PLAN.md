# OpenAI Proxy Mode Integration — Task Execution Plan

## Your Mission

Add OpenAI proxy mode to the agent application, enabling users to run the application without local OpenVINO models by routing requests through an OpenAI-compatible API endpoint.

**Plan File:** `.tasks/PLAN.md`
**Tasks Directory:** `.tasks/`

## Execution Steps

### 1. Read This Plan
Review this file for the next incomplete task, key decisions, and information from previous agents.

### 2. Understand Your Task
Read your task file: `.tasks/task-XX-[name].md`
- **Goal** — What you are trying to achieve
- **Key Points** — Important considerations
- **Done When** — Objective acceptance criteria

### 3. Execute the Task
- Make necessary code changes
- Ensure code compiles without errors
- Verify all Done When criteria are met

### 4. Update This Plan
- Mark the task as completed in `## Task Plan`
- Add a 1-2 sentence outcome summary in `## Shared Context`
- Document only critical decisions that affect future tasks

### 5. Await Approval (MANDATORY)
Wait for user confirmation before proceeding to the next task.

### 6. Review Task List (MANDATORY)
Analyze remaining tasks based on what you learned:
- Did you encounter unexpected complexity?
- Should any tasks be split, merged, removed, or reordered?
- Are there missing tasks?

### 7. Present Review Findings (MANDATORY)
Always present your findings — even if no changes are needed — and await user approval before proceeding.

### 8. Update Task Files (if approved)
- Modify/create task files as needed
- Update `## Task Plan` in PLAN.md accordingly

---

## Task Plan

- [ ] `task-01-openai-cli-arguments.md`: Add --openai CLI Arguments
- [ ] `task-02-openai-early-exit.md`: Add Early-Exit Path for OpenAI Mode
- [ ] `task-03-conditional-app-init.md`: Conditional App Initialization
- [ ] `task-04-imports-dependencies.md`: Update Imports and Dependencies

---

## Shared Context

### Overview
Add OpenAI proxy mode to enable running the agent without local OpenVINO models. When `--openai` flag is set, skip all OpenVINO pipeline setup and route requests through a remote OpenAI-compatible API endpoint.

### Project Context
- `agent/__main__.py` — Main entry point with 553 lines, contains `main()` function with argparse setup
- `agent/server.py` — Contains `init_openai_engine()` at line 129, already implemented
- `agent/openai/engine_rest_openai.py` — OpenAI controller implementation
- Current initialization logic supports sequential and continuous batching engines based on device type

### Key Decisions
- OpenAI mode bypasses all local model loading and OpenVINO pipeline setup
- Uses existing `init_openai_engine()` from `agent/server.py`
- CLI arguments are optional and only active when `--openai` flag is set
- Model name override via `--openai_model` maps to `model_name_override` parameter in `init_openai_engine()`

### Caveats & Problems
- Need to handle the case where model_path doesn't exist (skip validation for OpenAI mode)
- Must ensure no circular import issues when importing from `agent.server`
- Existing parser detection logic is skipped entirely in OpenAI mode
- Thread management and stop_signal handling must be compatible with OpenAI engine