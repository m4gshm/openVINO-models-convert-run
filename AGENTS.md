# Agents

AI agent based on **OpenVINO GenAI** with OpenAI-compatible REST API. Supports streaming, tool calls, and IDE
integrations.

## Quick Start

### Prerequisites

- **Python >= 3.11**
- Windows (bat scripts) or Linux/macOS (direct Python)

### Installation

```bash
cd agent
python -m venv .venv
.venv\\Scripts\\activate  # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -e .
```

### Run

```bash
# Via wrapper script (Windows)
agent-dev.bat [arguments]

# Direct Python
.venv\\Scripts\\python agent.py [arguments]

# Via uvicorn
uvicorn agent.server:app --host 0.0.0.0 --port 8000
```

## Architecture

### Entry Point

`agent/__main__.py` — CLI args parsing, model loading, engine initialization.

### Core Components

- **REST API** (`agent/openai/`) — OpenAI-compatible endpoints (chat completions, models list)
- **Server** (`agent/server.py`) — FastAPI wrapping the agent engine
- **Inference** (`agent/inference/token_handler.py`) — token streaming, markdown formatting, tool call fixing
- **Parsers** (`agent/parser/`) — model-specific prompt parsers (Gemma, Qwen, LFM2)
- **Tools** (`agent/tool/`) — IDE tool implementations (read_file, edit_file, list_dir, etc.)

## Code Quality Rules

### Testing

- All new or modified code **must** have corresponding `_test.py` tests (co-located in the same module)
- Tests use **pytest** — run via `task test` or `uv run python -m pytest agent -v`
- Test files follow naming convention: `<module>_test.py` (e.g., `token_handler_test.py`)
- Before submitting changes: run full test suite (`task test`) — all tests must pass
- Coverage: new code should achieve at least 80% line coverage

### Static Analysis

- **Pylint** — run via `task lint` or `uv run pylint agent`
- Fix all errors and warnings before merging
- No new `E` (error) or `W` (warning) codes allowed in diffs

### Distribution

- Standalone builds via PyInstaller: `task dist`
- Output goes to `dist/` directory
- After build: verify `dist/` contains the executable

### Pre-commit Checklist

1. Run `task lint` — zero pylint violations
2. Run `task test` — all tests pass
3. Verify no new dependencies unless explicitly approved
4. Confirm model batch scripts still work after changes

Core: `openvino-genai`, `fastapi`, `uvicorn`, `openai`, `pydantic`, `json-repair`

Dev: `pylint`, `astroid`

Full list in `agent/pyproject.toml`.