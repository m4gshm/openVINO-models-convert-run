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
- **Multimodal** (`agent/multimodal/`) — input modality detection, content parts parsing, PNG/WAV
  decoding into `ov.Tensor`, media tags for the prompt
- **Tools** (`agent/tool/`) — IDE tool implementations (read_file, edit_file, list_dir, etc.)

## Multimodal Input

Chat requests may carry images and audio next to text when the loaded model supports them.
Capabilities are read from the model itself (see «Capability detection» below) — there is no CLI
flag for them. Text-only models reject media with an explanatory assistant message instead of a
hard error.

### Request format

```bash
curl http://127.0.0.1:8888/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "what is on the picture?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    {"type": "input_audio", "input_audio": {"data": "...", "format": "wav"}}
  ]}]}'
```

Accepted part shapes: OpenAI (`image_url`, `input_audio`), HuggingFace style (`image`, `audio`,
`video`) and `video_url`. Media may arrive as a base64 data URL, inline base64 `data`, or an
`http(s)` URL; local paths require `--allow_local_media_files on`.

`/v1/models` reports `supported_modalities` of the running engine. In OpenAI proxy mode the value is
passed through from the remote endpoint: the OpenAI response schema has no such field
(`Model` is `id/object/created/owned_by`), so servers that do not declare it make the field simply
absent rather than guessed.

### Capability detection

Order of precedence in `agent/multimodal/model_capabilities.py`:

1. **Media encoder IR files** in the model folder — the ground truth of what the pipeline loads:
   `openvino_vision_embeddings_model.xml` (also `..._merger`, `..._pos`, `openvino_resampler`) → images,
   `openvino_audio_embeddings_model.xml` → audio.
2. **`config.json`** keys when there is no IR: `vision_config` / `image_token_id`,
   `audio_config` / `audio_token_id`, `video_token_id` (searched in nested sub-configs too).
3. **`processor_config.json` / `preprocessor_config.json`**: `image_processor`, `feature_extractor`
   (its `sampling_rate` also sets the audio resample target), `video_processor`.
4. **Architecture-name heuristics** (`agent/multimodal/modality.detect_modalities`) as a fallback for
   GGUF files and folders without any of the above.

File presence matters: `Qwen3.5-9B-int4-ov` (`Qwen3_5ForConditionalGeneration`) ships
`openvino_vision_embeddings_model.xml` and is image-capable, while `Qwen3.5-9B-abliterated-int4-ov`
(`Qwen3_5ForCausalLM`) of the same family ships no media encoder at all; none of the Qwen3.5/3.6
exports here has `openvino_audio_embeddings_model.xml`, so audio stays off for them. Audio turns on
only for a folder that really carries the audio IR (a Qwen3-Omni or Qwen2-Audio style export).
`--pipe` still bounds what can be delivered at all: `LLM` accepts text only, `CB` accepts images but
has no audio input; `--pipe` itself defaults to `VLM` when the folder shows media encoders. To
deliberately run a multimodal checkpoint as text only, use `--pipe LLM`. Video is detected but never
advertised, because decoding a video container needs an
external demuxer — send frames as images.

### Formats

- **Built in (no dependencies):** PNG (8/16 bit, color, gray, RGBA, palette, non-interlaced) and
  WAV (PCM 8/16/24/32 bit, IEEE float32, any rate/channels — mixed down to mono, resampled to the
  rate the model declares, 16 kHz by default).
- **Via `pillow` (regular dependency):** JPEG/WEBP/GIF/BMP/TIFF — and EXIF orientation is applied,
  so photos taken on a phone reach the model upright.
- **Optional:** MP3/FLAC/OGG/M4A need `soundfile` or `librosa`. Without them such parts fail with
  a message naming the missing package.

### Options

| Argument | Default | Meaning |
| --- | --- | --- |
| `--max_media_mb` | 20 | size limit of one attachment |
| `--max_media_items` | 16 | attachments per request |
| `--allow_local_media_files` | off | read media from local file paths |

Every request repeats the whole conversation (the OpenAI protocol is stateless), so media of
earlier turns is sent and decoded again; each item becomes an OpenVINO tag in the prompt
(`<ov_genai_image_0>`, `<ov_genai_audio_1>`, ...). Media expands the prompt with many tokens, so
`--max_prompt_len` must leave room for it.

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

Core: `openvino-genai`, `fastapi`, `uvicorn`, `openai`, `pydantic`, `json-repair`, `httpx`, `numpy`,
`pillow`

Optional for multimodal input: `soundfile` or `librosa` (more audio formats)

Runtime dependencies must be listed in **both** `pyproject.toml` and `agent/pyproject.toml`: the venv
is synced from the root project, which does not depend on the `agent` workspace member, so a package
added only to `agent/pyproject.toml` never lands in `.venv`.

Dev: `pylint`, `astroid`

Full list in `agent/pyproject.toml`.