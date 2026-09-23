# Task 02: Add Early-Exit Path for OpenAI Mode

**Type:** Code Modification

## Goal

When `--openai` is specified, skip the model_path validation and entire OpenVINO pipeline setup.

## What to Do

- Check if `--openai` flag is set early in `main()`
- Skip the model_path existence validation block (lines 195-210)
- Skip all OpenVINO pipeline property setup (CPU/GPU/NPU properties)
- Skip parser detection logic based on model architecture
- Proceed directly to OpenAI engine initialization

## Files/Areas

- `agent/__main__.py` — Add conditional check after argument parsing

## Key Points

- The model_path validation currently exits with error if path doesn't exist
- OpenAI mode doesn't need local model files at all
- Skip the architecture detection and parser initialization for OpenAI mode
- Keep all the common setup (logging, generate_opts, scheduler_config) that's still needed

## Done When

- [ ] OpenAI mode skips model_path validation without errors
- [ ] OpenAI mode bypasses parser detection logic
- [ ] OpenAI mode still initializes logging and generate_opts correctly