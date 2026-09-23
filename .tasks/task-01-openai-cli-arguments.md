# Task 01: Add --openai CLI Arguments

**Type:** Code Modification

## Goal

Add OpenAI proxy mode CLI arguments to the argument parser in `__main__.py`.

## What to Do

- Add `--openai` boolean flag to enable OpenAI proxy mode
- Add `--openai_api_key` string argument for the API key
- Add `--openai_base_url` string argument for the API endpoint URL
- Add `--openai_model` string argument for model name override

## Files/Areas

- `agent/__main__.py` — Add argument definitions near line 130-180

## Key Points

- Follow the existing argument style in the file
- Place OpenAI arguments together as a logical group
- Use `argparse` standard patterns consistent with existing code
- No default values needed for these new arguments (they're optional when `--openai` is set)

## Done When

- [ ] All four OpenAI CLI arguments are added to the argument parser
- [ ] Arguments follow the existing code style and formatting conventions