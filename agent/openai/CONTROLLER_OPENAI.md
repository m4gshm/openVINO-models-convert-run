# OpenAI Controller

OpenAI-compatible controller that uses OpenAI Python SDK to call remote/inference endpoints without requiring OpenVINO libraries.

## Overview

This module provides an alternative to the OpenVINO-based controllers (`ContinuousBatchingController`, `VlmController`) that works on machines without GPU support (e.g., macOS without Intel GPUs).

## Features

- **OpenAI SDK Integration**: Uses the official OpenAI Python client to call remote APIs
- **No OpenVINO Dependencies**: Runs on any machine with Python, no GPU required
- **Streaming Support**: Full streaming response support via chunk_generator
- **Error Handling**: Graceful error handling with proper error responses
- **Test Coverage**: Comprehensive test suite with pytest
- **Sync Methods**: Uses slots_sync() for testing (async slots endpoint removed)

## Architecture

```
OpenAiController
��── __init__() - Initialize OpenAI client
��── chunk_generator() - Generate chunks using OpenAI API
��── slots_sync() - Return slot info (sync version for testing)
��── shutdown() - Clean up resources
```

**Note**: The `slots()` async endpoint was removed from BaseController. Use `slots_sync()` for testing.

## Usage

```python
from agent.openai.engine_rest_openai import OpenAiController
from agent.openai import GenerateOpts
from agent.openai.engine_rest_common import ControllerConfig
from agent.inference.token_handler import TokenHandlerConfig
import threading
from agent.parser import Parser

# Initialize controller
config = ControllerConfig(
    model_name="your-model",
    max_prompt_len=4096,
    model_architectures={"llama"}
)

parser = Parser(...)
handler_config = TokenHandlerConfig()
stop_signal = threading.Event()
generate_opts = GenerateOpts()

controller = OpenAiController(
    config=config,
    parser=parser,
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    handler_config=handler_config,
    stop_signal=stop_signal,
    generate_opts=generate_opts
)
```

## Testing

Run tests:

```bash
python -m pytest agent/openai/engine_rest_openai_test.py -v
```

Run all tests:

```bash
python -m pytest agent/ -v
```

## Dependencies

- `openai>=2.48.0` - OpenAI Python SDK
- `starlette` - Web framework utilities
- `pydantic` - Data validation

## Files

- `engine_rest_openai.py` - Main controller implementation
- `engine_rest_openai_test.py` - Test suite
- `__init__.py` - Module exports

## License

Same as the parent project.