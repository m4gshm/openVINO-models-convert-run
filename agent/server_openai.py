import sys
import threading

import uvicorn
from fastapi import FastAPI

from agent.__main__ import log
from agent.common.metric_mem import get_current_memory
from agent.inference.token_handler import TokenHandlerConfig
from agent.openai import get_default_generate_opts, GenerateOpts
from agent.openai.engine_rest_common import ControllerConfig
from agent.openai.engine_rest_openai import OpenAiController
from agent.parser import Parser
from agent.server import log, new_app


def run_openai_proxy(args):
    """Run application in OpenAI proxy mode without OpenVINO dependencies.

    This mode forwards all requests to an external OpenAI-compatible API
    (e.g. https://api.openai.com/v1). Useful for machines without GPU
    where OpenVINO cannot load native accelerators (e.g. macOS).
    """
    if not args.openai_api_key:
        log.error("OpenAI API key is required for OpenAI proxy mode")
        sys.exit(1)

    log.info(f"starting OpenAI proxy mode: base_url={args.openai_base_url}, model={args.openai_model or args.model}")

    from agent.parser import Parser
    from agent.openai.engine_rest_common import ControllerConfig
    from agent.inference.token_handler import TokenHandlerConfig

    # --- Build configuration objects ---
    controller_config = ControllerConfig(
        model_name=args.openai_model or args.model,
        max_prompt_len=4096,
        model_architectures=set(),
        is_fix_tool_type=False,
        is_detect_cycled_tool_call=False
    )

    handler_config = TokenHandlerConfig()
    stop_signal = threading.Event()
    default_generate_opts = get_default_generate_opts()
    generate_opts = default_generate_opts
    parser = Parser()

    # --- Initialize engine and start server ---
    app = init_openai_engine(
        controller_config=controller_config,
        handler_config=handler_config,
        api_key=args.openai_api_key,
        base_url=args.openai_base_url,
        parser=parser,
        stop_signal=stop_signal,
        generate_opts=generate_opts,
        model_name_override=args.openai_model or args.model
    )

    log.info(f"listening {args.host}:{args.port}")

    def server_handle():
        uvicorn.run(app, host=args.host, port=args.port, reload=False, timeout_graceful_shutdown=0)

    server_thread = threading.Thread(target=server_handle, daemon=True)
    server_thread.start()

    try:
        stopped = False
        while not stopped:
            stopped = stop_signal.wait(timeout=1)
        log.debug(f"main thread finish")
    except Exception as e:
        log.debug(f"main thread finish with error: {e}")


def init_openai_engine(controller_config: ControllerConfig,
                       handler_config: TokenHandlerConfig,
                       api_key: str,
                       base_url: str,
                       parser: Parser,
                       stop_signal: threading.Event,
                       generate_opts=GenerateOpts(),
                       model_name_override: str | None = None) -> FastAPI:
    """
    Initialize OpenAI-compatible controller without OpenVINO dependencies.

    This enables running on machines without GPU (e.g., macOS) where
    OpenVINO cannot load native accelerators.

    Args:
        controller_config: Controller configuration (model_name, max_prompt_len, etc.)
        handler_config: Token handler configuration
        api_key: OpenAI API key for authentication
        base_url: OpenAI-compatible API endpoint URL (e.g., https://api.openai.com/v1)
        parser: Parser for model-specific token handling
        stop_signal: Threading event for stopping inference
        generate_opts: Generation options (temperature, max_tokens, etc.)
        model_name_override: Optional override for model name in responses

    Returns:
        FastAPI: Initialized FastAPI application with OpenAI controller
    """
    start_mem = get_current_memory()
    log.debug(f"consumed memory: {start_mem:.2f} MB")

    log.info(f"initializing OpenAI engine: base_url={base_url}, model={model_name_override or controller_config.model_name}")

    controller = OpenAiController(
        config=controller_config,
        parser=parser,
        api_key=api_key,
        base_url=base_url,
        generate_opts=generate_opts,
        handler_config=handler_config,
        stop_signal=stop_signal,
        model_name_override=model_name_override
    )

    loaded_mem = get_current_memory()
    delta = loaded_mem - start_mem
    log.debug(f"consumed memory: {loaded_mem:.2f} MB, delta: {delta:.2f} MB")

    return new_app(controller)
