"""OpenAI proxy mode: forwards requests to a remote OpenAI-compatible API."""
import logging
import sys
import threading
from argparse import ArgumentParser

from fastapi import FastAPI

from agent.common.metric_mem import get_current_memory
from agent.inference.token_handler import TokenHandlerConfig
from agent.openai import get_default_generate_opts, GenerateOpts
from agent.openai.engine_rest_common import ControllerConfig
from agent.openai.engine_rest_openai import OpenAiController
from agent.server import new_app, serve_until_stop

log = logging.getLogger(__name__)


def run_openai_proxy(args):
    """Run application in OpenAI proxy mode without OpenVINO dependencies.

    This mode forwards all requests to an external OpenAI-compatible API
    (e.g. https://api.openai.com/v1). Useful for machines without GPU
    where OpenVINO cannot load native accelerators (e.g. macOS).
    """
    if not args.openai_api_key:
        log.error("OpenAI API key is required for OpenAI proxy mode")
        sys.exit(1)

    log.info("starting OpenAI proxy mode: base_url=%s", args.openai_base_url)

    # --- Build configuration objects ---
    # The proxy neither decodes media nor judges capabilities: content parts are forwarded as they
    # came, and /v1/models reports what the remote endpoint declares for each of its models.
    controller_config = ControllerConfig(
        max_prompt_len=4096,
        model_architectures=set(),
        is_fix_tool_type=False,
        is_detect_cycled_tool_call=False
    )

    handler_config = TokenHandlerConfig()
    stop_signal = threading.Event()
    default_generate_opts = get_default_generate_opts()
    generate_opts = default_generate_opts

    # --- Initialize engine and start server ---
    app = init_openai_engine(
        controller_config=controller_config,
        handler_config=handler_config,
        api_key=args.openai_api_key,
        base_url=args.openai_base_url,
        stop_signal=stop_signal,
        generate_opts=generate_opts,
    )

    serve_until_stop(app, args.host, args.port, stop_signal, log)


# Keyword config-object factory mirroring the controller constructor.
# pylint: disable=too-many-arguments,too-many-positional-arguments
def init_openai_engine(controller_config: ControllerConfig,
                       handler_config: TokenHandlerConfig,
                       api_key: str,
                       base_url: str,
                       stop_signal: threading.Event,
                       generate_opts=GenerateOpts()) -> FastAPI:
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

    Returns:
        FastAPI: Initialized FastAPI application with OpenAI controller
    """
    start_mem = get_current_memory()
    log.debug("consumed memory: %.2f MB", start_mem)

    log.info("initializing OpenAI engine: base_url=%s", base_url)

    controller = OpenAiController(
        config=controller_config,
        api_key=api_key,
        base_url=base_url,
        generate_opts=generate_opts,
        handler_config=handler_config,
        stop_signal=stop_signal,
    )

    loaded_mem = get_current_memory()
    delta = loaded_mem - start_mem
    log.debug("consumed memory: %.2f MB, delta: %.2f MB", loaded_mem, delta)

    return new_app(controller)


def add_openai_args(args_parser: ArgumentParser):
    """Register CLI arguments of the OpenAI proxy mode."""
    args_parser.add_argument("--openai_api_key", type=str, default="", help="OpenAI API key")
    args_parser.add_argument("--openai_base_url", type=str, default="https://api.openai.com/v1",
                             help="OpenAI API base URL")
