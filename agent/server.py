import asyncio
import logging
import sys
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from openvino_genai import py_openvino_genai

from agent.common.metric_mem import get_current_memory
from agent.inference.token_handler import TokenHandlerConfig
from agent.openai import GenerateOpts
from agent.openai.engine_rest_cb import ContinuousBatchingController, ControllerConfig
from agent.openai.engine_rest_common import BaseController
from agent.openai.engine_rest_openai import OpenAiController
from agent.openai.engine_rest_vlm import VlmController
from agent.openai.logger_rest import LoggingRoute
from agent.parser import Parser

log = logging.getLogger(__name__)


def init_continuous_batching_engine(controller_config: ControllerConfig,
                                    handler_config: TokenHandlerConfig,
                                    model_path: str, device: str, parser: Parser,
                                    stop_signal: threading.Event,
                                    scheduler_config=py_openvino_genai.SchedulerConfig(),
                                    generate_opts=GenerateOpts(),
                                    pipeline_properties: dict[str, Any] | None = None,
                                    tokenizer_properties: dict[str, Any] | None = None,
                                    vision_encoder_properties: dict[str, Any] | None = None) -> FastAPI:
    start_mem = get_current_memory()
    log.debug(f"consumed memory: {start_mem:.2f} MB")

    if not pipeline_properties:
        pipeline_properties = {}
    if not tokenizer_properties:
        tokenizer_properties = {}
    if not vision_encoder_properties:
        vision_encoder_properties = {}
    try:
        pipe = py_openvino_genai.ContinuousBatchingPipeline(models_path=model_path,
                                                            scheduler_config=scheduler_config,
                                                            device=device,
                                                            properties=pipeline_properties,
                                                            tokenizer_properties=tokenizer_properties,
                                                            vision_encoder_properties=vision_encoder_properties)
        log.info(f"model loaded successfully, pipe {type(pipe)}")

        loaded_pipe_mem = get_current_memory()
        delta = loaded_pipe_mem - start_mem

        log.debug(f"consumed memory: {loaded_pipe_mem:.2f} MB, delta: {delta:.2f} MB")
    except Exception as e:
        log.error(f"instantiate pipeline error: {e}", exc_info=e)
        sys.exit(1)

    return new_app(ContinuousBatchingController(config=controller_config,
                                                parser=parser, pipe=pipe,
                                                generate_opts=generate_opts,
                                                handler_config=handler_config,
                                                stop_signal=stop_signal))


def init_sequential_engine(controller_config: ControllerConfig,
                           handler_config: TokenHandlerConfig,
                           model_path: str,
                           device: str, vlm: bool, parser: Parser,
                           stop_signal: threading.Event,
                           scheduler_config: py_openvino_genai.SchedulerConfig | None = None,
                           generate_opts=GenerateOpts(),
                           pipeline_properties: dict[str, Any] | None = None) -> FastAPI:
    if not pipeline_properties:
        pipeline_properties = {}
    if scheduler_config:
        pipeline_properties["scheduler_config"] = scheduler_config

    start_mem = get_current_memory()
    log.debug(f"consumed memory: {start_mem:.2f} MB")

    pipe = (
        py_openvino_genai.VLMPipeline(models_path=model_path, device=device, **pipeline_properties) if vlm else
        py_openvino_genai.LLMPipeline(models_path=model_path, device=device, **pipeline_properties)
    )

    log.info(f"model loaded successfully, pipe {type(pipe)}")
    loaded_pipe_mem = get_current_memory()
    delta = loaded_pipe_mem - start_mem

    log.debug(f"consumed memory: {loaded_pipe_mem:.2f} MB, delta: {delta:.2f} MB")

    return new_app(VlmController(config=controller_config,
                                 parser=parser, pipe=pipe,
                                 generate_opts=generate_opts,
                                 handler_config=handler_config,
                                 stop_signal=stop_signal))


def new_app(controller: BaseController) -> FastAPI:
    async def lifespan(app: FastAPI):
        app.state.main_loop = asyncio.get_running_loop()
        # stop_signal.is_set()
        yield
        log.info("controller is shutdown")
        controller.shutdown()

    app = FastAPI(lifespan=(asynccontextmanager(lifespan)))
    app_router = app.router
    app_router.route_class = LoggingRoute
    # OpenAI compatible endpoints
    app_router.post("/v1/completions", response_model_exclude_none=True)(controller.completions)
    app_router.post("/v1/chat/completions", response_model_exclude_none=True)(controller.chat)
    app_router.get("/v1/models", response_model_exclude_none=True)(controller.models)
    
    # llama.cpp compatible endpoints
    app_router.post("/completion")(controller.completions)
    app_router.post("/chat/completion")(controller.chat)
    app_router.get("/health")(controller.health)
    app_router.get("/models")(controller.models)
    app_router.post("/tokenize")(controller.tokenize)
    app_router.post("/detokenize")(controller.detokenize)
    
    app.add_exception_handler(RequestValidationError, controller.validation_exception_handler)
    return app


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

