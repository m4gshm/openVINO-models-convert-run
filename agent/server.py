import asyncio
import logging
import sys
import threading
from contextlib import asynccontextmanager
from typing import Any

import openvino_genai
from fastapi import FastAPI
from openvino_genai import py_openvino_genai

from agent.common.metric_mem import get_current_memory
from agent.inference.token_handler import TokenHandlerConfig
from agent.openai import GenerateOpts
from agent.openai.engine_rest import ContinuousBatchingController, ControllerConfig
from agent.openai.engine_rest_common import BaseController
from agent.openai.engine_rest_vlm import VlmController
from agent.openai.logger_rest import LoggingRoute
from agent.parser import Parser

log = logging.getLogger(__name__)


def init_continuous_batching_engine(model: str, model_path: str, device: str, parser: Parser,
                                    is_fix_tool_type: bool, stop_signal: threading.Event,
                                    scheduler_config=py_openvino_genai.SchedulerConfig(),
                                    generate_config=GenerateOpts(), handler_config=TokenHandlerConfig(),
                                    pipeline_properties: dict[str, Any] | None = None,
                                    tokenizer_properties: dict[str, Any] | None = None,
                                    vision_encoder_properties: dict[str, Any] | None = None,
                                    chat_template='') -> FastAPI:
    log.info(f"model loading {model_path}, device: {device}, scheduler_config {scheduler_config.to_string()}")

    start_mem = get_current_memory()
    log.debug(f"consumed memory: {start_mem:.2f} MB")

    if not pipeline_properties:
        pipeline_properties = {}
    if not tokenizer_properties:
        tokenizer_properties = {}
    if not vision_encoder_properties:
        vision_encoder_properties = {}
    try:
        pipe = openvino_genai.ContinuousBatchingPipeline(models_path=model_path,
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

    return new_app(ContinuousBatchingController(config=ControllerConfig(model_name=model), parser=parser, pipe=pipe,
                                                chat_template=chat_template,
                                                generate_config=generate_config, handler_config=handler_config,
                                                is_fix_tool_type=is_fix_tool_type, stop_signal=stop_signal))


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
    app_router.post("/v1/completions")(controller.completions)
    app_router.post("/v1/chat/completions")(controller.chat)
    app_router.get(path="/v1/models", response_model_exclude_none=True)(controller.models)
    return app


def init_sequential_engine(model_name: str, model_path: str, device: str, vlm: bool, parser: Parser,
                           is_fix_tool_type: bool, stop_signal: threading.Event,
                           generate_config=GenerateOpts(),
                           handler_config=TokenHandlerConfig(),
                           pipeline_properties: dict[str, Any] | None = None, chat_template='') -> FastAPI:
    if not pipeline_properties:
        pipeline_properties = {}

    log.info(f"model loading {model_name}, device: {device}")

    start_mem = get_current_memory()
    log.debug(f"consumed memory: {start_mem:.2f} MB")

    if vlm:
        pipe = openvino_genai.VLMPipeline(models_path=model_path, device=device, **pipeline_properties)
    else:
        pipe = openvino_genai.LLMPipeline(models_path=model_path, device=device, **pipeline_properties)
    if chat_template:
        pipe.set_chat_template(chat_template)

    log.info(f"model loaded successfully, pipe {type(pipe)}")
    loaded_pipe_mem = get_current_memory()
    delta = loaded_pipe_mem - start_mem

    log.debug(f"consumed memory: {loaded_pipe_mem:.2f} MB, delta: {delta:.2f} MB")

    return new_app(VlmController(config=ControllerConfig(model_name=model_name), parser=parser, pipe=pipe,
                                 generate_config=generate_config, chat_template=chat_template,
                                 handler_config=handler_config, is_fix_tool_type=is_fix_tool_type,
                                 stop_signal=stop_signal))
