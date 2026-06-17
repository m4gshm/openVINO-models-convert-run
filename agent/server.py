import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

import openvino_genai
from fastapi import FastAPI
from openvino_genai import py_openvino_genai

from common.metric_mem import get_current_memory
from inference.streamer import StreamerConfig
from openai import GenerateConfig
from openai.engine_rest import Controller, ControllerConfig
from openai.logger_rest import LoggingRoute
from parser.qwen3 import Qwen3Parser


def init_engine(model: str, model_path: str, device: str, scheduler_config=py_openvino_genai.SchedulerConfig(),
                generate_config=GenerateConfig(), streamer_config=StreamerConfig(),
                pipeline_properties: dict[str, Any] | None = None,
                tokenizer_properties: dict[str, Any] | None = None,
                vision_encoder_properties: dict[str, Any] | None = None) -> FastAPI:
    log = logging.getLogger(__name__)

    log.info(f"model loading {model_path}, device: {device}")

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

        log.info(f"model loaded successfully")

        loaded_pipe_mem = get_current_memory()
        delta = loaded_pipe_mem - start_mem

        log.debug(f"consumed memory: {loaded_pipe_mem:.2f} MB, delta: {delta:.2f} MB")
    except Exception as e:
        log.error(f"instantiate pipeline error: {e}", exc_info=e)
        sys.exit(1)

    parser = Qwen3Parser()

    async def lifespan(app: FastAPI):
        yield
        controller.shutdown()

    app = FastAPI(lifespan=(asynccontextmanager(lifespan)))
    app_router = app.router
    app_router.route_class = LoggingRoute
    controller = Controller(config=ControllerConfig(model_name=model), parser=parser, pipe=pipe, router=app_router,
                            generate_config=generate_config, streamer_config=streamer_config)
    return app
