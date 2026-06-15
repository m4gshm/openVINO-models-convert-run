import logging
import sys
from typing import Any

import openvino_genai as ov_genai
from fastapi import FastAPI

from common.log import LoggingRoute
from common.metric_mem import get_current_memory
from inference.streamer import StreamerConfig
from openai import GenerateConfig
from openai.engine_rest import Controller
from parser.qwen3 import Qwen3Parser


def init_engine(model: str, model_path: str, device: str, generate_config: GenerateConfig = GenerateConfig(),
                streamer_config: StreamerConfig = StreamerConfig(),
                pipe_config: dict[str, Any] | None = None) -> FastAPI:
    log = logging.getLogger(__name__)

    log.info(f"model loading {model_path}, device: {device}")

    start_mem = get_current_memory()
    log.debug(f"consumed memory: {start_mem:.2f} MB")

    try:
        pipe = ov_genai.VLMPipeline(models_path=model_path, device=device, **pipe_config if pipe_config else {})
        # pipe = ov_genai.ContinuousBatchingPipeline(models_path=model_path, device=device_name,
        #                                            scheduler_config=scheduler_config,
        #                                            **config)

        log.info(f"model loaded successfully")

        loaded_pipe_mem = get_current_memory()
        pipe_cost = loaded_pipe_mem - start_mem

        log.debug(f"consumed memory: {loaded_pipe_mem:.2f} MB, pipe loading delta: {pipe_cost:.2f} MB")
    except Exception as e:
        log.error(f"instantiate pipeline error: {e}", exc_info=e)
        sys.exit(1)

    parser = Qwen3Parser()

    app: FastAPI = FastAPI()
    app.router.route_class = LoggingRoute
    controller = Controller(model_name=model, parser=parser, pipe=pipe, generate_config=generate_config,
                            streamer_config=streamer_config)
    app.include_router(controller.router)
    return app
