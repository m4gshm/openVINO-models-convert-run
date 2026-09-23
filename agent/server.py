import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from agent.openai.engine_rest_common import BaseController
from agent.openai.logger_rest import LoggingRoute

log = logging.getLogger(__name__)

stop_signal = threading.Event()


def enum_value[T: Enum](member: T) -> Any:
    return member.value


def new_app(controller: BaseController) -> FastAPI:
    """Factory: create a FastAPI app wired to a BaseController instance.

    Registers OpenAI-compatible endpoints (/v1/*) and llama.cpp-compatible
    endpoints (/completion, /chat/completion, etc.) on the same app.
    Lifespan handler stores the running asyncio loop and calls controller
    shutdown on app teardown.
    """

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
