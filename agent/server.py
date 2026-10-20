"""FastAPI app factory shared by all engines, plus the common serve-and-wait loop."""
import asyncio
import logging
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from agent.ollama.ollama_handlers import OllamaApi
from agent.openai.engine_rest_common import BaseController
from agent.openai.logger_rest import LoggingRoute

log = logging.getLogger(__name__)

stop_signal = threading.Event()


def serve_until_stop(app: FastAPI, host: str, port: int, stop_event: threading.Event,
                     logger: logging.Logger) -> None:
    """Run uvicorn on a daemon thread and block the caller until stop_event is set."""
    logger.info("listening %s:%s", host, port)

    def server_handle():
        uvicorn.run(app, host=host, port=port, reload=False, timeout_graceful_shutdown=0)

    server_thread = threading.Thread(target=server_handle, daemon=True)
    server_thread.start()

    try:
        stopped = False
        while not stopped:
            stopped = stop_event.wait(timeout=1)
        logger.debug("main thread finish")
    except Exception as e:  # pylint: disable=broad-exception-caught  # the wait loop must outlive any error
        logger.debug("main thread finish with error: %s", e)


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

    app = FastAPI(lifespan=asynccontextmanager(lifespan))
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

    # Ollama compatible endpoints
    ollama = OllamaApi(controller)
    app_router.get("/api/tags", response_model_exclude_none=True)(ollama.tags)
    app_router.get("/api/version", response_model_exclude_none=True)(ollama.version)
    app_router.post("/api/show", response_model_exclude_none=True)(ollama.show)
    app_router.post("/api/chat", response_model_exclude_none=True)(ollama.chat)
    app_router.post("/api/generate", response_model_exclude_none=True)(ollama.generate)
    # app_router.post("/tokenize")(controller.tokenize)
    # app_router.post("/detokenize")(controller.detokenize)

    app.add_exception_handler(RequestValidationError, controller.validation_exception_handler)
    return app
