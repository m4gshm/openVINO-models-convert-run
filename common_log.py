import logging.config
import os
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute

logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

log_format_prefix = "%(asctime)s - %(name)s - %(levelname)s"
log_format_detailed = (log_format_prefix + " - [%(filename)s:%(lineno)d] - %(message)s")
log_format_simple = (log_format_prefix + " - %(message)s")

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": log_format_simple,
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "detailed": {
            "format": log_format_detailed
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "detailed",
            "filename": f"{logs_dir}/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        },
        "file_http": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "detailed",
            "filename": f"{logs_dir}/http.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        },
    },
    "loggers": {
        "http": {
            "level": "DEBUG",
            "handlers": ["console", "file_http"],
            "propagate": False,
        },
        "inference.stream": {
            "level": "INFO",
            "propagate": True,
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"],
    },
}

logging.config.dictConfig(LOGGING_CONFIG)


class LoggingRoute(APIRoute):

    def get_route_handler(self) -> Callable:
        log_http = logging.getLogger("http")

        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            body_bytes = await request.body()
            request_body_str = body_bytes.decode("utf-8") if body_bytes else None
            log_http.info(f"--> inbound {request.method} {request.url.path} body {request_body_str}")

            response: Response = await original_route_handler(request)

            if isinstance(response, StreamingResponse):
                log_http.info(f"<-- outbound {response.status_code}, media-type {response.media_type}")
                old_iterator = response.body_iterator

                async def re_iterator():
                    async for chunk in old_iterator:
                        if isinstance(chunk, str):
                            chunk_bytes = chunk.encode("utf-8")
                        else:
                            chunk_bytes = chunk

                        res_body_str = chunk_bytes.decode("utf-8", errors="ignore")

                        log_http.info(f"<-- outbound body chunk {res_body_str}")
                        yield chunk

                response.body_iterator = re_iterator()
            else:
                res_body_str = response.body.decode("utf-8", errors="ignore") if response.body else None
                log_http.info(
                    f"<-- outbound {response.status_code}, media-type {response.media_type}, body {res_body_str}")

            return response

        return custom_route_handler
