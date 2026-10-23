"""Logging configuration shared by the server entry points."""
import logging
import os

LOG_FORMAT_PREFIX = "%(asctime)s - %(threadName)s - %(name)s - %(levelname)s"
LOG_FORMAT_DETAILED = LOG_FORMAT_PREFIX + " - [%(filename)s:%(lineno)d] - %(message)s"
LOG_FORMAT_SIMPLE = LOG_FORMAT_PREFIX + " - %(message)s"


# Filter-style callback class: a single hook method is its whole interface.
# pylint: disable=too-few-public-methods
class DropClientResetFilter(logging.Filter):
    """Drop harmless asyncio noise produced when a client resets a connection.

    On the Windows Proactor loop the transport cleanup
    (_ProactorBasePipeTransport._call_connection_lost) calls sock.shutdown()
    on a socket the client already reset, which raises ConnectionResetError
    [WinError 10054] logged by the "asyncio" logger as "Exception in
    callback ...". The request is already finished at that point, so the
    record carries no information. Everything else is kept.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "asyncio" or record.exc_info is None:
            return True
        exc = record.exc_info[1]
        if not isinstance(exc, ConnectionResetError) or getattr(exc, "winerror", None) != 10054:
            return True
        return "_call_connection_lost" not in record.getMessage()


def logging_config(logs_dir: str):
    """Build a dictConfig for console, app and per-purpose rotating file handlers."""
    os.makedirs(logs_dir, exist_ok=True)
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": LOG_FORMAT_SIMPLE,
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": LOG_FORMAT_DETAILED
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
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": f"{logs_dir}/app.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            },
            "file_prompt": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": f"{logs_dir}/prompt.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            },
            "file_generated": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": f"{logs_dir}/generated.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            },
            "file_http": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "simple",
                "filename": f"{logs_dir}/http.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            },
        },
        "filters": {
            "drop_client_reset": {
                "()": "agent.common.log.DropClientResetFilter",
            },
        },
        "loggers": {
            "uvicorn": {
                "level": "INFO",
                "propagate": True,
            },
            "asyncio": {
                "filters": ["drop_client_reset"],
                "propagate": True,
            },
            "http": {
                "level": "DEBUG",
                "handlers": ["file_http"],
                "propagate": True,
            },
            "agent.inference": {
                "level": "DEBUG",
                "propagate": True,
            },
            "agent.inference.token_handler": {
                "level": "DEBUG",
                "propagate": True,
            },
            "agent.inference.token_metrics": {
                "level": "INFO",
                "propagate": True,
            },
            "agent.inference.prompt": {
                "level": "DEBUG",
                "propagate": False,
                "handlers": ["file_prompt"],
            },
            "agent.inference.generated": {
                "level": "DEBUG",
                "propagate": True,
                "handlers": ["file_generated"],
            },
            "client.tool_call.last.generated": {
                "level": "DEBUG",
                "propagate": True,
                "handlers": ["file_generated"],
            }
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
        },
    }
