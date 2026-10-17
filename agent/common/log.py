import logging
import os

log_format_prefix = "%(asctime)s - %(threadName)s - %(name)s - %(levelname)s"
log_format_detailed = (log_format_prefix + " - [%(filename)s:%(lineno)d] - %(message)s")
log_format_simple = (log_format_prefix + " - %(message)s")


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
    os.makedirs(logs_dir, exist_ok=True)
    return {
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
