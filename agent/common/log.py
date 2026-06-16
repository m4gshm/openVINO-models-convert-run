import logging.config
import os

logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

log_format_prefix = "%(asctime)s - %(threadName)s - %(name)s - %(levelname)s"
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
        "file_prompt": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": f"{logs_dir}/prompt.log",
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
    "loggers": {
        "http": {
            "level": "DEBUG",
            "handlers": ["file_http"],
            "propagate": False,
        },
        "veai.tool_call_fixer": {
            "level": "INFO",
            "propagate": True,
        },
        "inference": {
            "level": "DEBUG",
            "propagate": True,
        },
        "inference.stream": {
            "level": "DEBUG",
            "propagate": True,
        },
        "inference.prompt": {
            "level": "DEBUG",
            "propagate": False,
            "handlers": ["file_prompt"],
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"],
    },
}

logging.config.dictConfig(LOGGING_CONFIG)

