import os

log_format_prefix = "%(asctime)s - %(threadName)s - %(name)s - %(levelname)s"
log_format_detailed = (log_format_prefix + " - [%(filename)s:%(lineno)d] - %(message)s")
log_format_simple = (log_format_prefix + " - %(message)s")


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
        "loggers": {
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
            }
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
        },
    }
