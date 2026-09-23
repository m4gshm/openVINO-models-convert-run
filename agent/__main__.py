import argparse
import logging.config
import os
import signal
import sys
from enum import Enum

from pydantic.json import pydantic_encoder

from agent.server import stop_signal
from agent.server_openai import run_openai_proxy, add_openai_args
from agent.server_openvino import run_openvino, default_model, add_openvino_args

os.environ["OPENVINO_LOG_LEVEL"] = "4"
os.environ["ONEDNN_VERBOSE"] = "ON"
os.environ["ONEDNN_VERBOSE_TIMESTAMP"] = "1"

log = logging.getLogger(__name__)



def handle_exit_signal(signum, frame):
    log.info(f"handle signal {signum}")
    stop_signal.set()


signal.signal(signal.SIGINT, handle_exit_signal)
signal.signal(signal.SIGTERM, handle_exit_signal)


def main():
    args_parser = argparse.ArgumentParser()
    add_openvino_args(args_parser)
    add_openai_args(args_parser)

    args = args_parser.parse_args()

    # Determine mode: --model for OpenVINO, --openai_base_url for OpenAI proxy
    has_model = args.model != default_model
    has_openai_base = bool(args.openai_base_url and args.openai_base_url != "https://api.openai.com/v1")

    if has_model and has_openai_base:
        log.error("ERROR: Cannot specify both --model and --openai_base_url. Use one or the other.")
        sys.exit(1)

    if not has_model and not has_openai_base:
        log.error("ERROR: Must specify either --model for OpenVINO mode or --openai_base_url for OpenAI proxy mode.")
        sys.exit(1)

    is_openai_mode = has_openai_base

    if is_openai_mode:
        run_openai_proxy(args)
    else:
        run_openvino(args)


if __name__ == "__main__":
    main()
