import json
import logging
from typing import Any

import json_repair

from common.openai_model import ToolCall

log = logging.getLogger(__name__)


def fix_incorrect_arguments(tool_call: ToolCall) -> ToolCall:
    # function = tool_call.function
    # if "ask_user_with_options" == function.name:
    #     args_raw = function.arguments
    #     args: dict[str, Any] | None = None
    #     try:
    #         args = json.loads(args_raw)
    #     except json.decoder.JSONDecodeError as e:
    #         log.error(f"bad arguments of function '{function.name}', args '{args_raw}': {e}")
    #         args = json_repair.loads(args_raw)
    #         log.info(f"repaired arguments '{args}'")
    #
    #     if args:
    #         options_raw = args.get("options")
    #         options: list[str] | None = None
    #         if options_raw and isinstance(options_raw, str):
    #             try:
    #                 options = json.loads(options_raw)
    #             except json.decoder.JSONDecodeError as e:
    #                 log.error(f"bad options of function '{function.name}', options: '{options_raw}': {e}")
    #                 options = json_repair.loads(options_raw)
    #                 log.info(f"repaired options '{options}'")
    #         elif options_raw:
    #             log.error(f"unexpected options type, function '{function.name}', args '{args_raw}', "
    #                       f"options type {type(options_raw)}")
    #         else:
    #             log.error(f"missing options in args, function '{function.name}', args '{args_raw}'")
    #
    #         if options:
    #             args["options"] = json.dumps(options, ensure_ascii=False)
    #
    #         function.arguments = json.dumps(args, ensure_ascii=False)
    #         log.info(f"function after repairing, function {function.name}, arguments '{args}'")

    return tool_call
