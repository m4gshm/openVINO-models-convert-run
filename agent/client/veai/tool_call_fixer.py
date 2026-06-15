import logging
from typing import Any

from common.openai_model import ToolCall, ToolDefinition

log = logging.getLogger(__name__)


def fix_incorrect_arguments(tool_call: ToolCall) -> ToolCall:
    function = tool_call.function
    if "edit_file" == function.name:
        pass
    elif "write_file" == function.name:
        pass
    elif "read_file" == function.name:
        pass
    # if "ask_user_with_options" == function.name:
    #     args_raw = function.arguments
    #     try:
    #         args = json.loads(args_raw)
    #     except json.decoder.JSONDecodeError as e:
    #         log.error(f"bad arguments of function '{function.name}', args '{args_raw}': {e}")
    #         args = json_repair.loads(args_raw)
    #         log.info(f"repaired arguments '{args}'")
    #
    #     if args:
    #         options_raw = args.get("options")
    #         options: list[Any] | None = None
    #         if options_raw:
    #             if isinstance(options_raw, str):
    #                 try:
    #                     options = json.loads(options_raw)
    #                 except json.decoder.JSONDecodeError as e:
    #                     log.error(f"bad options of function '{function.name}', options: '{options_raw}': {e}")
    #                     options = json_repair.loads(options_raw)
    #                     log.info(f"repaired options '{options}'")
    #             elif isinstance(options_raw, list):
    #                 options = options_raw
    #             else:
    #                 log.error(f"unexpected options type, function '{function.name}', args '{args_raw}', "
    #                           f"options type {type(options_raw)}")
    #         else:
    #             log.error(f"missing options in args, function '{function.name}', args '{args_raw}'")
    #
    #         if options:
    #             args["options"] = options # json.dumps(options, ensure_ascii=False)
    #
    #         function.arguments = json.dumps(args, ensure_ascii=False)
    #         log.info(f"function after repairing, function {function.name}, arguments '{args}'")

    return tool_call


def fix_tool_definition_optional_property_as_null_type(tool: ToolDefinition) -> ToolDefinition:
    function = tool.function
    function.parameters = _fix_tool_definition_optional_property_as_null_type(function.parameters, function.name)
    return tool


def _fix_tool_definition_optional_property_as_null_type(parameters: dict[str, Any], parent_name: str) -> dict[str, Any]:
    properties = parameters.get("properties", {})
    required: list | None = parameters.get("required")
    for prop_name, prop_params in properties.items():
        params: dict[str, Any] = prop_params
        type = params.get("type")
        if isinstance(type, list):
            if len(type) >= 1:
                opt = False
                for i in range(1, len(type)):
                    if type[i] == "null":
                        opt = True
                        break

                new_type = type[0]
                params["type"] = new_type

                if opt and required:
                    required.remove(prop_name)
                    parameters["required"] = required

                log.debug(
                    f"fix parameter type: parent object '{parent_name}', property '{prop_name}',"
                    f" new type '{new_type}', old type '{type}', optional {opt}")

                if type == "object":
                    sub_properties = params.get("properties")
                    if isinstance(sub_properties, dict):
                        params["properties"] = _fix_tool_definition_optional_property_as_null_type(sub_properties,
                                                                                                   prop_name)

    return properties
