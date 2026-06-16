import json
import logging
from typing import Any

import json_repair

from common.openai_model import ToolCall, ToolDefinition, FunctionCall
from veai.tool import edit_file, read_file, write_file, search_for_text
from veai.tool.edit_file import EditFile
from veai.tool.read_file import ReadFile
from veai.tool.search_for_text import SearchForText
from veai.tool.write_file import WriteFile

log = logging.getLogger(__name__)


def fix_incorrect_arguments(tool_call: ToolCall) -> ToolCall:
    if edit_file.function_name == tool_call.function.name:
        return fix_edit_file(tool_call)
    elif write_file.function_name == tool_call.function.name:
        return fix_write_file(tool_call)
    elif read_file.function_name == tool_call.function.name:
        return fix_read_file(tool_call)
    elif search_for_text.function_name == tool_call.function.name:
        return fix_search_for_text(tool_call)
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


def fix_edit_file(tool_call: ToolCall) -> ToolCall:
    args_raw = tool_call.function.arguments
    args = rea_args_as_json(args_raw, tool_call.function)
    if args:
        target_file = args.get("target_file")
        edits = args.get("edits")
        if target_file and edits:
            allow_multiple_matches: bool = args.get("allow_multiple_matches")
            if not allow_multiple_matches:
                # invalid
                # log
                new_function = EditFile().new_call(target_file, edits)
                tool_call.function = new_function

    return tool_call


def fix_write_file(tool_call: ToolCall) -> ToolCall:
    args_raw = tool_call.function.arguments
    args = rea_args_as_json(args_raw, tool_call.function)
    if args:
        target_file = args.get("target_file")
        content = args.get("content")
        if target_file and content:
            allow_overwrite: bool = args.get("allow_overwrite")

            if not allow_overwrite:
                # invalid
                # log
                new_function = WriteFile().new_call(target_file, content)
                tool_call.function = new_function

    return tool_call

def fix_search_for_text(tool_call: ToolCall) -> ToolCall:
    args_raw = tool_call.function.arguments
    args = rea_args_as_json(args_raw, tool_call.function)
    if args:
        target_path_or_url = args.get("target_path_or_url")
        text_snippet = args.get("text_snippet")
        if target_path_or_url and text_snippet:
            is_case_sensitive = args.get("is_case_sensitive")
            if  is_case_sensitive is None:
                # log
                new_function = SearchForText().new_call(target_path_or_url, text_snippet, is_case_sensitive)
                tool_call.function = new_function

    return tool_call


def fix_read_file(tool_call: ToolCall) -> ToolCall:
    args_raw = tool_call.function.arguments
    args = rea_args_as_json(args_raw, tool_call.function)
    if args:
        target_file = args.get("target_file")
        if target_file:
            start_line = args.get("start_line")
            end_line = args.get("end_line")
            line_offset = args.get("line_offset")

            valid = isinstance(line_offset, int) or (isinstance(start_line, int) and isinstance(end_line, int))
            if not valid:
                # log
                new_function = ReadFile().new_call(target_file)
                tool_call.function = new_function

    return tool_call


def rea_args_as_json(args_raw: str, function: FunctionCall) -> Any:
    try:
        args = json.loads(args_raw)
    except json.decoder.JSONDecodeError as e:
        log.error(f"bad arguments of function '{function.name}', args '{args_raw}': {e}")
        args = json_repair.loads(args_raw)
        log.info(f"repaired arguments '{args}'")
    return args


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
