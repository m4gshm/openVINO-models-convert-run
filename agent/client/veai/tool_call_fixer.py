import json
import logging
from typing import Any

import json_repair

from agent.client.veai.tool import edit_file, read_file, write_file, search_for_text, ask_user_with_options
from agent.client.veai.tool.edit_file import EditFile
from agent.client.veai.tool.read_file import ReadFile
from agent.client.veai.tool.search_for_text import SearchForText
from agent.client.veai.tool.write_file import WriteFile
from agent.openai.chat_completions_api import ToolCall, ToolDefinition, FunctionCall

log = logging.getLogger(__name__)


def veai_fix_incorrect_arguments(tool_call: ToolCall) -> ToolCall:
    function = tool_call.function
    if "run_command" == function.name:
        pass
    elif edit_file.function_name == function.name:
        return fix_edit_file(tool_call)
    elif write_file.function_name == function.name:
        return fix_write_file(tool_call)
    elif read_file.function_name == function.name:
        return fix_read_file(tool_call)
    elif search_for_text.function_name == function.name:
        return fix_search_for_text(tool_call)
    if ask_user_with_options.function_name == function.name:
        return fix_ask_user_with_options(tool_call)

    return tool_call


def fix_ask_user_with_options(tool_call: ToolCall) -> ToolCall:
    function = tool_call.function
    args_raw = function.arguments
    args = read_args_as_json(args_raw, function)
    if args:
        options_raw = args.get("options")
        is_multiple_choice = as_bool_or_none(args.get("is_multiple_choice"), "is_multiple_choice")
        if not is_multiple_choice:
            is_multiple_choice = False
            args["is_multiple_choice"] = is_multiple_choice
        question = args.get("question")
        if not question:
            args["question"] = "[*]" if is_multiple_choice else "(*)"
        options: Any = None
        if options_raw:
            if isinstance(options_raw, str):
                try:
                    options = json.loads(options_raw)
                except json.decoder.JSONDecodeError as e:
                    log.error(f"bad options of function '{function.name}', options: '{options_raw}': {e}")
                    options = json_repair.loads(options_raw)
                    log.info(f"repaired options '{options}'")
            elif isinstance(options_raw, list):
                options = options_raw
            else:
                log.error(f"unexpected options type, function '{function.name}', args '{args_raw}', "
                          f"options type {type(options_raw)}")
        else:
            log.error(f"missing options in args, function '{function.name}', args '{args_raw}'")

        if options:
            args["options"] = options  # json.dumps(options, ensure_ascii=False)

        function.arguments = json.dumps(args, ensure_ascii=False)
        log.info(f"function after repairing, function {function.name}, arguments '{args}'")
    return tool_call


def fix_edit_file(tool_call: ToolCall) -> ToolCall:
    function = tool_call.function
    args_raw = function.arguments
    args = read_args_as_json(args_raw, function)
    if args:
        target_file = args.get("target_file")
        if not target_file:
            log.warning(f"tool call error: tool={function.name}, target_file is empty but required")
        edits_str = args.get("edits")
        if target_file and edits_str:
            allow_multiple_matches = as_bool_or_none(args.get("allow_multiple_matches"), "allow_multiple_matches")
            invalid = False
            if not allow_multiple_matches:
                invalid = True
                allow_multiple_matches = True
            try:
                edits = json.loads(edits_str)
            except json.decoder.JSONDecodeError as e:
                invalid = True
                log.info(f"bad edits of function '{function.name}', options: '{edits_str}': {e}")
                edits = json_repair.loads(edits_str)
                edits_str = json.dumps(edits)
                log.info(f"repaired edits '{edits_str}'")
            if invalid:
                new_function = EditFile().new_call(target_file, edits, allow_multiple_matches=allow_multiple_matches)
                tool_call.function = new_function

    return tool_call


def fix_write_file(tool_call: ToolCall) -> ToolCall:
    args_raw = tool_call.function.arguments
    args = read_args_as_json(args_raw, tool_call.function)
    if args:
        target_file = args.get("target_file")
        content = args.get("content")
        if target_file and content:
            allow_overwrite: bool = args.get("allow_overwrite")

            if not allow_overwrite:
                # invalid
                # log
                new_function = WriteFile().new_call(target_file, content, allow_overwrite = True)
                tool_call.function = new_function

    return tool_call


def fix_search_for_text(tool_call: ToolCall) -> ToolCall:
    args_raw = tool_call.function.arguments
    args = read_args_as_json(args_raw, tool_call.function)
    if args:
        target_path_or_url = args.get("target_path_or_url")
        text_snippet = args.get("text_snippet")
        if target_path_or_url and text_snippet:
            is_case_sensitive = as_bool_or_none(args.get("is_case_sensitive"), "is_case_sensitive")
            if is_case_sensitive is None:
                # log
                new_function = SearchForText().new_call(target_path_or_url, text_snippet, True)
                tool_call.function = new_function

    return tool_call


def fix_read_file(tool_call: ToolCall) -> ToolCall:
    args_raw = tool_call.function.arguments
    args = read_args_as_json(args_raw, tool_call.function)
    if args:
        target_file = args.get("target_file")

        invalid = not target_file
        if invalid:
            # gemma4 case
            target_file = args.get("file_path")

            invalid = not target_file
            # gemma4 case 2
            if invalid:
                target_file = args.get("file")

            invalid = not target_file
            # gemma4 case 3
            if invalid:
                target_file = args.get("path")

        if target_file:
            start_line = as_int_or_none(args.get("start_line"), "start_line")
            end_line = as_int_or_none(args.get("end_line"), "end_line")
            line_offset = as_int_or_none(args.get("line_offset"), "line_offset")

            if line_offset:
                pass
            else:
                if not start_line:
                    invalid = True
                    start_line = 1
                if not end_line:
                    invalid = True
                    end_line = 500
            if invalid:
                log.info(
                    f"fix invalid read_file: target_file={target_file}, start_line={start_line}, end_line={end_line}")
                new_function = ReadFile().new_call(target_file=target_file, start_line=start_line, end_line=end_line,
                                                   line_offset=line_offset)
                tool_call.function = new_function

    return tool_call


def as_int_or_none(val, name: str) -> int | None:
    return as_type_or_none(int, val, name)


def as_bool_or_none(val, name: str) -> bool | None:
    return as_type_or_none(bool, val, name)


def as_type_or_none[T](t: type[T], val, name: str) -> T | None:
    if val and not isinstance(val, t):
        try:
            return t(val)
        except ValueError:
            log.info(f"{name} is not an {t}: '{val}', '{t(val)}'")
    return None


def read_args_as_json(args_raw: str, function: FunctionCall) -> Any:
    try:
        args = json.loads(args_raw)
    except json.decoder.JSONDecodeError as e:
        log.error(f"bad arguments of function '{function.name}', args '{args_raw}': {e}")
        args = json_repair.loads(args_raw)
        log.info(f"repaired arguments '{args}'")
    return args


def veai_fix_tool_definition_optional_property_as_null_type(tool: ToolDefinition) -> ToolDefinition:
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
