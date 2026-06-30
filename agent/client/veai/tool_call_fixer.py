import json
import logging
from typing import Any

import json_repair

from agent.client.veai.tool import edit_file, read_file, write_file, search_for_text, ask_user_with_options, list_dir, \
    search_file_by_name, file_structure
from agent.client.veai.tool.edit_file import EditFile
from agent.client.veai.tool.file_structure import FileStructure
from agent.client.veai.tool.list_dir import ListDir
from agent.client.veai.tool.read_file import ReadFile
from agent.client.veai.tool.search_file_by_name import SearchFileByName
from agent.client.veai.tool.search_for_text import SearchForText
from agent.client.veai.tool.write_file import WriteFile
from agent.openai.chat_completions_api import ToolDefinition
from agent.parser import ParsedFunctionCall

ROOT = "."

log = logging.getLogger(__name__)


def veai_fix_incorrect_arguments(function: ParsedFunctionCall) -> ParsedFunctionCall:
    if "run_command" == function.name:
        pass
    elif list_dir.function_name == function.name:
        return fix_list_dir(function)
    elif file_structure.function_name == function.name:
        return fix_file_structure(function)
    elif edit_file.function_name == function.name:
        return fix_edit_file(function)
    elif write_file.function_name == function.name:
        return fix_write_file(function)
    elif read_file.function_name == function.name:
        return fix_read_file(function)
    elif search_for_text.function_name == function.name:
        return fix_search_for_text(function)
    elif search_file_by_name.function_name == function.name:
        return fix_search_file_by_name(function)
    elif ask_user_with_options.function_name == function.name:
        return fix_ask_user_with_options(function)
    return function


def fix_ask_user_with_options(function: ParsedFunctionCall) -> ParsedFunctionCall:
    args = get_args(function)
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
            log.error(f"unexpected options type, function '{function.name}', args '{args}', "
                      f"options type {type(options_raw)}")
    else:
        log.error(f"missing options in args, function '{function.name}', args '{args}'")

    if options:
        args["options"] = options  # json.dumps(options, ensure_ascii=False)

    function.arguments = json.dumps(args, ensure_ascii=False)
    log.info(f"function after repairing, function {function.name}, arguments '{args}'")
    return function


def fix_file_structure(function: ParsedFunctionCall) -> ParsedFunctionCall:
    args = get_args(function)
    target_file, invalid = get_target_file(args)
    if invalid:
        new_function = FileStructure().new_call(target_file)
        return new_function
    return function


def fix_edit_file(function: ParsedFunctionCall) -> ParsedFunctionCall:
    args = get_args(function)
    target_file, invalid = get_target_file(args)
    if not target_file:
        log.warning(f"tool call error: tool={function.name}, target_file is empty but required")
    edits_raw = args.get("edits")
    if target_file and edits_raw:
        allow_multiple_matches = as_bool_or_none(args.get("allow_multiple_matches"), "allow_multiple_matches")
        invalid = False
        if not allow_multiple_matches:
            invalid = True
            allow_multiple_matches = True
        # qwen2 case
        if isinstance(edits_raw, list):
            edits_list = edits_raw
            edits_raw = edits_list[0] if edits_list else ""
            log.debug(f"convert 'edits' list to str: list={edits_list}, str={edits_raw}")

        if not isinstance(edits_raw, str):
            log.error(
                f"unexpected edits type, function '{function.name}', args '{edits_raw}', type {type(edits_raw)}")
        try:
            edits = json.loads(edits_raw)
        except json.decoder.JSONDecodeError as e:
            invalid = True
            log.info(f"bad edits of function '{function.name}', options: '{edits_raw}': {e}")
            edits = json_repair.loads(edits_raw)
            edits_raw = json.dumps(edits)
            log.info(f"repaired edits '{edits_raw}'")
        if invalid:
            new_function = EditFile().new_call(target_file, edits, allow_multiple_matches=allow_multiple_matches)
            return new_function

    return function


def fix_write_file(function: ParsedFunctionCall) -> ParsedFunctionCall:
    args = get_args(function)
    target_file, invalid = get_target_file(args)
    content = args.get("content")
    if target_file and content:
        allow_overwrite: bool = args.get("allow_overwrite")

        if not allow_overwrite:
            invalid = True
            allow_overwrite = True

        if invalid:
            new_function = WriteFile().new_call(target_file, content, allow_overwrite=allow_overwrite)
            return new_function
    else:
        log.error(f"no required args for function {function.name}, args={args}, "
                  f"required args = ['target_file', 'content']")

    return function


def fix_search_for_text(function: ParsedFunctionCall) -> ParsedFunctionCall:
    args = get_args(function)
    target_path_or_url = args.get("target_path_or_url")
    text_snippet = args.get("text_snippet")
    if target_path_or_url and text_snippet:
        is_case_sensitive = as_bool_or_none(args.get("is_case_sensitive"), "is_case_sensitive")
        if is_case_sensitive is None:
            # log
            new_function = SearchForText().new_call(target_path_or_url, text_snippet, True)
            return new_function

    return function


def fix_search_file_by_name(function: ParsedFunctionCall) -> ParsedFunctionCall:
    args = get_args(function)
    glob_pattern = args.get("glob_pattern")
    invalid = not glob_pattern
    if invalid:
        # gemma 4
        glob_pattern = args.get("glob")

    invalid = not glob_pattern
    if invalid:
        # gemma 4
        glob_pattern = args.get("pattern")

    invalid = not glob_pattern
    if invalid:
        # gemma 4
        glob_pattern = args.get("query")

    search_directory = args.get("search_directory")
    if not search_directory:
        invalid = True
        search_directory = ROOT

    if invalid:
        log.info(
            f"fix invalid {function.name}: glob_pattern={glob_pattern}, search_directory={search_directory}")
        new_function = SearchFileByName().new_call(glob_pattern, search_directory)
        return new_function
    return function


def fix_read_file(function: ParsedFunctionCall) -> ParsedFunctionCall:
    args = get_args(function)
    target_file, invalid = get_target_file(args)

    anonymous_arguments = function.anonymous_arguments
    if not target_file and anonymous_arguments:
        invalid = True
        target_file = anonymous_arguments[0]

    if not target_file:
        log.error(f"no target file for function '{function.name}'")
    else:
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
                f"fix invalid {function.name}: target_file={target_file}, start_line={start_line}, "
                f"end_line={end_line}")
            new_function = ReadFile().new_call(target_file=target_file, start_line=start_line, end_line=end_line,
                                               line_offset=line_offset)
            return new_function

    return function


def get_target_file(args) -> tuple[str, bool]:
    target_file = args.get("target_file")

    invalid = not target_file
    if invalid:
        # gemma4 case
        target_file = args.get("file_path")

        # gemma4 case 2
        if not target_file:
            invalid = True
            target_file = args.get("file")

        # gemma4 case 3
        if not target_file:
            invalid = True
            target_file = args.get("path")
    return target_file, invalid


def fix_list_dir(function: ParsedFunctionCall) -> ParsedFunctionCall:
    args = get_args(function)
    directory_path = args.get("directory_path")
    invalid = False
    if not directory_path:
        invalid = True
        # gemma4 case
        directory_path = args.get("dir")

    if not directory_path:
        invalid = True
        root = True
        directory_path = ROOT
    else:
        root = False

    depth = args.get("depth")
    if not depth:
        invalid = True
        depth = 5 if root else 2

    if invalid:
        log.info(
            f"fix invalid {function.name}: directory_path={directory_path}, depth={depth}")
        new_function = ListDir().new_call(directory_path=directory_path, depth=depth)
        return new_function

    return function


def get_args(function: ParsedFunctionCall) -> dict[str, Any]:
    return function.arguments or {}


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


def read_args_as_json(args: dict[str, Any]) -> Any:
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
