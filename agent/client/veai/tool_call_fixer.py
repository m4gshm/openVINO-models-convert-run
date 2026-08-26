import json
import logging
import re
from json import JSONDecodeError
from typing import Any, Iterable

import json_repair

from agent.client.user_context import UserContext
from agent.client.veai.tool import edit_file, read_file, write_file, search_for_text, ask_user_with_options, list_dir, \
    search_file_by_name, file_structure, run_command, run_configuration
from agent.client.veai.tool.edit_file import EditFile
from agent.client.veai.tool.file_structure import FileStructure
from agent.client.veai.tool.list_dir import ListDir
from agent.client.veai.tool.read_file import ReadFile
from agent.client.veai.tool.run_command import RunCommand
from agent.client.veai.tool.run_configuration import RunConfiguration
from agent.client.veai.tool.search_file_by_name import SearchFileByName
from agent.client.veai.tool.search_for_text import SearchForText
from agent.client.veai.tool.write_file import WriteFile
from agent.openai.chat_api import ROLE_ASSISTANT
from agent.openai.chat_completions_api import ChatCompletionFunctionToolParam
from agent.parser import ParsedFunctionCall

GEMMA_4 = "Gemma4ForConditionalGeneration"

TARGET_FILE = "target_file"

ROOT = "."

log = logging.getLogger(__name__)


def veai_fix_incorrect_arguments(function: ParsedFunctionCall,
                                 user_context: UserContext | None = UserContext()) -> list[
                                                                                          ParsedFunctionCall] | ParsedFunctionCall:
    if run_command.function_name == function.name:
        return fix_run_command(function, user_context)
    elif list_dir.function_name == function.name:
        return fix_list_dir(function, user_context)
    elif file_structure.function_name == function.name:
        return fix_file_structure(function, user_context)
    elif edit_file.function_name == function.name:
        return fix_edit_file(function, user_context)
    elif write_file.function_name == function.name:
        return fix_write_file(function, user_context)
    elif read_file.function_name == function.name:
        return fix_read_file(function, user_context)
    elif search_for_text.function_name == function.name:
        return fix_search_for_text(function, user_context)
    elif search_file_by_name.function_name == function.name:
        return fix_search_file_by_name(function, user_context)
    elif ask_user_with_options.function_name == function.name:
        return fix_ask_user_with_options(function, user_context)
    elif run_configuration.function_name == function.name:
        return fix_run_configuration(function, user_context)
    return function


def fix_ask_user_with_options(function: ParsedFunctionCall, context: UserContext = UserContext()) -> ParsedFunctionCall:
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


def fix_file_structure(function: ParsedFunctionCall, context: UserContext = UserContext()) -> ParsedFunctionCall:
    args = get_args(function)
    target_file, invalid = get_target_file(args, context)
    target_file, found = try_find_target_file_from_prev_tool_call_if_need(args, context, function, target_file)
    if found:
        invalid = True

    if not target_file:
        log.error(f"tool call error: tool={function.name}, target_file is empty but required")

    if invalid:
        new_function = FileStructure().new_call(target_file)
        return new_function
    return function


def clean_file_path(target_file):
    if not target_file:
        return target_file

    cleaned_target_file = ""
    for i, s in enumerate(target_file):
        ignore = s == "}" or s == "]"
        if not ignore and (s == "\"" or s == "'"):
            prev = target_file[i - 1] if i > 0 else None
            if prev != "\\":
                ignore = True

        if not ignore:
            cleaned_target_file += s
    log.debug(f"clean_file_path: path={target_file}, cleaned={cleaned_target_file}")
    return cleaned_target_file


def fix_edit_file(function: ParsedFunctionCall, context: UserContext = UserContext()) -> list[
                                                                                             ParsedFunctionCall] | ParsedFunctionCall:
    args = get_args(function)
    target_file, invalid = get_target_file(args, context)
    edits = args.get("edits")
    unused_anonymous_edits = []
    if edits:
        unused_anonymous_edits = handle_edits(edits)

    # recheck edits fullness:
    global_new_text = args.get("new_text")
    if global_new_text:
        if not "new_text" in edits:
            if not edits:
                edits = {}
            edits["new_text"] = global_new_text
        else:
            is_set_new_text = False
            for edit in edits:
                if "new_text" not in edit and "old_text" in edit:
                    is_set_new_text = True
                    edit["new_text"] = global_new_text
            if is_set_new_text:
                del args["new_text"]

    global_old_text = args.get("old_text")
    if global_old_text:
        if not "old_text" in edits:
            if not edits:
                edits = {}
            edits["old_text"] = global_old_text
        else:
            is_set_old_text = False
            for edit in edits:
                if "old_text" not in edit and "new_text" in edit:
                    is_set_old_text = True
                    edit["old_text"] = global_old_text
            if is_set_old_text:
                del args["old_text"]

    if not target_file and edits:
        # gemma 4
        # try to search in edits
        for i, edit in enumerate(edits):
            if isinstance(edit, dict):
                target_file = edit.get("target_file")
                if target_file:
                    del edit["target_file"]
                    if len(edit) == 0:
                        del edits[i]
                    break
    if not target_file and unused_anonymous_edits:
        expect_target_file = False
        for v in unused_anonymous_edits:
            if expect_target_file:
                target_file = v
                break
            elif "target_file" == v:
                expect_target_file = True
            elif "target_file" in v:
                pass
    if not target_file:
        target_file = args.get("file_path")
        if target_file:
            del args["file_path"]

    if not edits:
        edits = {}
        # lfm 2.5 case
        old_text = args.get("old_text")
        if old_text:
            edits["old_text"] = old_text
            del args["old_text"]
        new_text = args.get("new_text")
        if new_text:
            edits["new_text"] = new_text
            del args["new_text"]
        else:
            content = args.get("content")
            if content:
                edits["new_text"] = content
                del args["content"]

    if target_file or edits:
        allow_multiple_matches = as_bool_or_none(args.get("allow_multiple_matches"), "allow_multiple_matches")
        if allow_multiple_matches is None:
            allow_multiple_matches = as_bool_or_none(args.get("allowed_multiple_matches"), "allowed_multiple_matches")
            if not allow_multiple_matches is None:
                invalid = True
                del args["allowed_multiple_matches"]

        if allow_multiple_matches is None:
            invalid = True
            allow_multiple_matches = False
        # qwen3.5 case
        if isinstance(edits, str):
            log.debug(f"convert string edits to json object, function='{function.name}', edist='{edits}'")
            try:
                edits = json.loads(edits)
            except json.decoder.JSONDecodeError as e:
                log.info(f"bad json edits of function='{function.name}', edits='{edits}': {e}")
                edits = json_repair.loads(str(edits))
                log.info(f"repaired edits type='{type(edits)}', payload='{json.dumps(edits)}'")

        # qwen2 case
        if isinstance(edits, list):
            for i, edit in enumerate(edits):
                if isinstance(edit, list):
                    invalid = True
                    edit = edit[0] if edit else None
                elif isinstance(edit, dict):
                    # valid
                    pass
                else:
                    edit_str: str | None = None
                    if isinstance(edits, bytes):
                        edit_str = bytearray(edits).decode()
                    elif isinstance(edits, bytearray):
                        edit_str = edits.decode()
                    elif isinstance(edits, str):
                        edit_str = edits
                    invalid = True
                    if edit_str is None:
                        edit = edits
                        log.error(
                            f"unexpected edits element type, function='{function.name}', element_{i}='{edit}', type {type(edit)}")
                    else:
                        try:
                            edit = json.loads(edit_str)
                        except json.decoder.JSONDecodeError as e:
                            log.info(f"bad edits of function='{function.name}', element_{i}='{edit_str}': {e}")
                            edit = json_repair.loads(str(edit_str))
                            log.info(f"repaired element_{i}='{json.dumps(edit)}'")

                edits[i] = edit

            if len(edits) > 1:
                # gemma 4
                # try to merge
                set_prev = False
                merged = False
                prev_old_text = None
                prev_new_text = None
                for edit in edits:
                    if isinstance(edit, dict):
                        old_text = edit.get("old_text")
                        new_text = edit.get("new_text")
                        if prev_old_text is None and prev_new_text is None:
                            set_prev = True
                            prev_old_text = old_text
                            prev_new_text = new_text
                        else:
                            if prev_old_text and old_text and isinstance(old_text, str) and isinstance(prev_old_text,
                                                                                                       str):
                                in_prev = old_text.startswith(prev_old_text)
                                if in_prev:
                                    log.debug(
                                        f"merge in pre, new_text '{prev_new_text}' with '{new_text}' for old_text '{prev_old_text}' and '{old_text}'")
                                    prev_new_text = prev_new_text + new_text
                                    prev_old_text = old_text
                                    merged = True
                    elif isinstance(edit, list):
                        log.error(f"unexpected edit type in edits: type={type(edit)} edit={edit}, edits={edits}")
                    else:
                        log.error(f"unexpected edit type in edits: type={type(edit)} edit={edit}, edits={edits}")

                if prev_old_text and prev_new_text:
                    edits = [{"new_text": prev_new_text, "old_text": prev_old_text}]
                    log.debug(f"merged edits={edits}")
                    invalid = True

        target_file, _ = try_find_target_file_from_prev_tool_call_if_need(args, context, function, target_file)
        target_file = clean_file_path(target_file)
        if not target_file:
            log.error(f"tool call error: tool={function.name}, target_file is empty but required")

        return EditFile().new_call(target_file, edits, allow_multiple_matches=allow_multiple_matches)
    else:
        return function


def try_find_target_file_from_prev_tool_call_if_need(args: dict[str, Any], context: UserContext,
                                                     function: ParsedFunctionCall,
                                                     target_file: str | Any) -> Any | None:
    if not target_file and (context and GEMMA_4 in context.model_architectures):
        prev_target_file = find_target_file_from_prev_tool_call(args, context, function.name, target_file)
        return prev_target_file, not prev_target_file is None
    return target_file, None


def handle_edits(edits: Any):
    new_text = None
    new_text_i = None
    old_text = None
    old_text_i = None
    on_delete_i = []
    anonymous_edits = []
    if isinstance(edits, dict):
        edits = [edits]
    elif not isinstance(edits, Iterable):
        log.warning(f"unexpected edits type='{type(edits)}', edits='{edits}'")
    for i, edit in enumerate(edits):
        if isinstance(edit, dict):
            if len(edit) == 0:
                on_delete_i.append(i)
            else:
                edit_new_text = edit.get("new_text")
                if edit_new_text is None:
                    edit_new_text = edit.get("new_config")
                    if edit_new_text:
                        edit["new_text"] = edit_new_text
                        del edit["new_config"]

                edit_old_text = edit.get("old_text")
                if edit_old_text is None:
                    edit_old_text = edit.get("old_config")
                    if edit_old_text:
                        edit["old_text"] = edit_old_text
                        del edit["old_config"]
                if not new_text_i:
                    if edit_new_text:
                        new_text = edit_new_text
                        new_text_i = i
                if not old_text_i:
                    if edit_old_text:
                        old_text = edit_old_text
                        old_text_i = i

                if old_text_i == new_text_i:
                    # expected
                    old_text_i = None
                    new_text_i = None
                elif old_text_i is not None and new_text_i is not None:
                    # merge
                    edits[new_text_i]["old_text"] = old_text
                    del edits[old_text_i]["old_text"]
                    if len(edits[old_text_i]) == 0:
                        on_delete_i.append(old_text_i)
                    old_text_i = None
                    new_text_i = None
        elif isinstance(edit, list) or isinstance(edit, set):
            first = next(iter(edit)) if len(edit) > 0 else None
            second = next(iter(edit)) if len(edit) > 1 else None
            if new_text is None and old_text:
                new_text = first
                edits[old_text_i]["new_text"] = new_text
                del edit[0]
            elif old_text is None and new_text:
                old_text = first
                edits[new_text_i]["old_text"] = old_text
                del edit[0]
                # if len(edit) == 0:
                #     on_delete_i.append(i)
            else:
                anonymous_edits.extend(edit)
            on_delete_i.append(i)
        else:
            anonymous_edits.append(edit)
            on_delete_i.append(i)

    for i in reversed(on_delete_i):
        del edits[i]

    unused_anonymous_edits = []
    if anonymous_edits:
        expect_next_next = False
        expect_old_text = False
        dirty_edits = []
        edit = {}
        for v in anonymous_edits:
            if expect_next_next:
                edit["new_text"] = v
                expect_next_next = False
            elif expect_old_text:
                edit["old_text"] = v
                expect_old_text = False
            elif v == "new_text":
                if "new_text" in edit:
                    dirty_edits.append(edit)
                    edit = {}
                expect_next_next = True
            elif v == "old_text":
                if "old_text" in edit:
                    dirty_edits.append(edit)
                    edit = {}
                expect_old_text = True
            else:
                unused_anonymous_edits.append(v)

        if edit:
            dirty_edits.append(edit)

        if isinstance(edits, list):
            edits.extend(dirty_edits)
            unused_anonymous_edits.extend(handle_edits(edits))
    return unused_anonymous_edits


def fix_write_file(function: ParsedFunctionCall, context: UserContext = UserContext()) -> ParsedFunctionCall:
    args = get_args(function)
    target_file, invalid = get_target_file(args, context)
    content = args.get("content")

    # target_file, found = try_find_target_file_from_prev_tool_call_if_need(args, context, function, target_file)
    # if found:
    #     invalid |= found

    if target_file and content:
        allow_overwrite = args.get("allow_overwrite")

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


def fix_search_for_text(function: ParsedFunctionCall, context: UserContext = UserContext()) -> ParsedFunctionCall:
    args = get_args(function)
    target_path_or_url = args.get("target_path_or_url")
    text_snippet = args.get("text_snippet")
    if target_path_or_url and text_snippet:
        target_path_or_url, fixed = fix_windows_path(target_path_or_url, context)
        is_case_sensitive = as_bool_or_none(args.get("is_case_sensitive"), "is_case_sensitive")
        if is_case_sensitive is None:
            # log
            new_function = SearchForText().new_call(target_path_or_url, text_snippet, True)
            return new_function

    return function


def fix_search_file_by_name(function: ParsedFunctionCall, context: UserContext = UserContext()) -> ParsedFunctionCall:
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
    else:
        search_directory, fixed = fix_windows_path(search_directory, context)
        if fixed:
            invalid = True

    if invalid:
        log.info(
            f"fix invalid {function.name}: glob_pattern={glob_pattern}, search_directory={search_directory}")
        new_function = SearchFileByName().new_call(glob_pattern, search_directory)
        return new_function
    return function


def fix_read_file(function: ParsedFunctionCall, context: UserContext = UserContext()) -> ParsedFunctionCall:
    args = get_args(function)
    target_file, invalid = get_target_file(args, context)

    anonymous_arguments = function.anonymous_arguments
    if not target_file and anonymous_arguments:
        invalid = True
        target_file = anonymous_arguments[0]

    target_file, found = try_find_target_file_from_prev_tool_call_if_need(args, context, function, target_file)
    if found:
        invalid = True

    if not target_file:
        log.error(f"tool call error: tool={function.name}, target_file is empty but required")
    else:
        start_line, fixed = as_int_or_none(args.get("start_line"), "start_line")
        invalid |= fixed
        end_line, fixed = as_int_or_none(args.get("end_line"), "end_line")
        invalid |= fixed
        line_offset, fixed = as_int_or_none(args.get("line_offset"), "line_offset")
        invalid |= fixed

        if line_offset and (start_line or end_line):
            invalid = True
            line_offset = None

        if not line_offset:
            if not start_line:
                invalid = True
                start_line = 1
            if not end_line:
                invalid = True
                end_line = 500
        if invalid:
            log.info(
                f"fix invalid {function.name}: target_file={target_file}, start_line={start_line}, "
                f"end_line={end_line}, line_offset={line_offset}")
            new_function = ReadFile().new_call(target_file=target_file, start_line=start_line, end_line=end_line,
                                               line_offset=line_offset)
            return new_function

    return function


def get_target_file(args, context: UserContext = UserContext()) -> tuple[str, bool]:
    target_file = args.get(TARGET_FILE)

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

    target_file, fixed = fix_windows_path(target_file, context)
    if fixed:
        invalid = True

    return target_file, invalid


def find_target_file_from_prev_tool_call(args, context: UserContext, function_name: str,
                                         target_file: Any | None) -> Any | None:
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"no required target_file, trying to get from previous cool call of '{function_name}', args='{args}'")
    else:
        log.info(f"no required target_file, trying to get from previous cool call of '{function_name}'")

    messages = context.messages
    for i, message in enumerate(reversed(messages)):
        if message.role == ROLE_ASSISTANT:
            for tool_call in message.tool_calls or []:
                function = tool_call.function
                if function.name == function_name:
                    try:
                        arguments = json.loads(function.arguments)
                    except JSONDecodeError as e:
                        log.debug(f"error on function arguments parsing: tool_call.id={tool_call.id}, "
                                  f"function.name={function.name}, arguments='{function.argumentsl}'")
                        arguments = {}

                    target_file = arguments.get(TARGET_FILE)
                    if target_file:
                        log.info(f"gets target file from previous tool call: tool_call.id={tool_call.id}, "
                                 f"function.name={function.name}, target_file='{target_file}'")
    return target_file


def fix_list_dir(function: ParsedFunctionCall, context: UserContext = UserContext()) -> ParsedFunctionCall:
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

    directory_path, fixed = fix_windows_path(directory_path, context)
    if fixed:
        invalid = True

    if invalid:
        log.info(
            f"fix invalid {function.name}: directory_path={directory_path}, depth={depth}")
        new_function = ListDir().new_call(directory_path=directory_path, depth=depth)
        return new_function
    else:
        return function


def is_windows(context: UserContext | None):
    return "windows" in context.os.lower() if context and context.os else False


def fix_run_command(function: ParsedFunctionCall, context: UserContext = UserContext()) -> ParsedFunctionCall:
    args = get_args(function)

    working_directory = args.get("working_directory")
    command = args.get("command")
    is_background = args.get("is_background")
    invalid = False
    if not is_background:
        invalid = True
        is_background = False
    safe_to_run = args.get("safe_to_run")
    if not safe_to_run:
        invalid = True
        safe_to_run = False

    working_directory, fixed = fix_windows_path(working_directory, context)
    if fixed:
        invalid = True

    if invalid:
        log.info(
            f"fix invalid {function.name}: command={command}, working_directory={working_directory}, "
            f"is_background={is_background}, safe_to_run={safe_to_run}")
        new_function = RunCommand().new_call(command=command, working_directory=working_directory,
                                             is_background=is_background, safe_to_run=safe_to_run)
        return new_function
    else:
        return function


def fix_windows_path(path: Any | None, context: UserContext = UserContext()) -> tuple[Any, bool]:
    fixed = False
    if path and isinstance(path, str) and is_windows(context):
        # SERA case
        if path.startswith("/"):
            fixed = True
            path = path[1:]
    return path, fixed


def get_args(function: ParsedFunctionCall) -> dict[str, Any]:
    return function.arguments or {}


def as_int_or_none(val, name: str) -> tuple[int | None, bool]:
    result = as_type_or_none(int, val, name)
    if result is not None:
        return result, False
    if isinstance(val, str):
        match = re.search(r'-\d+|\d+', val)
        return int(match.group()) if match else None, True
    else:
        return None, False


def as_bool_or_none(val, name: str) -> bool | None:
    if isinstance(val, bool):
        return val
    elif isinstance(val, str):
        lower = val.lower()
        if lower == "true":
            return True
        else:
            return False
    return as_type_or_none(bool, val, name)


def as_type_or_none[T](t: type[T], val, name: str) -> T | None:
    if not val is None and not isinstance(val, t):
        try:
            return t(val)
        except ValueError:
            log.warning(f"{name} is not an {t}: '{val}'")
    return None


def read_args_as_json(args: dict[str, Any]) -> Any:
    return args


def veai_fix_tool_definition_optional_property_as_null_type(
        tool: ChatCompletionFunctionToolParam) -> ChatCompletionFunctionToolParam:
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


def fix_run_configuration(function: ParsedFunctionCall, context: UserContext = UserContext()) -> ParsedFunctionCall:
    args = get_args(function)
    target_file, invalid = get_target_file(args, context)
    configuration_name = args.get("configuration_name")

    target_file, found = try_find_target_file_from_prev_tool_call_if_need(args, context, function, target_file)
    if found:
        invalid |= found

    if target_file and configuration_name:
        line_number, fixed = as_int_or_none(args.get("line_number"), "line_number")
        invalid |= fixed
        if line_number is None:
            line_number = 0
            invalid = True

        timeout, fixed = as_int_or_none(args.get("timeout"), "timeout")
        invalid |= fixed

        if invalid:
            configuration_run_arguments = args.get("configuration_run_arguments")
            configuration_environment_variables = args.get("configuration_environment_variables")
            files_to_collect_coverage = args.get("files_to_collect_coverage")
            new_function = RunConfiguration().new_call(target_file=target_file, configuration_name=configuration_name,
                                                       line_number=line_number, timeout=timeout,
                                                       configuration_run_arguments=configuration_run_arguments,
                                                       configuration_environment_variables=configuration_environment_variables,
                                                       files_to_collect_coverage=files_to_collect_coverage)
            return new_function
    else:
        log.error(f"no required args for function {function.name}, args={args}, "
                  f"required args = ['target_file', 'configuration_name']")
    return function
