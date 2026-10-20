"""Reconstruction of user context (files, OS, project dir) from the Veai conversation."""
import json
import logging
from collections import defaultdict
from logging import Logger
from pathlib import Path
from typing import Any

from agent.client import is_agent
from agent.client.user_context import UserContext, UserContextFiles, OS
from agent.multimodal.content_parts import content_text
from agent.openai.chat_completions_api import ChatCompletionMessageParam, Function

END_LINE = "end_line"

TEXT = "text"

PROJECT_ABSOLUTE_PATH_ = "Project absolute path:"
OS_INFO_ = "OS info:"

log: Logger = logging.getLogger(__name__)


def is_veai_agent(messages: list[ChatCompletionMessageParam]) -> bool:
    """Check that the conversation is a Veai Agent session."""
    return is_agent(messages, "You are Veai Agent")


def get_veai_context(messages: list[ChatCompletionMessageParam]) -> UserContext | None:
    """Extract the user context carried by the first (system) message."""
    # The system message of the IDE looks like this:
    #
    #   <project_information>
    #   OS info: Windows 11, version: 10.0, arch: amd64
    #   IDE: OpenIDE 2025.3
    #   Project name: java
    #   Project absolute path: C:\project
    #   </project_information>
    first_message = messages[0] if messages else None
    if not first_message:
        return None

    content = content_text(first_message.content)

    context = _get_context(content)
    if context:
        context.files = _get_files(context.workdir, messages)
    return context


def _get_context(system_prompt: str | list[dict[str, Any]] | None) -> UserContext:
    context = UserContext()
    if isinstance(system_prompt, str):
        has_start = system_prompt.find("<project_information>")
        has_end = system_prompt.find("</project_information>")
        if has_start >= 0 and 0 <= has_end < len(system_prompt):
            project_info = system_prompt[has_start:has_end].splitlines()
            for line in project_info:
                if line.startswith(OS_INFO_):
                    os_full = line[len(OS_INFO_):].strip()
                    context.os_full = os_full
                    context.os_type = OS.Windows if "windows" in os_full.lower() else None
                elif line.startswith(PROJECT_ABSOLUTE_PATH_):
                    context.workdir = Path(line[len(PROJECT_ABSOLUTE_PATH_):].strip())
    return context


def _get_files(root: Path | None, messages: list[ChatCompletionMessageParam]) -> UserContextFiles:
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-nested-blocks  # message-scan dispatcher; splitting it is a separate refactor
    list_dir_call: dict[str, str] = {}
    list_dir_result: dict[str, list[str]] = {}
    file_content_result: dict[Path, bytes | None] = {}

    list_dir = "list_dir"
    read_file = "read_file"
    write_file = "write_file"
    edit_file = "edit_file"
    read_contents = {}
    read_files: dict[Path, dict[int, tuple[int, str]]] = {}
    write_statuses = dict[str, str]()
    start_line_key = "start_line"
    for message in messages:
        is_tool = message.role == "tool"
        if is_tool:
            tool_call_id = message.tool_call_id
            if message.name == list_dir:
                list_dir_result[tool_call_id] = read_list_dir(message)
            elif message.name == read_file:
                parsed_content = parse_content(message)
                result = parsed_content.get("result")
                if result == "success with json content":
                    file_content = parsed_content.get("content")
                    if file_content:
                        read_contents[tool_call_id] = file_content
                else:
                    pass
            elif message.name == write_file:
                parsed_content = parse_content(message)
                result = parsed_content.get("result")
                write_statuses[tool_call_id] = result
                # reset prev read file
            elif message.name == edit_file:
                # reset prev read file
                # edit_file tool result example:
                # '{"result":"success with json content","content":'
                # '{"file_path":"agent/requirements.txt","failed_edits":[]}}'
                pass

        tool_calls = message.tool_calls
        for tool_call in (tool_calls if tool_calls else []):
            function = tool_call.function
            call_id = tool_call.id
            function_name = function.name
            if function_name == list_dir:
                arguments = parse_arguments(function)
                if isinstance(arguments, dict):
                    directory_path = arguments.get("directory_path")
                    if directory_path:
                        list_dir_call[call_id] = directory_path
            elif function_name == write_file:
                arguments = parse_arguments(function)
                if isinstance(arguments, dict):
                    target_file = get_target_file_as_path(arguments)
                    content = arguments.get("content")
                    if target_file:
                        file_content_result[target_file] = content.encode('utf-8') if content else None
                        clean_read_files(target_file, read_files)
            elif function_name == edit_file:
                arguments = parse_arguments(function)
                if isinstance(arguments, dict):
                    target_file = get_target_file_as_path(arguments)
                    if target_file:
                        removed_file_content = clean_read_files(target_file, read_files)
                        if removed_file_content:
                            log.debug("file content removed from context by %s, file=%s", edit_file, target_file)
            elif function_name == read_file:
                arguments = parse_arguments(function)
                if isinstance(arguments, dict):
                    target_file = get_target_file_as_path(arguments)
                    if target_file:
                        start_line_int = as_int_or_none(arguments.get(start_line_key))
                        end_line_int = as_int_or_none(arguments.get(END_LINE))
                        if not (start_line_int is None or end_line_int is None):
                            lines_ranges = read_files.setdefault(target_file, {})
                            lines_ranges.setdefault(start_line_int, (end_line_int, call_id))

    files_hierarchy = tree_structure()
    if root:
        add_path_to_dict(files_hierarchy, root)
    for call_id, directory in list_dir_call.items():
        if directory != ".":
            add_path_to_dict(files_hierarchy, directory)

    for file_name, line_ranges_with_call_id in read_files.items():
        if len(line_ranges_with_call_id) > 1:
            # merge
            start_lines = {}
            for start_line, (_end_line, call_id) in line_ranges_with_call_id.items():
                content = read_contents.get(call_id)
                if content:
                    start_lines[start_line] = content
            sorted_start_lines = sorted(start_lines.keys())
            text = ""
            for i in sorted_start_lines:
                chunk = start_lines[i]
                chunk_text = chunk.get(TEXT, None)
                text += chunk_text if chunk_text else ""
            if text:
                file_content_result[file_name] = text.encode('utf-8')
        else:
            items = line_ranges_with_call_id.items()
            start_line, (_end_line, call_id) = next(iter(items))
            content = read_contents.get(call_id)
            text = content.get(TEXT, None) if content else None
            if text:
                file_content_result[file_name] = text.encode('utf-8')

    return UserContextFiles(file_content_result)


def clean_read_files(target_file: Path,
                     read_files: dict[Path, dict[int, tuple[int, str]]]) -> dict[int, tuple[int, str]] | None:
    """Drop and return the tracked line ranges of the file after write/edit."""
    return read_files.pop(target_file, None)


def as_int_or_none(val: Any | None) -> int | None:
    """Return the value as int; None when it is neither str nor int."""
    if isinstance(val, str):
        return int(val)
    if isinstance(val, int):
        return val
    return None


def read_list_dir(message: ChatCompletionMessageParam) -> list[str] | None:
    # pylint: disable=too-many-nested-blocks  # guard chain over optional result fields; early-return rewrite is a separate refactor
    """Return directory tree lines from a list_dir tool result message."""
    parsed_content = parse_content(message)
    result = parsed_content.get("result")
    if result == "success with json content":
        tree_content = parsed_content.get("content")
        if tree_content:
            directory_tree_str = tree_content.get("directory_tree")
            if directory_tree_str and isinstance(directory_tree_str, str):
                subdirs: list[str] | None = directory_tree_str.split("\n")
                too_long_prefix = "The directory_tree field is too large, it was dumped to file"
                too_long_postfix = "Be sure to use the read_file tool right away to read it"
                if len(subdirs) == 1 and directory_tree_str.startswith(too_long_prefix):
                    # read dumped file
                    tail = directory_tree_str[len(too_long_prefix):]
                    postfix_start = tail.index(too_long_postfix)
                    if postfix_start > 0:
                        dump_file_name_raw = tail[:postfix_start].strip()
                        if dump_file_name_raw.startswith("\""):
                            dump_file_name_raw = dump_file_name_raw[1:]
                            last_quote_index = dump_file_name_raw.rfind("\"")
                            if last_quote_index > 0:
                                dump_file_name_raw = dump_file_name_raw[:last_quote_index]

                        dump_file_name = dump_file_name_raw
                        try:
                            with open(dump_file_name, 'r', encoding='utf-8') as file:
                                content = file.read()
                                subdirs = content.split("\n")
                        except FileNotFoundError:
                            log.error("dump file not found: %s", dump_file_name)
                            subdirs = None
                return subdirs
    return None


def get_target_file_as_path(arguments: dict) -> Path | None:
    """Return the target_file argument as a Path, if present."""
    target_file = arguments.get("target_file")
    return Path(target_file) if target_file else None


def parse_arguments(function: Function) -> Any:
    """Parse model-generated tool-call arguments, degrading to empty dict on failure."""
    arguments_str = function.arguments
    try:
        arguments = json.loads(arguments_str)
    except Exception as e:  # pylint: disable=broad-exception-caught  # model-generated JSON: any failure must degrade to {}
        log.error("function arguments parsing error, function='%s', error='%s'", function.name, e)
        arguments = {}
    return arguments


def parse_content(message: ChatCompletionMessageParam) -> Any:
    """Parse a tool-result message content, degrading to empty dict on failure."""
    content = content_text(message.content)
    try:
        parsed_content = json.loads(content)
    except Exception:  # pylint: disable=broad-exception-caught  # model-generated JSON: any failure must degrade to {}
        log.error("error on parse content of tool_call=%s, content='%s'", message.tool_call_id, content)
        parsed_content = {}
    return parsed_content


def tree_structure() -> defaultdict:
    """Build an infinitely nested defaultdict used as a path tree."""
    return defaultdict(tree_structure)


def add_path_to_dict(tree: defaultdict, path: str | Path):
    """Add the path parts into the nested tree, branch per path segment."""
    path = path if isinstance(path, Path) else Path(path)
    parts = path.parts
    current = tree
    for part in parts:
        if part not in ('/', '\\'):
            current = current[part]
