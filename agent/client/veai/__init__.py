import json
import logging
from typing import Any

from agent.client import is_agent
from agent.client.user_context import UserContext, UserContextFiles
from agent.openai.chat_completions_api import ChatCompletionMessageParam, Function

PROJECT_ABSOLUTE_PATH_ = "Project absolute path:"
OS_INFO_ = "OS info:"

log = logging.getLogger(__name__)


def is_veai_agent(messages: list[ChatCompletionMessageParam]) -> bool:
    return is_agent(messages, "You are Veai Agent")


def get_veai_context(messages: list[ChatCompletionMessageParam]) -> UserContext | None:
    # class ContextField(Enum):
    #     OS_INFO = 'OS info', lambda uc: uc.os
    #     PROJECT_ABSOLUTE_PATH = 'Project absolute path', lambda uc: uc.workdir
    #
    #     def __new__(cls, *args, **kwds):
    #         obj = object.__new__(cls)
    #         obj._value_ = args[0]
    #         return obj
    #
    #     def __init__(self, _: str, context_field: Callable[[str, UserContext], None]):
    #         self.context_field: Callable[[str, UserContext], None] = context_field
    #
    #     def __str__(self):
    #         return self.value
    #
    #     @property
    #     def value(self) -> str:
    #         return self._value_
    #
    #     def set_context_value(self, value: str, uc: UserContext):
    #         self.context_field(value, uc)
    #
    # expected_fields: dict[str, ContextField] = {member.name: member for member in ContextField}

    # OS info: Windows 11, version: 10.0, arch: amd64
    # IDE: OpenIDE 2025.3
    # Project name: java
    # Project absolute path: C:\project

    first_message = messages[0] if messages else None
    if not first_message:
        return None

    content = first_message.content

    context = _get_context(content)
    context.files = _get_files(messages)
    return context


def _get_context(system_prompt: str | list[dict[str, Any]] | None) -> UserContext | None:
    if isinstance(system_prompt, str):
        has_start = system_prompt.find("<project_information>")
        has_end = system_prompt.find("</project_information>")
        if has_start >= 0 and 0 <= has_end < len(system_prompt):
            context = UserContext()
            project_info = system_prompt[has_start:has_end].splitlines()
            for line in project_info:
                if line.startswith(OS_INFO_):
                    context.os = line[len(OS_INFO_):].strip()
                elif line.startswith(PROJECT_ABSOLUTE_PATH_):
                    context.workdir = line[len(PROJECT_ABSOLUTE_PATH_):].strip()
            return context

    return None


def _get_files(messages: list[ChatCompletionMessageParam]) -> UserContextFiles:
    file_content_result: dict[str, bytes | None] = {}

    read_file = "read_file"
    write_file = "write_file"
    edit_file = "edit_file"
    read_contents = {}
    read_files: dict[str, list[str]] = {}
    lines = {}
    write_statuses = dict[str, str]()
    for message in messages:
        is_tool = message.role == "tool"
        if is_tool:
            message_name = message.name
            tool_call_id = message.tool_call_id
            if message_name == read_file:
                parsed_content = parse_content(message)
                result = parsed_content.get("result")
                if result == "success with json content":
                    file_content = parsed_content.get("content")
                else:
                    file_content = None
                    pass
                read_contents[tool_call_id] = file_content
            elif message_name == write_file:
                parsed_content = parse_content(message)
                result = parsed_content.get("result")
                write_statuses[tool_call_id] = result
            elif message_name == edit_file:
                pass

        tool_calls = message.tool_calls
        for tool_call in (tool_calls if tool_calls else []):
            function = tool_call.function
            call_id = tool_call.id
            function_name = function.name
            if function_name == write_file:
                arguments = parse_arguments(function)
                if isinstance(arguments, dict):
                    target_file = arguments.get("target_file")
                    content = arguments.get("content")
                    if target_file:
                        file_content_result[target_file] = content.encode('utf-8') if content else None
                        read_files.pop(target_file, None)
            elif function_name == read_file:
                arguments = parse_arguments(function)
                if isinstance(arguments, dict):
                    target_file = arguments.get("target_file")
                    if target_file:
                        read_files.setdefault(target_file, []).append(call_id)
                        lines[call_id] = {
                            "start_line": arguments.get("start_line"),
                            "end_line": arguments.get("end_line"),
                        }

    # file_content_fullness = set[str]()
    for file_name, call_ids in read_files.items():
        if len(call_ids) > 1:
            # merge
            start_lines = {}
            for call_id in call_ids:
                line_info = lines[call_id]
                start_line = line_info.get("start_line")
                content = read_contents.get(call_id)
                start_lines[start_line] = content
            sorted_start_lines = sorted(start_lines.keys())
            text = ""
            for i in sorted_start_lines:
                chunk = start_lines[i]
                text += chunk
            file_content_result[file_name] = text.encode('utf-8')
        else:
            call_id = call_ids[0]
            content = read_contents.get(call_id)
            if content:
                text = content.get("text")
                # file_line_count = content.get("file_line_count")
                # lines_read = content.get("lines_read")
                # if file_line_count is not None and lines_read == file_line_count:
                #     file_content_fullness.add(file_name)
                file_content_result[file_name] = text.encode('utf-8')
                # else:
                #     pass

    return UserContextFiles(file_content_result)


def parse_arguments(function: Function) -> Any:
    arguments_str = function.arguments
    try:
        arguments = json.loads(arguments_str)
    except Exception as e:
        log.error(f"function arguments parsing error, function='{function.name}', error='{e}'")
        arguments = {}
    return arguments


def parse_content(message: ChatCompletionMessageParam) -> Any:
    content = message.content
    try:
        parsed_content = json.loads(content)
    except Exception as e:
        log.error(f"error on parse content of tool_call={message.tool_call_id}, content='{content}'")
        parsed_content = {}
    return parsed_content
