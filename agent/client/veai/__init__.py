from typing import Any

from agent.client import is_agent
from agent.client.user_context import UserContext
from agent.openai.chat_completions_api import ChatCompletionMessageParam

PROJECT_ABSOLUTE_PATH_ = "Project absolute path:"
OS_INFO_ = "OS info:"


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

    return _get_context(content)


def _get_context(content: str | list[dict[str, Any]] | None) -> UserContext | None:
    if isinstance(content, str):
        has_start = content.find("<project_information>")
        has_end = content.find("</project_information>")
        if has_start >= 0 and 0 <= has_end < len(content):
            context = UserContext()
            project_info = content[has_start:has_end].splitlines()
            for line in project_info:
                if line.startswith(OS_INFO_):
                    context.os = line[len(OS_INFO_):].strip()
                elif line.startswith(PROJECT_ABSOLUTE_PATH_):
                    context.workdir = line[len(PROJECT_ABSOLUTE_PATH_):].strip()
            return context

    return None
