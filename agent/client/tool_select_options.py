from typing import List

from . import ToolSelectOptions
from .veai.tool import ask_user_with_options
from ..openai.chat_completions_api import ToolDefinition


def detect_select_options(tool_definitions: List[ToolDefinition] | None) -> ToolSelectOptions | None:
    if tool_definitions:
        for tool_definition in tool_definitions:
            parsed = ask_user_with_options.detect(tool_definition)
            if parsed:
                return parsed
    return None
