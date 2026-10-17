"""Detection of user-selection tools among provided tool definitions."""
from typing import List

from agent.client.tool_select_model import ToolSelectOptions
from agent.client.veai.tool import ask_user_with_options
from agent.openai.chat_completions_api import ChatCompletionFunctionToolParam


def detect_select_options(tool_definitions: List[ChatCompletionFunctionToolParam] | None) -> ToolSelectOptions | None:
    """Return the first tool definition that is a user-selection tool, if any."""
    if tool_definitions:
        for tool_definition in tool_definitions:
            parsed = ask_user_with_options.detect(tool_definition)
            if parsed:
                return parsed
    return None
