"""Detector and call builder for the ask_user_with_options tool."""
from agent.client.tool_select_model import ToolSelectOptions
from agent.openai.chat_completions_api import ChatCompletionFunctionToolParam
from agent.parser import ParsedFunctionCall
from . import Tool

FUNCTION_NAME = "ask_user_with_options"

function_parameters = {"question", "options", "is_multiple_choice"}


class AskUserWithOptions(ToolSelectOptions, Tool):
    """Builds ask_user_with_options tool calls from user-selection arguments."""

    @property
    def name(self) -> str:
        """Tool name as declared in tool definitions."""
        return FUNCTION_NAME

    def new_call(self, question: str, options: list[str], is_multiple_choice: bool = False) -> ParsedFunctionCall:
        """Build the parsed function call with question, options and choice mode."""
        return ParsedFunctionCall(name=FUNCTION_NAME, arguments={
            "question": question,
            "options": options,
            "is_multiple_choice": is_multiple_choice,
        })


def detect(tool_definition: ChatCompletionFunctionToolParam) -> AskUserWithOptions | None:
    """Return AskUserWithOptions when the tool definition matches this tool."""
    if tool_definition.type == "function" and tool_definition.function.name == FUNCTION_NAME:
        parameters = tool_definition.function.parameters
        params_type = parameters.type
        required = parameters.required
        if "object" == params_type and function_parameters == required:
            return AskUserWithOptions()
    return None
