"""Detector and call builder for the OpenCode 'question' tool."""
from agent.client.tool_select_model import ToolSelectOptions
from agent.openai.chat_completions_api import ChatCompletionFunctionToolParam
from agent.parser import ParsedFunctionCall

FUNCTION_NAME = "question"


class QuestionTool(ToolSelectOptions):  # pylint: disable=too-few-public-methods
    """Builds OpenCode 'question' tool calls."""

    @property
    def name(self) -> str:
        return FUNCTION_NAME

    def new_call(self, question_text: str, options: list[str], is_multiple_choice: bool = False) -> ParsedFunctionCall:
        """Build a parsed function call for the OpenCode 'question' tool."""
        return ParsedFunctionCall(name=FUNCTION_NAME, arguments={
            "questions": [{
                "question": question_text,
                "header": "",
                "options": [{"label": opt, "description": ""} for opt in options],
                "multiple": is_multiple_choice,
            }],
        })


def detect(tool_definition: ChatCompletionFunctionToolParam) -> QuestionTool | None:
    """Return QuestionTool when the tool definition matches OpenCode 'question'."""
    if tool_definition.type == "function" and tool_definition.function.name == FUNCTION_NAME:
        # Verify it has the 'questions' array parameter
        params = tool_definition.function.parameters
        if hasattr(params, "properties") and "questions" in params.properties:
            return QuestionTool()
    return None