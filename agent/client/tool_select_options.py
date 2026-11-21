"""Detection of user-selection tools among provided tool definitions."""
from typing import List

from agent.client.opencode import question_tool as opencode_question
from agent.client.tool_select_model import AgentType, DetectedSelectOptions, ToolSelectOptions
from agent.client.veai.tool import ask_user_with_options as veai_ask
from agent.openai.chat_completions_api import ChatCompletionFunctionToolParam


def detect_select_options(tool_definitions: List[ChatCompletionFunctionToolParam] | None) -> DetectedSelectOptions | None:
    """Return agent type + detected user-selection tool, if any.

    Checks each agent's detector in order; returns the first match.
    """
    if not tool_definitions:
        return None
    for tool_definition in tool_definitions:
        veai_tool = veai_ask.detect(tool_definition)
        if veai_tool:
            return DetectedSelectOptions(agent=AgentType.VEAI, tool=veai_tool)
        question_tool = opencode_question.detect(tool_definition)
        if question_tool:
            return DetectedSelectOptions(agent=AgentType.OPENCODE, tool=question_tool)
    return None