import json
from typing import Any

from agent.client.tool_select_options import ToolSelectOptions
from agent.openai.chat_completions_api import FunctionCall, ToolDefinition
from . import Tool

function_name = "ask_user_with_options"

function_parameters = {"question", "options", "is_multiple_choice"}


class AskUserWithOptions(ToolSelectOptions, Tool):
    @property
    def name(self) -> str:
        return function_name

    def new_call(self, question: str, answers: list[str], is_multiple_choice: bool = False) -> FunctionCall:
        arguments: dict[str, Any] = {
            "question": question,
            "options": answers,
            "is_multiple_choice": is_multiple_choice,
        }
        arguments_str = json.dumps(arguments, ensure_ascii=False)
        return FunctionCall(name=function_name, arguments=arguments_str)


def detect(tool_definition: ToolDefinition) -> AskUserWithOptions | None:
    if tool_definition.type == "function" and tool_definition.function.name == function_name:
        parameters = tool_definition.function.parameters
        type = parameters.get("type")
        required = set(parameters.get("required"))
        if "object" == type and function_parameters == required:
            return AskUserWithOptions()
    return None
