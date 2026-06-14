import json
from typing import Any, List

from client.tool_select_options import ToolSelectOptions
from common.openai_model import FunctionCall, ToolDefinition

function_name = "ask_user_with_options"

class AskUserWithOptions(ToolSelectOptions):

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
        properties = parameters.get("properties")
        if "object" == type and properties and isinstance(properties, dict):
            question = properties.get("question")
            options = properties.get("question")
            is_multiple_choice = properties.get("is_multiple_choice")
            if question and options and is_multiple_choice:
                return AskUserWithOptions()
    return None
