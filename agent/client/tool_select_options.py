from abc import ABC, abstractmethod
from typing import List

from common.openai_model import ToolDefinition, FunctionCall
from veai.tool import ask_user_with_options


class ToolSelectOptions(ABC):
    @abstractmethod
    def new_call(self, question: str, answers: list[str], is_multiple_choice: bool = False) -> FunctionCall:
        pass


def detect_select_options(tool_definitions: List[ToolDefinition] | None) -> ToolSelectOptions | None:
    if tool_definitions:
        for tool_definition in tool_definitions:
            parsed = ask_user_with_options.detect(tool_definition)
            if parsed:
                return parsed
    return None
