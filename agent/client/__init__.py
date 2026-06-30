from abc import ABC, abstractmethod

from agent.openai.chat_completions_api import FunctionCall
from agent.parser import ParsedFunctionCall


class ToolSelectOptions(ABC):
    @abstractmethod
    def new_call(self, question: str, answers: list[str], is_multiple_choice: bool = False) -> ParsedFunctionCall:
        pass
