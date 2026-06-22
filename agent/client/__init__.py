from abc import ABC, abstractmethod

from agent.openai.chat_completions_api import FunctionCall


class ToolSelectOptions(ABC):
    @abstractmethod
    def new_call(self, question: str, answers: list[str], is_multiple_choice: bool = False) -> FunctionCall:
        pass
