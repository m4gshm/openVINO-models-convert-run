from abc import ABC, abstractmethod
from typing import Callable

from agent.multimodal.content_parts import content_text
from agent.openai.chat_completions_api import ChatCompletionMessageParam
from agent.parser import ParsedFunctionCall


class ToolSelectOptions(ABC):
    @abstractmethod
    def new_call(self, question: str, answers: list[str], is_multiple_choice: bool = False) -> ParsedFunctionCall:
        pass


def is_agent(messages: list[ChatCompletionMessageParam], marker: str) -> bool:
    def check(content: str) -> bool:
        return marker in content

    return check_agent(messages, check)


def check_agent(messages: list[ChatCompletionMessageParam], check: Callable[..., bool]) -> bool:
    first_message = messages[0] if messages else None
    if not first_message:
        return False
    return check(content_text(first_message.content))
