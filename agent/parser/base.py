from abc import abstractmethod, ABC

from common.openai_model import FunctionDefinition, ToolCall


class Parser(ABC):

    @abstractmethod
    def think_is_over(self, subword: str) -> bool:
        pass

    @abstractmethod
    def think_is_started(self, subword: str) -> bool:
        pass

    @abstractmethod
    def is_conversation_start(self, text: str) -> bool:
        pass

    @abstractmethod
    def is_conversation_end(self, text: str) -> bool:
        pass

    @abstractmethod
    def is_tool_call_start(self, text: str) -> bool:
        pass

    @abstractmethod
    def is_tool_call_end(self, text: str) -> bool:
        pass

    @abstractmethod
    def is_prompt_start_thinking(self, prompt: str) -> bool:
        pass

    @abstractmethod
    def is_partial_tool_call(self, text: str) -> bool:
        pass

    @abstractmethod
    def parse_tool_calls(self, text: str, supported_functions: dict[str, FunctionDefinition] | None = None) -> tuple[
        list[ToolCall], bool]:
        pass
