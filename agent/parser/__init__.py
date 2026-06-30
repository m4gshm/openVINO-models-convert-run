import json
import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel

from agent.common.roles import ROLE_ASSISTANT
from agent.openai.chat_completions_api import FunctionDefinition, FunctionCall

log = logging.getLogger(__name__)


class StateEvent(Enum):
    CONVERSATION = 1
    THINK = 2
    TOOL_CALL = 3
    TOOL_RESPONSE = 4
    FIM_MIDDLE = 5


class ParserState:
    def __init__(self, supported_functions: dict[str, FunctionDefinition] | None = None):
        super().__init__()
        self.supported_functions = supported_functions if supported_functions else {}
        self.__events: list[StateEvent] = []
        self.fim_middle_start = False
        # self.thinking_progress_counter: int = 0
        self.expect_tool_response = False
        self.role: str | None = None


    def get_function_parameters(self, func_name: str) -> dict[str, Any] | dict[Any, Any]:
        supported_functions = self.supported_functions
        function = supported_functions.get(func_name)
        parameters = function.parameters if function is not None else {}
        return parameters

    def start_event(self, event: StateEvent):
        return self.__events.append(event)

    def get_current_event(self) -> StateEvent | None:
        return self.__events[-1] if self.__events else None

    def has_event(self, event: StateEvent) -> bool:
        return event in self.__events

    def events(self) -> list[StateEvent]:
        return self.__events

    def finish_current_event(self, expected_state: StateEvent | None, parent_log: logging.Logger | None = None):
        s = self.get_current_event()
        if not expected_state or s == expected_state:
            self.__events.pop()
        else:
            l = parent_log if parent_log else log
            l.error(f"unexpected state {s}, expected {expected_state}")

    def finalize(self, token: str):
        pass


def _is_conversation_start(tag: str, token: str) -> tuple[bool, str]:
    token = token.strip()
    b = token.startswith(tag)
    tail = token[len(tag): len(token)] if b and len(token) > len(tag) else ""
    return b, tail


class ParsedFunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any]
    anonymous_arguments: list[str] = []

    def to_openai_function_call(self) -> FunctionCall:
        return FunctionCall(name=self.name, arguments=json.dumps(self.arguments, ensure_ascii=False))


class Parser[State: ParserState]():
    def new_state(self, init_chat_events=True) -> State:
        state = self._new_state()
        if init_chat_events:
            state.start_event(StateEvent.CONVERSATION)
            state.role = ROLE_ASSISTANT
        return state

    def process_chat_prompt(self, prompt: str) -> str:
        return prompt

    def _new_state(self) -> ParserState:
        return ParserState()

    def is_end(self, state: State, token: str) -> bool:
        return False

    def is_fim_middle(self, state: State, token: str) -> bool:
        return False

    def is_think_end(self, state: State, token: str) -> bool:
        return False

    def is_think_start(self, state: State, token: str) -> bool:
        return False

    def is_conversation_start(self, state: State, token: str) -> tuple[bool, str]:
        return False, token

    def is_conversation_end(self, state: State, token: str) -> bool:
        return False

    def is_tool_call_start(self, state: State, token: str) -> bool:
        return False

    def is_tool_call_end(self, state: State, token: str) -> bool:
        return False

    def is_tool_response_start(self, state: State, token: str) -> bool:
        return False

    def is_tool_response_end(self, state: State, token: str) -> bool:
        return False

    def is_prompt_start_thinking(self, prompt: str) -> bool:
        pass

    def parse_tool_calls(self, state: State, tool_call_expression: str | None) -> tuple[list[ParsedFunctionCall], bool]:
        return [], False

    def is_assistant(self, role):
        return self.get_assistant_role_name() == role

    def get_assistant_role_name(self) -> str:
        return ROLE_ASSISTANT

    def is_erase(self, state: State, token: str) -> bool:
        return False
