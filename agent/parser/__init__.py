import json
import logging
from enum import Enum
from typing import Any, Literal, Callable

from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCallFunction
from pydantic import BaseModel

from agent.openai.chat_api import ROLE_ASSISTANT
from agent.openai.chat_completions_api import FunctionDefinition

log = logging.getLogger(__name__)

THINK_START = "<think>"
THINK_END = "</think>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


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
        self.role: Literal["developer", "system", "user", "assistant", "tool"] | None = None
        self.prefill_tokens: list[str] | None = None

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
    anonymous_arguments: list[Any] = []

    def to_openai_function_call(self) -> ChoiceDeltaToolCallFunction:
        return ChoiceDeltaToolCallFunction(name=self.name, arguments=json.dumps(self.arguments, ensure_ascii=False))


class Parser[State: ParserState]():
    def new_state(self, prompt: str = "", init_chat_events=True) -> State:
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
        return token.strip() == THINK_END

    def is_think_start(self, state: State, token: str) -> bool:
        return token.strip() == THINK_START

    def is_conversation_start(self, state: State, token: str) -> tuple[bool, str]:
        return False, token

    def is_sequence_end(self, state: State, token: str) -> bool:
        return IM_END == token.strip()

    def is_text_end(self, state: State, token: str) -> bool:
        return False

    def is_probably_tool_call_start(self, state: State, token: str) -> bool:
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

    def parse_tool_calls(self, state: State, tool_call_expression: str) -> tuple[list[ParsedFunctionCall], bool]:
        return [], False

    def is_assistant(self, role):
        return self.get_assistant_role_name() == role

    def get_assistant_role_name(self) -> str:
        return ROLE_ASSISTANT

    def is_erase(self, state: State, token: str) -> bool:
        return False


def fill_state_by_prompt_tail(init_chat_events: bool, prompt: str, state: ParserState,
                              is_assistant: Callable[[str], bool]):
    prompt = prompt.rstrip()
    tail_size = 200
    tail = prompt[-tail_size:] if len(prompt) > tail_size else prompt
    tail_lines = tail.rstrip().splitlines()
    if init_chat_events:
        is_think = None
        is_conversation = None
        role = ""

        for i, line in enumerate(reversed(tail_lines)):
            line = line.strip()
            if line.endswith(THINK_START):
                is_think = i
                log.debug(f"state init is_think: {is_think}")
            elif line.startswith(IM_START):
                is_conversation = i
                log.debug(f"state init is_conversation: {is_conversation}")
                prompt_role = line[len(IM_START):].strip()
                is_assistant = is_assistant(prompt_role)
                if is_assistant:
                    role = ROLE_ASSISTANT
                    log.debug(f"state init role: {role}")
                    break
        prefill_i = None
        if not is_conversation is None:
            state.start_event(StateEvent.CONVERSATION)
            prefill_i = (len(tail_lines) - 1 - is_conversation)
        if not is_think is None:
            state.start_event(StateEvent.THINK)
            prefill_i = (len(tail_lines) - 1 - is_think)
        state.role = role

        if not prefill_i is None:
            prefill_i += 1
            if prefill_i < len(tail_lines):
                out_tokens = tail_lines[prefill_i:]
                out_tokens.append("\n")
                state.prefill_tokens = out_tokens
