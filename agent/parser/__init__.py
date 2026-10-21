"""Streaming prompt parsers: shared state machine and the parser interface.

A parser turns the incremental token stream of a model into conversation events
(think blocks, tool calls, tool responses) and parses finished tool call
expressions into ParsedFunctionCall objects. Model families subclass Parser and
override the token predicates they need.
"""
import json
import logging
import re
from enum import Enum
from typing import Any, Iterator, Literal, Callable

from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCallFunction
from openvino_genai import Tokenizer
from pydantic import BaseModel

from agent.inference.phrase import Phrase
from agent.openai.chat_api import ROLE_ASSISTANT
from agent.openai.chat_completions_api import FunctionDefinitionParameters

log = logging.getLogger(__name__)

THINK_START = "<think>"
THINK_END = "</think>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


class StateEvent(Enum):
    """Kind of the section of the generated text the parser is currently inside."""

    CONVERSATION = 1
    THINK = 2
    TOOL_CALL = 3
    TOOL_RESPONSE = 4
    FIM_MIDDLE = 5


class ParserState:
    """Mutable parsing context shared by a parser and the token handler."""

    def __init__(self, supported_functions: dict[str, FunctionDefinitionParameters] | None = None):
        super().__init__()
        self.supported_functions = supported_functions if supported_functions else dict[
            str, FunctionDefinitionParameters]()
        self.__events: list[StateEvent] = []
        self.role: Literal["developer", "system", "user", "assistant", "tool"] | None = None
        self.prefill_tokens: list[str] | None = None
        self.is_streaming_delayed = False

    def get_function_parameters(self, func_name: str) -> dict:
        """Return the JSON schema properties expected for a supported function."""
        supported_functions = self.supported_functions
        parameters = supported_functions.get(func_name)
        return parameters.properties if parameters else {}

    def start_event(self, event: StateEvent):
        """Push a new section event onto the state stack."""
        return self.__events.append(event)

    def get_current_event(self) -> StateEvent | None:
        """Return the innermost open section event."""
        return self.__events[-1] if self.__events else None

    def has_event(self, event: StateEvent) -> bool:
        """True when the event was opened and is not finished yet."""
        return event in self.__events

    def events(self) -> list[StateEvent]:
        """Return the stack of the open section events."""
        return self.__events

    def finish_current_event(self, expected_state: StateEvent | None, parent_log: logging.Logger | None = None):
        """Pop the innermost event, logging a mismatch against ``expected_state``."""
        s = self.get_current_event()
        if not expected_state or s == expected_state:
            self.__events.pop()
        else:
            l = parent_log if parent_log else log
            l.error("unexpected state %s, expected %s", s, expected_state)

    def finalize(self, token: str):
        """Handle the very last token of the generation (no-op by default)."""


def _is_conversation_start(tag: str, token: str) -> tuple[bool, str]:
    """Match a conversation start tag and return it with the remaining tail."""
    token = token.strip()
    b = token.startswith(tag)
    tail = token[len(tag): len(token)] if b and len(token) > len(tag) else ""
    return b, tail


def match_name_and_tail(pattern: str, parameters_block: str) -> tuple[str | None, str | None]:
    """Split a function block by ``pattern`` into the stripped name and tail groups."""
    match = re.search(pattern, parameters_block, re.DOTALL)
    if match:
        name = match.group(1).strip()
        tail = match.group(2).strip()
        return name, tail
    return None, None


def iter_tool_call_blocks(tool_call_expression: str, tool_call_start: str, tool_call_end: str,
                          function_start_pref: str) -> list[tuple[list[str], bool]]:
    """Split a tool call expression into ``(function blocks, is_call_closed)`` groups.

    Shared skeleton of the tool call parsers: the expression is cut by
    ``tool_call_start``; a block that does not end with ``tool_call_end`` is a
    partially streamed call; every block is then split into raw function blocks
    by ``function_start_pref``.
    """
    blocks: list[tuple[list[str], bool]] = []
    for call_block in tool_call_expression.lstrip().split(tool_call_start):
        if len(call_block) == 0:
            continue
        call_block_rstrip = call_block.rstrip()
        is_call_closed = call_block_rstrip.endswith(tool_call_end)
        if is_call_closed:
            call_block_rstrip = call_block_rstrip[:-len(tool_call_end)]
        function_blocks = [function_block.rstrip() for function_block in
                           call_block_rstrip.lstrip().split(function_start_pref) if len(function_block) > 0]
        blocks.append((function_blocks, is_call_closed))
    return blocks


def iter_function_blocks(tool_call_expression: str, tool_call_start: str, tool_call_end: str,
                         function_start_pref: str) -> Iterator[str]:
    """Yield the raw function blocks of a tool call expression, flattened over its call blocks."""
    for function_blocks, _ in iter_tool_call_blocks(tool_call_expression, tool_call_start, tool_call_end,
                                                    function_start_pref):
        yield from function_blocks


class ParsedFunctionCall(BaseModel):
    """A single tool call parsed from the model output."""

    name: str
    arguments: dict[str, Any]
    anonymous_arguments: list[Any] = []

    def to_openai_function_call(self) -> ChoiceDeltaToolCallFunction:
        """Serialize into an OpenAI protocol tool call delta."""
        return ChoiceDeltaToolCallFunction(name=self.name, arguments=json.dumps(self.arguments, ensure_ascii=False))


class DelayedResult(BaseModel):
    """Content and tool calls held back while streaming is delayed."""

    content: str | None = None
    tool_calls: list[ParsedFunctionCall] = []


# pylint: disable=too-many-public-methods
# Hook arguments are part of the uniform interface: the base implementations
# below ignore most of them, but subclasses override them by name.
# pylint: disable=unused-argument
class Parser[State: ParserState]:
    """Base parser interface: token predicates and tool call parsing hooks.

    Default implementations ignore ``state``/``token``; subclasses override the
    hooks their model syntax needs while keeping the uniform signature used by
    the token handler.
    """

    def __init__(self, tokenizer: Tokenizer | None = None):
        """Store the tokenizer of the model and read the end token from it."""
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.get_eos_token() if tokenizer is not None else ""

    def new_state(self, prompt: str = "", supported_functions: dict[str, FunctionDefinitionParameters] | None = None,
                  init_chat_events=True) -> State:
        """Create a fresh state, optionally seeded with the assistant chat event."""
        state = self._new_state(supported_functions)
        if init_chat_events:
            state.start_event(StateEvent.CONVERSATION)
            state.role = ROLE_ASSISTANT
        return state

    def process_chat_prompt(self, prompt: str) -> str:
        """Adjust the raw chat prompt before generation (no-op by default)."""
        return prompt

    def _new_state(self, supported_functions: dict[str, FunctionDefinitionParameters] | None = None) -> ParserState:
        """Factory of the concrete state object."""
        return ParserState(supported_functions=supported_functions)

    def is_end(self, state: State, token: str) -> bool:
        """True when the token ends the generation."""
        return token == self.eos_token

    def is_fim_middle(self, state: State, token: str) -> bool:
        """True when the token switches the parser into fill-in-the-middle mode."""
        return False

    def is_think_end(self, state: State, token: str) -> bool:
        """True when the token closes a think block."""
        return token.strip() == THINK_END

    def is_think_start(self, state: State, token: str) -> bool:
        """True when the token opens a think block."""
        return token.strip() == THINK_START

    def is_conversation_start(self, state: State, token: str) -> tuple[bool, str]:
        """True (with the unparsed tail) when the token starts the assistant turn."""
        return False, token

    def is_sequence_end(self, state: State, token: str) -> bool:
        """True when the token ends the whole generated sequence."""
        return IM_END == token.strip()

    def is_text_end(self, state: State, token: str) -> bool:
        """True when the token ends a plain text section."""
        return False

    def is_delay_streaming_start(self, state: State, token: str) -> bool:
        """True when content streaming must be delayed starting from this token."""
        return False

    def is_delay_streaming_end(self, state: State, token: str) -> bool:
        """True when the delayed content must be flushed to the client."""
        return False

    def is_support_multiple_tool_calls(self, state: State, token: str) -> bool:
        """True when several tool calls may share a single expression."""
        return True

    def is_tool_call_start(self, state: State, token: str) -> bool:
        """True when the token starts a tool call expression."""
        return False

    def is_tool_call_end(self, state: State, token: str) -> bool:
        """True when the token ends a tool call expression."""
        return False

    def is_tool_response_start(self, state: State, token: str) -> bool:
        """True when the token starts a tool response section."""
        return False

    def is_tool_response_end(self, state: State, token: str) -> bool:
        """True when the token ends a tool response section."""
        return False

    def is_prompt_start_thinking(self, prompt: str) -> bool:
        """True when the prompt already ends with a thinking marker."""
        return False

    def parse_tool_calls(self, state: State, tool_call_expression: str) -> tuple[list[ParsedFunctionCall], bool]:
        """Parse a tool call expression into calls plus a partially streamed flag."""
        return [], False

    def is_assistant(self, role):
        """True when the role name is the assistant role of the model."""
        return self.get_assistant_role_name() == role

    def get_assistant_role_name(self) -> str:
        """Role name the chat template uses for assistant messages."""
        return ROLE_ASSISTANT

    def is_erase(self, state: State, token: str) -> bool:
        """True when the token must be erased from the accumulated text."""
        return False

    def handle_delayed_phrase(self, phrase: Phrase) -> DelayedResult | None:
        """Parse a delayed phrase into content and tool calls (no-op by default)."""
        return None


def fill_state_by_prompt_tail(init_chat_events: bool, prompt: str, state: ParserState,
                              is_assistant: Callable[[str], bool]):
    """Rebuild conversation/think events from the tail of a continued chat prompt."""
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
                log.debug("state init is_think: %s", is_think)
            elif line.startswith(IM_START):
                is_conversation = i
                log.debug("state init is_conversation: %s", is_conversation)
                prompt_role = line[len(IM_START):].strip()
                is_assistant = is_assistant(prompt_role)
                if is_assistant:
                    role = ROLE_ASSISTANT
                    log.debug("state init role: %s", role)
                    break
        prefill_i = None
        if not is_conversation is None:
            state.start_event(StateEvent.CONVERSATION)
            prefill_i = len(tail_lines) - 1 - is_conversation
        if not is_think is None:
            state.start_event(StateEvent.THINK)
            prefill_i = len(tail_lines) - 1 - is_think
        state.role = role

        if not prefill_i is None:
            prefill_i += 1
            if prefill_i < len(tail_lines):
                out_tokens = tail_lines[prefill_i:]
                out_tokens.append("\n")
                state.prefill_tokens = out_tokens
