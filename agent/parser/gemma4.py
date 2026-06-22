import json
import logging
import re
from typing import Any

import agent
from agent.openai.chat_api import new_tool_call
from agent.openai.chat_completions_api import ToolCall, FunctionCall, FunctionDefinition
from agent.parser import Parser, _is_conversation_start, StateEvent

ROLE = "model"

EXPECTED_PROPERTY_TYPE = 'type'
EXPECTED_PARAMETERS_PROPERTIES = "properties"

THOUGHT = "thought"

CHANNEL_START = "<|channel>"
CHANNEL_END = "<channel|>"

TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"

TOOL_RESPONSE_START = "<|tool_response>"
TOOL_RESPONSE_END = "<tool_response|>"

FUNCTION_START_PREF = "call:"

TURN_START = "<|turn>"
TURN_END = "<turn|>"

# REASONING_START = "<|turn>model"
# REASONING_END = "</thinking>"

spec = {CHANNEL_START, CHANNEL_END, TOOL_CALL_START, TOOL_CALL_END, TOOL_RESPONSE_START, TOOL_RESPONSE_END, TURN_START,
        TURN_END}

log = logging.getLogger(__name__)


class ParserState(agent.parser.ParserState):

    def __init__(self, supported_functions: dict[str, FunctionDefinition] | None = None):
        super().__init__(supported_functions)
        self.prev_token: str | None = None

    def finalize(self, token: str):
        self.prev_token = token

    def get_prev_token(self) -> str | None:
        prev_token = self.prev_token
        if prev_token:
            prev_token = prev_token.strip()
        return prev_token


def parse_name(parameters_block) -> tuple[str | None, str | None]:
    pattern = r"(.*?)({.*})"
    match = re.search(pattern, parameters_block, re.DOTALL)
    if match:
        name = match.group(1).strip()
        tail = match.group(2).strip()
        return name, tail
    else:
        return None, None


def get_arguments(arguments_block: str, expected_parameters: dict[str, Any] | None = None,
                  is_block_partial: bool = False) -> tuple[str, bool]:
    arguments_block = arguments_block.strip()
    if not arguments_block:
        return "{}", is_block_partial

    if arguments_block.startswith("{"):
        arguments_block = arguments_block[1:]

    if arguments_block.endswith("}"):
        arguments_block = arguments_block[:-1]

    partial = is_block_partial

    pattern = r"(\w+):(?:<\|\"\|>)?(.*?)(?:<\|\"\|>)?(?=,|\s|$)"
    kv_pairs = re.findall(pattern, arguments_block)
    structured_parameter = {key: value for key, value in kv_pairs}
    arguments_str = json.dumps(structured_parameter, ensure_ascii=False)
    return arguments_str, partial


class Gemma4ChannelParser(Parser[ParserState]):
    def new_state(self, init_chat_events=True) -> ParserState:
        state = super().new_state()
        if init_chat_events:
            state.start_event(StateEvent.CONVERSATION)
        return state

    def _new_state(self) -> ParserState:
        return ParserState()

    def is_erase(self, state: ParserState, token: str) -> bool:
        return super().is_erase(state, token) or token in spec

    def is_think_end(self, state: ParserState, token: str) -> bool:
        return token.strip() == CHANNEL_END  # and state.has_event(StateEvent.THINK)

    def is_think_start(self, state: ParserState, token: str) -> bool:
        return state.get_prev_token() == CHANNEL_START and token.strip() == THOUGHT

    def is_conversation_start(self, state: ParserState, token: str) -> tuple[bool, str]:
        return _is_conversation_start(TURN_START, token)

    def is_conversation_end(self, state: ParserState, token: str) -> bool:
        return TURN_END == token.strip()

    def is_tool_call_start(self, state: ParserState, token: str) -> bool:
        return token.strip().startswith(TOOL_CALL_START)

    def is_tool_call_end(self, state: ParserState, token: str) -> bool:
        return TOOL_CALL_END == token.strip()

    def is_tool_response_start(self, state: ParserState, token: str) -> bool:
        return TOOL_RESPONSE_START == token.strip()

    def is_tool_response_end(self, state: ParserState, token: str) -> bool:
        return TOOL_RESPONSE_END == token.strip()

    def is_prompt_start_thinking(self, prompt: str) -> bool:
        return False
        # return prompt.endswith(REASONING_START, 0, len(prompt) - 1) if prompt.endswith(
        #     '\n') else prompt.endswith(REASONING_START)

    def is_assistant(self, role):
        return ROLE == role

    def get_assistant_role_name(self) -> str:
        return ROLE

    def parse_tool_calls(self, state: ParserState, tool_call_expression: str) -> tuple[list[ToolCall], bool]:
        tool_call_expression = tool_call_expression.lstrip()
        tool_call_blocks = tool_call_expression.split(TOOL_CALL_START)

        parsed_calls: list[ToolCall] = []
        partial = False
        for call_block in tool_call_blocks:
            if len(call_block) == 0:
                continue

            call_block_rstrip = call_block.rstrip()
            if call_block_rstrip.endswith(TOOL_CALL_END):
                call_block = call_block_rstrip[:-len(TOOL_CALL_END)]

            call_block = call_block.lstrip()
            function_blocks = call_block.split(FUNCTION_START_PREF)
            for function_block in function_blocks:
                if len(function_block) == 0:
                    continue
                function_block = function_block.rstrip()

                # if function_block_rstrip.endswith(FUNCTION_END):
                #     function_block = function_block_rstrip[:-len(FUNCTION_END)]

                func_name, tail = parse_name(function_block)
                if func_name is None:
                    # log
                    continue

                supported_functions = state.supported_functions
                function = supported_functions.get(func_name) if supported_functions else None

                parameters = function.parameters if function is not None else []
                arguments, partial_param = get_arguments(tail, parameters)
                if partial_param:
                    partial = True

                parsed_calls.append(new_tool_call(FunctionCall(name=func_name, arguments=arguments)))
        return parsed_calls, partial
