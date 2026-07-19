import json
import logging
import re
from typing import Any

import json_repair

import agent
from agent.openai.chat_completions_api import FunctionDefinition
from agent.parser import Parser, _is_conversation_start, ParsedFunctionCall

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


def get_arguments(arguments_block: str, is_block_partial: bool = False) -> tuple[dict[str, Any], list[str], bool]:
    arguments_block = arguments_block.strip()
    if not arguments_block:
        return {}, [], is_block_partial

    if arguments_block.startswith("{"):
        arguments_block = arguments_block[1:]

    if arguments_block.endswith("}"):
        arguments_block = arguments_block[:-1]

    partial = is_block_partial

    value_tag_wrapper = "<|\"|>"

    named_parameters = {}
    anonymous_parameters = []

    if arguments_block.startswith("{") and arguments_block.endswith("}"):
        log.debug(f"trying to parse as json: {arguments_block}")
        possible_json_args = arguments_block.replace(value_tag_wrapper, "\"")
        try:
            arguments: dict[str, Any] = json.loads(possible_json_args)
        except json.decoder.JSONDecodeError as e:
            try:
                arguments = json_repair.loads(possible_json_args)
            except Exception as e:
                arguments = {}
        if not arguments:
            named_parameters: dict[str, Any] = {}
            log.error(f"unrepairable json arguments: {arguments_block}")
        else:
            named_parameters = arguments
    else:
        expect_name = True
        expect_end_delimiter = False
        expect_kv_delim = False
        word = ''
        name: str | None = None

        valid_kv_delim = ','
        valid_name_val_delim = ":"
        invalid_but_possible_delim = "="

        for i, token in enumerate(arguments_block):
            if expect_name:
                if token == valid_name_val_delim:
                    if word.startswith(value_tag_wrapper):
                        # wrapped anonymous arg
                        expect_name = False
                        expect_end_delimiter = True
                        word = word[len(value_tag_wrapper):] + token
                    else:
                        word_parts = word.split(invalid_but_possible_delim)
                        if len(word_parts) == 2:
                            name = word_parts[0]
                            word = word_parts[1] + token
                            expect_name = False
                        else:
                            expect_name = False
                            name = word
                            word = ''

                elif token == valid_kv_delim or i == len(arguments_block) - 1:
                    if expect_kv_delim:
                        expect_kv_delim = False
                    else:
                        word_parts = word.split(invalid_but_possible_delim)
                        if len(word_parts) == 2:
                            named_parameters[word_parts[0]] = word_parts[1]
                        elif word:
                            anonymous_parameters.append(word)
                        name = ''
                        word = ''
                else:
                    word += token
            else:
                # expected value
                if not expect_end_delimiter and (token == valid_kv_delim or token == invalid_but_possible_delim):
                    expect_name = True
                    if name:
                        named_parameters[name] = word
                    elif word:
                        anonymous_parameters.append(word)
                    name = ''
                    word = ''
                else:
                    word += token
                    if word.endswith(value_tag_wrapper):
                        if not expect_end_delimiter:
                            expect_end_delimiter = True
                            word = ''
                        else:
                            value = word[:len(word) - len(value_tag_wrapper)]
                            if name:
                                named_parameters[name] = value
                            elif word:
                                anonymous_parameters.append(value)
                            name = ''
                            word = ''
                            expect_end_delimiter = False
                            expect_name = True
                            expect_kv_delim = True

        if name:
            named_parameters[name] = word
        elif word:
            anonymous_parameters.append(word)

        for k, v in named_parameters.items():
            if v and v.startswith("\"") and v.endswith("\""):
                named_parameters[k] = v[1:-1]

        for i, v in enumerate(anonymous_parameters):
            if v and v.startswith("\"") and v.endswith("\""):
                anonymous_parameters[i] = v[1:-1]

    log.debug(
        f"tool call arguments parsed: src={arguments_block}, named_parameters={named_parameters}, partial={partial}"
        f"anonymous_parameters={anonymous_parameters}")
    return named_parameters, anonymous_parameters, partial


class Gemma4ChannelParser(Parser[ParserState]):
    def new_state(self, init_chat_events=True) -> ParserState:
        state = super().new_state()
        return state

    def process_chat_prompt(self, prompt: str) -> str:
        expected = f"{TURN_START}{self.get_assistant_role_name()}\n"
        if not prompt.endswith(expected):
            log.debug(f"parser appends prompt by {expected}")
            prompt += expected
        return prompt

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

    def parse_tool_calls(self, state: ParserState, tool_call_expression: str) -> tuple[list[ParsedFunctionCall], bool]:
        tool_call_expression = tool_call_expression.lstrip()
        tool_call_blocks = tool_call_expression.split(TOOL_CALL_START)

        parsed_calls: list[ParsedFunctionCall] = []
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

                func_name, tail = parse_name(function_block)
                if func_name is None:
                    # log
                    continue

                arguments, anonymous_parameters, partial_param = get_arguments(tail or "")
                if partial_param:
                    partial = True

                parsed_calls.append(
                    ParsedFunctionCall(name=func_name, arguments=arguments, anonymous_arguments=anonymous_parameters))
        return parsed_calls, partial
