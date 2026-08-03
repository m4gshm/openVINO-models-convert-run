import json
import logging
import re
from typing import Any, LiteralString

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
EOS = "<eos>"

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


ARGS_DELIM = ','
VALID_NAME_VAL_DELIM = ":"
INVALID_BUT_POSSIBLE_DELIM = "="
ARRAY_START = "["
ARRAY_END = "]"
OBJECT_START = "{"
OBJECT_END = "}"


def parse_object_arguments(arguments_block: str) -> tuple[
    dict[str, Any], list[str], str]:
    arguments_block = arguments_block.strip()
    if not arguments_block:
        return {}, [], ""

    object_expect = False

    if arguments_block.startswith(OBJECT_START):
        object_expect = True
        arguments_block = arguments_block[1:]
    else:
        raise Exception(f"expected object, {arguments_block}")

    value_tag_wrapper = "<|\"|>"

    named_parameters = {}
    anonymous_parameters = []

    possible_json = arguments_block.startswith(OBJECT_START)
    if possible_json:  # and arguments_block[:possible_json_object_close_tag_position].endswith(object_end):
        if object_expect:
            arguments_block = arguments_block[:-1]
        log.debug(f"trying to parse as json: {arguments_block}")
        possible_json_args = arguments_block.replace(value_tag_wrapper, "\"")
        try:
            arguments: dict[str, Any] = json.loads(possible_json_args)
        except json.decoder.JSONDecodeError as e:
            try:
                arguments = json_repair.loads(possible_json_args)
            except Exception as e:
                arguments = {}
        named_parameters: dict[str, Any] = {}
        if not arguments:
            log.error(f"unrepairable json arguments: {arguments_block}")
        else:
            for k, v in arguments.items():
                clean_k = clean(k)
                clean_v = clean(v)
                named_parameters[clean_k] = clean_v
        unparsed_tail = ""
    else:
        array_end_expect = False
        expect_name = object_expect
        expect_value_end_delimiter = False
        expect_args_delim = False
        word = ''
        name: str | None = None

        on_parse = arguments_block
        next_token_i = 0
        while next_token_i < len(on_parse):
            token = on_parse[next_token_i]
            last_token_i = len(on_parse) - 1
            is_last_token = next_token_i == last_token_i
            next_token_i += 1
            is_arg_end_by_delim = token == ARGS_DELIM
            is_array_end = array_end_expect and token == ARRAY_END
            is_object_end = object_expect and token == OBJECT_END
            if expect_name:
                if token == VALID_NAME_VAL_DELIM:
                    if word.startswith(value_tag_wrapper):
                        # wrapped anonymous arg
                        expect_name = False
                        expect_value_end_delimiter = True
                        word = word[len(value_tag_wrapper):] + token
                    else:
                        word_parts = word.split(INVALID_BUT_POSSIBLE_DELIM)
                        if len(word_parts) == 2:
                            name = word_parts[0]
                            word = word_parts[1] + token
                            expect_name = False
                        else:
                            expect_name = False
                            name = word
                            word = ''
                elif is_arg_end_by_delim or is_object_end or is_array_end or is_last_token:
                    if not (is_arg_end_by_delim or is_object_end or is_array_end):
                        word += token
                    if expect_args_delim:
                        expect_args_delim = False
                    else:
                        word_parts = word.split(INVALID_BUT_POSSIBLE_DELIM)
                        if len(word_parts) == 2:
                            named_parameters[word_parts[0]] = word_parts[1]
                        elif word:
                            anonymous_parameters.append(word)
                        name = ''
                        word = ''
                    if is_object_end:
                        break
                    # if is_array_end:
                    #     break

                # elif object_expect and token == object_end:
                #     # log
                #     break
                else:
                    word += token
            else:
                # expected value
                if not expect_value_end_delimiter and (
                        is_arg_end_by_delim or token == INVALID_BUT_POSSIBLE_DELIM):
                    expect_name = True
                    if name:
                        named_parameters[name] = word
                    elif word:
                        anonymous_parameters.append(word)
                    name = ''
                    word = ''
                elif token == ARRAY_START:
                    array, next_token_i, on_parse = parse_array(arguments_block, next_token_i)
                    named_parameters[name] = array
                    name = ''
                    word = ''
                    expect_name = True
                    expect_args_delim = True
                else:
                    if not (is_last_token and (is_object_end or is_array_end)):
                        word += token
                        if word.endswith(value_tag_wrapper):
                            if not expect_value_end_delimiter:
                                expect_value_end_delimiter = True
                                word = ''
                            else:
                                value = word[:len(word) - len(value_tag_wrapper)]
                                if name:
                                    named_parameters[name] = value
                                elif word:
                                    anonymous_parameters.append(value)
                                name = ''
                                word = ''
                                expect_value_end_delimiter = False
                                expect_name = True
                                expect_args_delim = True
                    else:
                        pass
        unparsed_tail = on_parse[next_token_i:]

        if name:
            named_parameters[name] = word
        elif word:
            anonymous_parameters.append(word)

        for k, v in named_parameters.items():
            if v and isinstance(v, str):
                named_parameters[k] = unquote(v)
            else:
                pass

        for i, v in enumerate(anonymous_parameters):
            if v and isinstance(v, str):
                anonymous_parameters[i] = unquote(v)
            else:
                pass

    log.debug(
        f"tool call object parsed: src={arguments_block}, named_parameters={named_parameters}, "
        f"anonymous_parameters={anonymous_parameters}")

    return named_parameters, anonymous_parameters, unparsed_tail


def clean(dirty_val: Any) -> Any:
    if isinstance(dirty_val, str):
        stripped = dirty_val.strip()
        if stripped.startswith("{") or stripped.startswith("\""):
            stripped = clean(stripped[1:len(dirty_val)])
        if stripped.endswith("}") or stripped.endswith("\""):
            stripped = clean(stripped[:-1])

        dirty_val = stripped

        dirty_val = fix_escaped_symbol(dirty_val)
    return dirty_val


def unquote(val: Any) -> Any:
    if isinstance(val, str):
        if val.startswith("\""):
            val = clean(val[1:len(val)])
        if val.endswith("\""):
            val = clean(val[:-1])
        val = fix_escaped_symbol(val)
    return val


def fix_escaped_symbol(val: str | Any) -> LiteralString:
    if "\\\"" in val:
        fix_escaped = val.replace( "\\\"", "\"")
        log.debug(f"fix escaped:\nold={val}\nnew={fix_escaped}")
        val = fix_escaped

    if "\\r\\n" in val:
        new_lines_dirty_val = val.replace("\\r\\n", "\r\n")
        log.debug(f"fix new lines windows:\nold={val}\nnew={new_lines_dirty_val}")
        val = new_lines_dirty_val
    elif "\\n" in val:
        new_lines_dirty_val = val.replace("\\n", "\n")
        log.debug(f"fix new lines linux:\nold={val}\nnew={new_lines_dirty_val}")
        val = new_lines_dirty_val
    return val


def parse_array(arguments_block: str, next_token_i: int) -> tuple[list[dict[str, Any]], int, str]:
    array_tail = arguments_block[next_token_i:]
    parsed_object_in_array, parsed_anonymous_in_array, unparsed_tail = parse_object_arguments(
        arguments_block=array_tail)
    next_token_i = 0
    on_parse = unparsed_tail

    array = [parsed_object_in_array]

    while next_token_i < len(on_parse):
        token = on_parse[next_token_i]
        next_token_i += 1
        if token == ARGS_DELIM:
            parsed_object_in_array, parsed_anonymous_in_array, unparsed_tail = parse_object_arguments(
                arguments_block=array_tail)
            array.append(parsed_object_in_array)
            next_token_i = 0
            on_parse = unparsed_tail
        elif token == ARRAY_END:
            on_parse = on_parse[next_token_i:]
            next_token_i = 0
            break
    return array, next_token_i, on_parse


class Gemma4ChannelParser(Parser[ParserState]):
    def new_state(self, prompt: str = "", init_chat_events=True) -> ParserState:
        state = super().new_state(prompt, init_chat_events)
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

    def is_sequence_end(self, state: ParserState, token: str) -> bool:
        return EOS == token.strip()

    def is_text_end(self, state: ParserState, token: str) -> bool:
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

                arguments, anonymous_parameters, unparsed_tail = parse_object_arguments(tail or "")

                parsed_calls.append(
                    ParsedFunctionCall(name=func_name, arguments=arguments, anonymous_arguments=anonymous_parameters))
        return parsed_calls, partial
