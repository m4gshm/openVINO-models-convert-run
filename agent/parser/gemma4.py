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


ARGS_DELIM = ','
VALID_NAME_VAL_DELIM = ":"
INVALID_BUT_POSSIBLE_DELIM = "="
ARRAY_START = "["
ARRAY_END = "]"
OBJECT_START = "{"
OBJECT_END = "}"

VALUE_TAG_WRAPPER = "<|\"|>"
STR_WRAPPERS = [VALUE_TAG_WRAPPER, "\"", "'"]


def parse_object_arguments(arguments_block: str, array_end_expect=False) -> tuple[dict[str, Any], list[str], str]:
    arguments_block = arguments_block.strip()
    if not arguments_block:
        return {}, [], ""

    if arguments_block.startswith(OBJECT_START):
        object_expect = True
        arguments_block = arguments_block[len(OBJECT_START):]
    else:
        object_expect = False

    anonymous_parameters = []

    possible_json = arguments_block.startswith(OBJECT_START)
    if possible_json:
        if object_expect and arguments_block[-1] == OBJECT_END:
            arguments_block = arguments_block[:-1]
        log.debug(f"trying to parse as json: {arguments_block}")
        possible_json_args = arguments_block.replace(VALUE_TAG_WRAPPER, "\"")
        try:
            arguments: dict[str, Any] = json.loads(possible_json_args)
            if not isinstance(arguments, dict):
                log.error(f"unexpected type of args '{type(arguments)}', arguments='{arguments}'")
        except json.decoder.JSONDecodeError as e:
            try:
                arguments = json_repair.loads(possible_json_args)
                if not isinstance(arguments, dict):
                    log.error(f"unexpected type of repaired args '{type(arguments)}', arguments='{arguments}'")
            except Exception as e:
                arguments = {}

        named_parameters: dict[str, Any] = {}
        if arguments:
            for k, v in arguments.items():
                clean_k = clean(k)
                clean_v = strip_unquote_if_wrapped_by_empty_and_quoted(unescape(v))
                named_parameters[clean_k] = cast_value(clean_v)
        else:
            log.error(f"unrepairable json arguments: {arguments_block}")
        unparsed_tail = ""
    else:
        arguments = dict[str, Any]()
        expect_name = object_expect
        expect_value_end_delimiter = None
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
                    if word.startswith(VALUE_TAG_WRAPPER):
                        # wrapped anonymous arg
                        expect_name = False
                        expect_value_end_delimiter = VALUE_TAG_WRAPPER
                        word = word[len(VALUE_TAG_WRAPPER):] + token
                    else:
                        word_parts = word.split(INVALID_BUT_POSSIBLE_DELIM)
                        if len(word_parts) == 2:
                            name = word_parts[0]
                            word = word_parts[1] + token
                            str_wrapper = is_start_by_wrapper(word)
                            if str_wrapper:
                                expect_value_end_delimiter = str_wrapper
                                word = word[len(str_wrapper):]
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
                            arguments[word_parts[0]] = word_parts[1]
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
                elif word or (not word and token != ' '):
                    word += token
            else:
                # expected value
                if not expect_value_end_delimiter and (
                        is_arg_end_by_delim or token == INVALID_BUT_POSSIBLE_DELIM):
                    if name:
                        arguments[name] = word
                    elif word:
                        anonymous_parameters.append(word)
                    expect_name = True
                    name = ''
                    word = ''
                elif token == ARRAY_START:
                    array_tail = arguments_block[next_token_i:]
                    array, next_token_i, on_parse_after_array = parse_array(array_tail)
                    arguments[name] = array
                    name = ''
                    word = ''
                    expect_name = True
                    expect_args_delim = True
                    on_parse = on_parse_after_array
                else:
                    if is_last_token and (is_object_end or is_array_end):
                        pass
                    elif expect_value_end_delimiter:
                        # inside string
                        word += token
                        if is_end_by_wrapper(word, expect_value_end_delimiter):
                            # end of string
                            word = word[:len(word) - len(expect_value_end_delimiter)]
                            if name:
                                arguments[name] = word
                            elif word:
                                anonymous_parameters.append(word)
                            name = ''
                            word = ''
                            expect_value_end_delimiter = None
                            expect_name = True
                            expect_args_delim = True
                    else:
                        if word or not token == ' ':
                            word += token
                        str_wrapper = is_start_by_wrapper(word)
                        if str_wrapper:
                            # start of string
                            expect_value_end_delimiter = str_wrapper
                            word = ''
                        else:
                            # may be string, may be out of string
                            if is_object_end:
                                word = word[:len(word) - len(OBJECT_END)]
                                break
                            elif is_array_end:
                                word = word[:len(word) - len(ARRAY_END)]
                                break
                            else:
                                pass
        unparsed_tail = on_parse[next_token_i:]

        word = word.rstrip() if word and not expect_value_end_delimiter else word
        if name:
            arguments[name] = word
        elif word:
            anonymous_parameters.append(word)

        named_parameters: dict[str, Any] = {}
        for k, v in arguments.items():
            clean_k = strip(k)
            clean_v = unescape(v)
            cast_clean_v = cast_value(clean_v)
            named_parameters[clean_k] = cast_clean_v
            # if clean_v is None:
            #     raise Exception(f"k={k}, " + arguments_block)

        for i, v in enumerate(anonymous_parameters):
            anonymous_parameters[i] = unescape(v)

    log.debug(
        f"tool call object parsed: src='{arguments_block}', named_parameters='{named_parameters}', "
        f"anonymous_parameters='{anonymous_parameters}', unparsed_tail='{unparsed_tail}'")

    return named_parameters, anonymous_parameters, unparsed_tail


def parse_name(parameters_block) -> tuple[str | None, str | None]:
    pattern = r"(.*?)({.*})"
    match = re.search(pattern, parameters_block, re.DOTALL)
    if match:
        name = match.group(1).strip()
        tail = match.group(2).strip()
        return name, tail
    else:
        return None, None


def is_start_by_wrapper(word: str):
    for wrapper in STR_WRAPPERS:
        if word.startswith(wrapper):
            return wrapper
    return None


def is_end_by_wrapper(word: str | Any, wrapper: str) -> bool:
    return word and word.endswith(wrapper) and not word.endswith("\\" + wrapper)


def clean(dirty_val: Any) -> Any:
    if isinstance(dirty_val, str):
        stripped = dirty_val.strip()
        if stripped.startswith("{") or stripped.startswith("\""):
            stripped = clean(stripped[1:len(dirty_val)])
        if stripped.endswith("}") or stripped.endswith("\""):
            stripped = clean(stripped[:-1])
        dirty_val = unescape(stripped)
    return dirty_val


def strip(val: Any):
    if isinstance(val, str):
        if val.startswith(" ") or val.endswith(" "):
            old = val
            new = val.strip()
            log.debug(f"strip '{old}' to '{new}'")
            val = new
    return val


def unquote(val: Any) -> Any:
    if isinstance(val, str):
        if val.startswith("\"") or val.startswith("'"):
            val = val[1:len(val)]
        if val.endswith("\"") or val.startswith("'"):
            val = val[:-1]
    return val


def strip_unquote_if_wrapped_by_empty_and_quoted(val: Any):
    if isinstance(val, str):
        stripped = val
        last_left_empty = -1
        for i, symb in enumerate(val):
            if symb == " ":
                last_left_empty = i
            else:
                break

        left_quoted = None
        next_after_e = last_left_empty + 1
        if stripped and next_after_e < len(stripped):
            symb_next_after_empty = stripped[next_after_e] if stripped else None
            if symb_next_after_empty and (symb_next_after_empty == "\"" or symb_next_after_empty == "'"):
                left_quoted = symb_next_after_empty
                stripped = stripped[next_after_e + 1:]

        if left_quoted:
            stripped = stripped.rstrip()
            symb_last = stripped[len(stripped) - 1] if len(stripped) > 0 else None
            if symb_last and symb_last == left_quoted:
                stripped = stripped[:-1]

        log.debug(f"strip '{val}' to '{stripped}'")
        return stripped
    else:
        return val


unescaped = [("\\\"", "\""), ("\\r\\n", "\n"), ("\r\n", "\n"), ("\\n", "\n"), ("\\t", "\t")]


def unescape(val: str | Any):
    if isinstance(val, str):
        for (old, new) in unescaped:
            if old in val:
                after = val.replace(old, new)
                log.debug(f"unescaped '{old}' by '{new}':\nbefore={val}\nafter={after}")
                val = after
    return val

def cast_value(val: str | Any):
    if isinstance(val, str):
        low_val = val.lower()
        if low_val == "null":
            return None
        elif low_val == "false":
            return False
        elif low_val == "true":
            return True
    return val


# new_line_map = [("\r\n", "\n")]
#
#
# def normalize_new_line(val: str | Any):
#     if isinstance(val, str):
#         for (old, new) in new_line_map:
#             if old in val:
#                 after = val.replace(old, new)
#                 log.debug(f"normalize_new_line '{old}' by '{new}':\nbefore={val}\nafter={after}")
#                 val = after
#     return val


def parse_array(array_str: str) -> tuple[list[dict[str, Any]], int, str]:
    log.debug(f"parse array from '{array_str}'")
    next_token_i = 0
    parsed_object_in_array, parsed_anonymous_in_array, unparsed_tail = parse_object_arguments(
        arguments_block=array_str[next_token_i:], array_end_expect=True)
    array_tail = unparsed_tail

    array: list = [parsed_object_in_array]
    if parsed_anonymous_in_array:
        array.append(parsed_anonymous_in_array)

    while next_token_i < len(array_tail):
        token = array_tail[next_token_i]
        if token == ARRAY_END:
            next_token_i += 1
            array_tail = array_tail[next_token_i:]
            next_token_i = 0
            break
        elif token == OBJECT_END:
            log.warning(f"unexpected object end {token} at index {next_token_i} in array tail '{array_tail}'")
            next_token_i += 1
        else:
            if token == ARGS_DELIM:
                next_token_i += 1
            parsed_object_in_array, parsed_anonymous_in_array, unparsed_tail = parse_object_arguments(
                arguments_block=array_tail[next_token_i:], array_end_expect=True)
            array.append(parsed_object_in_array)
            if parsed_anonymous_in_array:
                array.append(parsed_anonymous_in_array)
            next_token_i = 0
            array_tail = unparsed_tail

    return array, next_token_i, array_tail


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
