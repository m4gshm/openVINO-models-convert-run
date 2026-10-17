"""Parser of the LFM2 python-tuple tool call syntax and delayed json phrases."""
import ast
import logging
import re
from typing import Any

from agent.inference.phrase import Phrase
from agent.openai.chat_completions_api import FunctionDefinitionParameters
from agent.parser import ParserState, ParsedFunctionCall, Parser, fill_state_by_prompt_tail, DelayedResult
from agent.parser.gemma4 import unescape
from agent.parser.json_fixer import ARRAY_START, OBJECT_START, try_to_parse_json

TOOL_CALL_START_PROBABLY = "{\n"

log = logging.getLogger(__name__)

TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"


class Lfm2Parser(Parser):
    """Parser for LFM2 models emitting python-tuple style tool calls."""

    def new_state(self, prompt: str = "", supported_functions: dict[str, FunctionDefinitionParameters] | None = None,
                  init_chat_events=True) -> ParserState:
        if not prompt:
            state = super().new_state(prompt, supported_functions, init_chat_events)
        else:
            state = self._new_state(supported_functions)
            fill_state_by_prompt_tail(init_chat_events, prompt, state, self.is_assistant)
        return state

    # def is_probably_tool_call_start(self, state: ParserState, token: str) -> bool:
    #     return TOOL_CALL_START_PROBABLY == token

    def is_tool_call_start(self, state: ParserState, token: str) -> bool:
        return TOOL_CALL_START == token.strip()

    def is_tool_call_end(self, state: ParserState, token: str) -> bool:
        return TOOL_CALL_END == token.strip()

    def parse_tool_calls(self, state: ParserState, tool_call_expression: str) -> tuple[list[ParsedFunctionCall], bool]:
        tool_call_expression = tool_call_expression.lstrip()
        tool_call_blocks = tool_call_expression.split(TOOL_CALL_START)

        parsed_calls = list[ParsedFunctionCall]()
        partial = False
        for call_block in tool_call_blocks:
            call_block_rstrip = call_block.rstrip()
            if call_block_rstrip.endswith(TOOL_CALL_END):
                call_block = call_block_rstrip[:-len(TOOL_CALL_END)]

            call_block = call_block.lstrip()
            if call_block.startswith(TOOL_CALL_START_PROBABLY):
                call_block = call_block[len(TOOL_CALL_START_PROBABLY):].strip()

            if call_block.startswith("\"") or call_block.startswith("'"):
                call_block = call_block[1:].strip()
                call_block = unescape(call_block)

            function_block = call_block

            if len(function_block) == 0:
                continue

            try:
                parsed_function_calls = parse_function_call(function_block)
                if parsed_function_calls:
                    parsed_calls.extend(parsed_function_calls)
            except SyntaxError:
                log.error("unparseable tool call: %s", function_block)

        return parsed_calls, partial

    def is_sequence_end(self, state: ParserState, token: str) -> bool:
        sequence_end = super().is_sequence_end(state, token)
        return sequence_end

    def is_delay_streaming_start(self, state: ParserState, token: str) -> bool:
        token_rsplit = token.strip()
        return token_rsplit in (ARRAY_START, OBJECT_START)

    def is_delay_streaming_end(self, state: ParserState, token: str) -> bool:
        return False

    def handle_delayed_phrase(self, phrase: Phrase) -> DelayedResult | None:
        full = phrase.full
        return parse_delayed_phrase(full)


def parse_delayed_phrase(delayed_raw_json: str) -> DelayedResult | None:  # pylint: disable=too-many-branches  # shape checks
    """Parse a delayed json phrase into content and tool calls, warning on mismatches."""
    parsed_json = try_to_parse_json(delayed_raw_json)
    if parsed_json is None:
        return None

    handled_content: str | None = None
    handled_tool_calls: list[ParsedFunctionCall] | None = None
    if not isinstance(parsed_json, dict):
        return None

    content = parsed_json.get("content")
    if not content is None:
        if isinstance(content, str):
            handled_content = content
        else:
            log.warning("unexpected content type='%s', content='%s'", type(content), content)

    tool_calls = parsed_json.get("tool_calls")
    if tool_calls is None:
        pass
    elif not isinstance(tool_calls, list):
        log.warning("unexpected tool_calls type='%s', content='%s'", type(tool_calls), tool_calls)
    else:
        parsed_tool_calls = list[ParsedFunctionCall]()
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                log.warning("unexpected tool_call type='%s', content='%s'", type(tool_call), tool_call)
            else:
                name = tool_call.get("name")
                arguments = tool_call.get("arguments")
                if not (name is None or arguments is None):
                    if not isinstance(name, str):
                        log.warning("unexpected tool_call name type='%s', content='%s'", type(name), name)

                    if not isinstance(arguments, dict):
                        log.warning("unexpected tool_call arguments type='%s', content='%s'",
                                    type(arguments), arguments)
                parsed_tool_calls.append(ParsedFunctionCall(name=name, arguments=arguments))
        if len(parsed_tool_calls) > 0:
            handled_tool_calls = parsed_tool_calls

    return DelayedResult(content=handled_content, tool_calls=handled_tool_calls) if (
            handled_content or handled_tool_calls) else None


def _repaired_tuple_block(clean_function_block: str) -> ast.Expression:
    """Ast-parse a tuple expression, iteratively repairing missing colons and brackets."""
    while True:
        try:
            return ast.parse(clean_function_block, mode="eval")
        except SyntaxError as e:
            offset = e.offset
            log.debug("parsing error: message='%s' line=%s, offset=%s, trying to fix", e.msg, e.lineno, offset)
            clean_function_block = _repair_tuple_syntax(clean_function_block, e, offset)


def _repair_tuple_syntax(clean_function_block: str, error: SyntaxError, offset: int | None) -> str:
    """Insert the colon or closing bracket requested by the ast error, or re-raise."""
    if error.msg == "':' expected after dictionary key":
        start_str = ""
        if offset > 0:
            prev = clean_function_block[offset - 1]
            if prev in ('\'', '"'):
                start_str = prev
        return insert_str(clean_function_block, ":" + start_str, offset)
    pattern = r"closing parenthesis '(?P<closing>.)' does not match opening parenthesis '(?P<opening>.)'"
    match = re.search(pattern, error.msg)
    if not match:
        raise error
    opening_bracket = match.group("opening")
    if opening_bracket == "{":
        return insert_str(clean_function_block, "}", offset - 1)
    if opening_bracket == "[":
        return insert_str(clean_function_block, "]", offset - 1)
    raise error


def parse_function_call(function_block: str) -> list[ParsedFunctionCall]:
    """Parse an LFM2 ``func(arg=value, ...)`` tuple into calls, repairing broken syntax."""
    clean_function_block = function_block.strip("[]")
    body = _repaired_tuple_block(clean_function_block).body

    elements = body.elts if isinstance(body, ast.Tuple) else [body]

    result: list[ParsedFunctionCall] = []
    for element in elements:
        if not isinstance(element, ast.Call):
            raise ValueError(f"unexpected ast parse result type '{type(element)}'")

        func: ast.Name = element.func
        func_name = func.id

        arguments = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in element.keywords
        }

        has_anonymous = True

        element_args = element.args
        if not arguments and element_args:
            first = element_args[0]
            if isinstance(first, ast.Dict):
                for i, key in enumerate(first.keys):
                    value = first.values[i]
                    if not isinstance(key, ast.Constant):
                        log.error("unexpected key type=%s, key=%s", type(key), key)
                    elif not isinstance(value, ast.Constant):
                        log.error("unexpected value type %s, value=%s", type(value), value)
                    else:
                        arguments[key.value] = value.value

                has_anonymous = not arguments
        anonymous_arguments = [ast.literal_eval(arg) for arg in element_args] if has_anonymous else []
        result.append(ParsedFunctionCall(name=func_name, arguments=arguments, anonymous_arguments=anonymous_arguments))

    return result


def insert_str(clean_function_block: str | Any, new_closing_bracket: str, offset: int | None) -> Any:
    """Insert a fragment into the block at the given offset."""
    clean_function_block = clean_function_block[0:offset] + new_closing_bracket + clean_function_block[offset:]
    return clean_function_block
