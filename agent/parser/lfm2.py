import ast
import logging
import re
from typing import Any

from agent.inference.phrase import Phrase
from agent.openai.chat_completions_api import FunctionDefinitionParameters
from agent.parser import ParserState, ParsedFunctionCall, Parser, fill_state_by_prompt_tail, DelayedResult
from agent.parser.gemma4 import unescape
from agent.parser.json_fixer import try_to_parse_json, ARRAY_START, OBJECT_START

TOOL_CALL_START_PROBABLY = "{\n"

log = logging.getLogger(__name__)

TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"


class Lfm2Parser(Parser):
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
            except SyntaxError as e:
                log.error(f"unparseable tool call: {function_block}")

        return parsed_calls, partial

    def is_sequence_end(self, state: State, token: str) -> bool:
        sequence_end = super().is_sequence_end(state, token)
        return sequence_end

    def is_delay_streaming_start(self, state: ParserState, token: str) -> bool:
        token_rsplit = token.strip()
        return token_rsplit == ARRAY_START or token_rsplit == OBJECT_START

    def is_delay_streaming_end(self, state: ParserState, token: str) -> bool:
        return False

    def handle_delayed_phrase(self, phrase: Phrase) -> DelayedResult | None:
        full = phrase.full
        parsed_json = try_to_parse_json(full)
        if parsed_json is None:
            return None
        else:
            handled_content: str | None = None
            handled_tool_calls: list[ParsedFunctionCall] | None = None
            if isinstance(parsed_json, dict):
                content = parsed_json.get("content")
                if not content is None:
                    if isinstance(content, str):
                        handled_content = content
                    else:
                        log.warning(f"unexpected content type='{type(content)}', content='{content}'")

                tool_calls = parsed_json.get("tool_calls")
                if not tool_calls is None:
                    if isinstance(tool_calls, list):
                        parsed_tool_calls = list[ParsedFunctionCall]()
                        for tool_call in tool_calls:
                            if not isinstance(tool_call, dict):
                                log.warning(f"unexpected tool_call type='{type(tool_call)}', content='{tool_call}'")
                            else:
                                name = tool_call.get("name")
                                arguments = tool_call.get("arguments")
                                if not (name is None or arguments is None):
                                    if not isinstance(name, str):
                                        log.warning(f"unexpected tool_call name type='{type(name)}', content='{name}'")

                                    if not isinstance(arguments, dict):
                                        log.warning(f"unexpected tool_call arguments type='{type(arguments)}',"
                                                    f" content='{arguments}'")
                                parsed_tool_calls.append(ParsedFunctionCall(name=name, arguments=arguments))
                        if len(parsed_tool_calls) > 0:
                            handled_tool_calls = parsed_tool_calls
                    else:
                        log.warning(f"unexpected tool_calls type='{type(tool_calls)}', content='{tool_calls}'")

            return DelayedResult(content=handled_content, tool_calls=handled_tool_calls) if (
                    handled_content or handled_tool_calls) else None


def parse_function_call(function_block: str) -> list[ParsedFunctionCall]:
    clean_function_block = function_block.strip("[]")
    tree: ast.Expression | None = None
    stop = False
    cycle = 0
    while not stop:
        try:
            cycle += 1
            tree: ast.Expression = ast.parse(clean_function_block, mode="eval")
            stop = True
        except SyntaxError as e:
            offset = e.offset
            log.debug(f"parsing error: message='{e.msg}' line={e.lineno}, offset={offset}, trying to fix")
            if e.msg == "':' expected after dictionary key":
                start_str = ""
                if offset > 0:
                    prev = clean_function_block[offset - 1]
                    if prev == '\'' or prev == '"':
                        start_str = prev
                clean_function_block = insert_str(clean_function_block, ":" + start_str, offset)
            else:
                pattern = r"closing parenthesis '(?P<closing>.)' does not match opening parenthesis '(?P<opening>.)'"
                match = re.search(pattern, e.msg)
                if match:
                    closing_bracket = match.group("closing")
                    opening_bracket = match.group("opening")
                    if opening_bracket == "{":
                        new_closing_bracket = "}"
                    elif opening_bracket == "[":
                        new_closing_bracket = "]"
                    else:
                        new_closing_bracket = None
                    if not new_closing_bracket is None:
                        clean_function_block = insert_str(clean_function_block, new_closing_bracket, offset - 1)
                    else:
                        raise e
                else:
                    raise e

    body = tree.body

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
                for i, k in enumerate(first.keys):
                    v = first.values[i]
                    if not isinstance(k, ast.Constant):
                        log.error(f"unexpected key type={type(k)}, key={k}")
                    elif not isinstance(v, ast.Constant):
                        log.error(f"unexpected value type {type(v)}, value={v}")
                    else:
                        arguments[k.value] = v.value

                has_anonymous = not arguments
        anonymous_arguments = [ast.literal_eval(arg) for arg in element_args] if has_anonymous else []
        result.append(ParsedFunctionCall(name=func_name, arguments=arguments, anonymous_arguments=anonymous_arguments))

    return result


def insert_str(clean_function_block: str | Any, new_closing_bracket: str, offset: int | None) -> Any:
    clean_function_block = clean_function_block[0:offset] + new_closing_bracket + clean_function_block[offset:]
    return clean_function_block
