import ast
import logging

from agent.parser import ParserState, ParsedFunctionCall, Parser, fill_state_by_prompt_tail
from agent.parser.gemma4 import unquote, unescape

TOOL_CALL_START_PROBABLY = "{\n"

log = logging.getLogger(__name__)

TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"


class Lfm2Parser(Parser):
    def new_state(self, prompt: str = "", init_chat_events=True) -> ParserState:
        if not prompt:
            state = super().new_state(prompt, init_chat_events)
        else:
            state = self._new_state()
            fill_state_by_prompt_tail(init_chat_events, prompt, state, self.is_assistant)
        return state

    def is_probably_tool_call_start(self, state: ParserState, token: str) -> bool:
        return TOOL_CALL_START_PROBABLY == token

    def is_tool_call_start(self, state: ParserState, token: str) -> bool:
        return TOOL_CALL_START == token.strip()

    def is_tool_call_end(self, state: ParserState, token: str) -> bool:
        return TOOL_CALL_END == token.strip()

    def parse_tool_calls(self, state: ParserState, tool_call_expression: str) -> tuple[list[ParsedFunctionCall], bool]:
        tool_call_expression = tool_call_expression.lstrip()
        tool_call_blocks = tool_call_expression.split(TOOL_CALL_START)

        parsed_calls: list[ParsedFunctionCall] = []
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

            parsed_function_call = parse_function_call(function_block)

            if parsed_function_call:
                parsed_calls.append(parsed_function_call)
        return parsed_calls, partial


def parse_function_call(function_block: str) -> ParsedFunctionCall:
    clean_function_block = function_block.strip("[]")

    # Парсим строку в абстрактное синтаксическое дерево
    stop = False
    cycle = 0
    while not stop:
        try:
            cycle += 1
            tree = ast.parse(clean_function_block, mode="eval")
            stop = True
        except SyntaxError as e:
            log.debug(f"parsing error: message='{e.msg}' line={e.lineno}, offset={e.offset}, trying to fix")
            if e.msg == "':' expected after dictionary key":
                clean_function_block = clean_function_block[0:e.offset] + ":" + clean_function_block[e.offset:]
            else:
                raise e

    body = tree.body
    if not isinstance(body, ast.Call):
        raise ValueError(f"unexpected ast parse result type '{type(body)}'")

    func: ast.Name = body.func
    func_name = func.id

    arguments = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in body.keywords
    }

    has_anonymous = True

    body_args = body.args
    if not arguments and body_args:
        first = body_args[0]
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

    anonymous_arguments = [ast.literal_eval(arg) for arg in body_args] if has_anonymous else []

    return ParsedFunctionCall(name=func_name, arguments=arguments, anonymous_arguments=anonymous_arguments)
