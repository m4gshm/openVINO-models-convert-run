import ast
import logging

from agent.parser import ParserState, ParsedFunctionCall, Parser, StateEvent, fill_state_by_prompt_tail

log = logging.getLogger(__name__)

TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"


class Lfm2Parser(Parser):
    def new_state(self, prompt: str = "", init_chat_events=True) -> ParserState:
        if not prompt:
            state = super().new_state(prompt, init_chat_events)
            if init_chat_events:
                state.start_event(StateEvent.THINK)
        else:
            state = self._new_state()
            fill_state_by_prompt_tail(init_chat_events, prompt, state, self.is_assistant)
        return state

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
            if len(call_block) == 0:
                continue

            call_block_rstrip = call_block.rstrip()
            if call_block_rstrip.endswith(TOOL_CALL_END):
                call_block = call_block_rstrip[:-len(TOOL_CALL_END)]

            call_block = call_block.lstrip()
            function_block = call_block

            parsed_function_call = parse_function_call(function_block)

            if parsed_function_call:
                parsed_calls.append(parsed_function_call)
        return parsed_calls, partial


def parse_function_call(function_block: str) -> ParsedFunctionCall:
    clean_function_block = function_block.strip("[]")

    # Парсим строку в абстрактное синтаксическое дерево
    tree = ast.parse(clean_function_block, mode="eval")

    body = tree.body
    if not isinstance(body, ast.Call):
        raise ValueError(f"unexpected ast parse result type '{type(body)}'")

    func: ast.Name = body.func
    func_name = func.id

    arguments = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in body.keywords
    }

    anonymous_arguments = [ast.literal_eval(arg) for arg in body.args]

    return ParsedFunctionCall(name=func_name, arguments=arguments, anonymous_arguments=anonymous_arguments)
