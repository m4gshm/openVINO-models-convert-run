"""Parser of the Qwen2 tool call syntax: json function definitions in blocks."""
import json
import logging

import json_repair

from agent.parser import ParserState, ParsedFunctionCall, iter_tool_call_blocks
from agent.parser.qwen_base import TOOL_CALL_START, TOOL_CALL_END, QwenBaseParser

log = logging.getLogger(__name__)

FUNCTION_START_PREF = "<function"


class Qwen2Parser(QwenBaseParser):
    """Parser for models emitting ``<tool_call>{json}</tool_call>`` expressions."""

    def parse_tool_calls(self, state: ParserState,
                         tool_call_expression: str | None) -> tuple[list[ParsedFunctionCall], bool]:
        """Parse json-defined tool calls, repairing truncated json on the fly."""
        parsed_calls: list[ParsedFunctionCall] = []
        partial = False
        for function_blocks, is_call_closed in iter_tool_call_blocks(tool_call_expression, TOOL_CALL_START,
                                                                     TOOL_CALL_END, FUNCTION_START_PREF):
            if not is_call_closed:
                partial = True
            for function_block in function_blocks:
                try:
                    parsed_function = json.loads(function_block)
                except json.decoder.JSONDecodeError:
                    # log
                    parsed_function = json_repair.loads(function_block)

                func_name = parsed_function.get("name") if isinstance(parsed_function, dict) else None
                if func_name is None:
                    # log
                    continue

                arguments = parsed_function.get("arguments")

                parsed_calls.append(ParsedFunctionCall(name=func_name, arguments=arguments))
        return parsed_calls, partial
