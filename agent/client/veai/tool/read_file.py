from typing import Any

from agent.client.veai.tool import Tool
from agent.parser import ParsedFunctionCall

function_name = "read_file"


class ReadFile(Tool):
    @property
    def name(self) -> str:
        return function_name

    @staticmethod
    def new_call(target_file: str, start_line: int | None = 1, end_line: int | None = 1000,
                 line_offset: int | None = None) -> ParsedFunctionCall:
        arguments: dict[str, Any] = {
            "target_file": target_file,
        }
        if line_offset:
            arguments["line_offset"] = line_offset
        else:
            arguments["start_line"] = start_line
            arguments["end_line"] = end_line
        return ParsedFunctionCall(name=function_name, arguments=arguments)
