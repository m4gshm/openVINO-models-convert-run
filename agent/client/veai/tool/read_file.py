"""Call builder for the read_file tool."""
from typing import Any

from agent.client.veai.tool import Tool
from agent.parser import ParsedFunctionCall

FUNCTION_NAME = "read_file"


class ReadFile(Tool):
    """Builds read_file tool calls from repaired arguments."""

    @property
    def name(self) -> str:
        """Tool name as declared in tool definitions."""
        return FUNCTION_NAME

    @staticmethod
    def new_call(target_file: str, start_line: int | None = 1, end_line: int | None = 1000,
                 line_offset: int | None = None) -> ParsedFunctionCall:
        """Build the parsed function call with target file and line range."""
        arguments: dict[str, Any] = {
            "target_file": target_file,
        }
        if line_offset:
            arguments["line_offset"] = line_offset
        else:
            arguments["start_line"] = start_line
            arguments["end_line"] = end_line
        return ParsedFunctionCall(name=FUNCTION_NAME, arguments=arguments)
