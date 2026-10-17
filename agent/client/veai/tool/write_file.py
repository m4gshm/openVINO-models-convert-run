"""Call builder for the write_file tool."""
from typing import Any

from agent.client.veai.tool import Tool
from agent.parser import ParsedFunctionCall

FUNCTION_NAME = "write_file"


class WriteFile(Tool):
    """Builds write_file tool calls from repaired arguments."""

    @property
    def name(self) -> str:
        """Tool name as declared in tool definitions."""
        return FUNCTION_NAME

    @staticmethod
    def new_call(target_file: str, content: Any, allow_overwrite=True) -> ParsedFunctionCall:
        """Build the parsed function call with target file and new content."""
        return ParsedFunctionCall(name=FUNCTION_NAME, arguments={
            "target_file": target_file,
            "content": content,
            "allow_overwrite": allow_overwrite,
        })
