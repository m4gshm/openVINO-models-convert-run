"""Call builder for the edit_file tool."""
from typing import Any

from agent.client.veai.tool import Tool
from agent.parser import ParsedFunctionCall

FUNCTION_NAME = "edit_file"


class EditFile(Tool):
    """Builds edit_file tool calls from repaired arguments."""

    @staticmethod
    def name() -> str:
        """Tool name as declared in tool definitions."""
        return FUNCTION_NAME

    @staticmethod
    def new_call(target_file: str, edits: Any, allow_multiple_matches=True) -> ParsedFunctionCall:
        """Build the parsed function call with target file and edits."""
        arguments = {
            "allow_multiple_matches": allow_multiple_matches
        }
        if target_file:
            arguments["target_file"] = target_file
        if edits:
            arguments["edits"] = edits
        return ParsedFunctionCall(name=FUNCTION_NAME, arguments=arguments)
