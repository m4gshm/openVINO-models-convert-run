"""Call builder for the file_structure tool."""
from agent.client.veai.tool import Tool
from agent.parser import ParsedFunctionCall

FUNCTION_NAME = "file_structure"


class FileStructure(Tool):
    """Builds file_structure tool calls from repaired arguments."""

    @property
    def name(self) -> str:
        """Tool name as declared in tool definitions."""
        return FUNCTION_NAME

    @staticmethod
    def new_call(target_file: str) -> ParsedFunctionCall:
        """Build the parsed function call with the target file path."""
        return ParsedFunctionCall(name=FUNCTION_NAME, arguments={
            "target_file": target_file,
        })
