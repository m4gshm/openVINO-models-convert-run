"""Call builder for the search_file_by_name tool."""
from agent.client.veai.tool import Tool
from agent.parser import ParsedFunctionCall

FUNCTION_NAME = "search_file_by_name"


class SearchFileByName(Tool):
    """Builds search_file_by_name tool calls from repaired arguments."""

    @property
    def name(self) -> str:
        """Tool name as declared in tool definitions."""
        return FUNCTION_NAME

    @staticmethod
    def new_call(glob_pattern: str | None, search_directory: str | None) -> ParsedFunctionCall:
        """Build the parsed function call with pattern and search directory."""
        arguments = {}
        if not glob_pattern is None:
            arguments["glob_pattern"] = glob_pattern
        if not search_directory is None:
            arguments["search_directory"] = search_directory
        return ParsedFunctionCall(name=FUNCTION_NAME, arguments=arguments)
