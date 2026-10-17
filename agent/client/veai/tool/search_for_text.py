"""Call builder for the search_for_text tool."""
from agent.client.veai.tool import Tool
from agent.parser import ParsedFunctionCall

FUNCTION_NAME = "search_for_text"


class SearchForText(Tool):
    """Builds search_for_text tool calls from repaired arguments."""

    @property
    def name(self) -> str:
        """Tool name as declared in tool definitions."""
        return FUNCTION_NAME

    @staticmethod
    def new_call(target_path_or_url: str, text_snippet: str, is_case_sensitive: bool = False) -> ParsedFunctionCall:
        """Build the parsed function call with search target and snippet."""
        return ParsedFunctionCall(name=FUNCTION_NAME, arguments={
            "target_path_or_url": target_path_or_url,
            "text_snippet": text_snippet,
            "is_case_sensitive": is_case_sensitive,
        })
