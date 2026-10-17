"""Abstract base for tools that synthesize a user-selection tool call.

Leaf module: imported by both agent.client.tool_select_options and
agent.client.veai.tool.ask_user_with_options, so it must not import them back.
"""
from abc import ABC, abstractmethod

from agent.parser import ParsedFunctionCall


class ToolSelectOptions(ABC):  # pylint: disable=too-few-public-methods  # abstract tool interface
    """Tool detected in tool definitions that asks the user to choose options."""

    @abstractmethod
    def new_call(self, question: str, options: list[str], is_multiple_choice: bool = False) -> ParsedFunctionCall:
        """Build a parsed function call asking the user to pick among options."""
