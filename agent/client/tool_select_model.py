"""Abstract base for tools that synthesize a user-selection tool call.

Leaf module: imported by both agent.client.tool_select_options and
agent.client.veai.tool.ask_user_with_options, so it must not import them back.
"""
from enum import Enum

from abc import ABC, abstractmethod

from agent.parser import ParsedFunctionCall


class AgentType(Enum):
    """Supported agent types that can provide user-selection tools."""
    VEAI = "veai"
    OPENCODE = "opencode"


class DetectedSelectOptions:
    """Result of detecting a user-selection tool in tool definitions."""

    def __init__(self, agent: AgentType, tool: "ToolSelectOptions"):  # noqa: F821
        self.agent = agent
        self.tool = tool


class ToolSelectOptions(ABC):  # pylint: disable=too-few-public-methods  # abstract tool interface
    """Tool detected in tool definitions that asks the user to choose options."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name as declared in tool definitions."""

    @abstractmethod
    def new_call(self, question: str, options: list[str], is_multiple_choice: bool = False) -> ParsedFunctionCall:
        """Build a parsed function call asking the user to pick among options."""