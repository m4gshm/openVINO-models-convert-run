"""Call builders and result models for the safe_delete tool."""
from pydantic import BaseModel

from agent.client.veai.tool import Tool
from agent.parser import ParsedFunctionCall

FUNCTION_NAME = "safe_delete"


class SafeDeleteTargets(BaseModel):
    """Single safe-delete target: file path, symbol name and its line."""

    path: str
    name: str
    line: int


class SafeDelete(Tool):
    """Builds safe_delete tool calls from repaired arguments."""

    @property
    def name(self) -> str:
        """Tool name as declared in tool definitions."""
        return FUNCTION_NAME

    @staticmethod
    def new_call(targets: list[dict], force: bool, dry_run: bool) -> ParsedFunctionCall:
        """Build the parsed function call with targets and delete options."""
        return ParsedFunctionCall(name=FUNCTION_NAME, arguments={
            "targets": targets,
            "force": force,
            "dry_run": dry_run,
        })
