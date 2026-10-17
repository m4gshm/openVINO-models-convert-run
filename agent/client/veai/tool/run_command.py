"""Call builder for the run_command tool."""
from agent.client.veai.tool import Tool
from agent.parser import ParsedFunctionCall

FUNCTION_NAME = "run_command"


class RunCommand(Tool):
    """Builds run_command tool calls from repaired arguments."""

    @property
    def name(self) -> str:
        """Tool name as declared in tool definitions."""
        return FUNCTION_NAME

    @staticmethod
    def new_call(working_directory: str | None, command: str | None, safe_to_run: bool | None = False,
                 is_background: bool | None = False) -> ParsedFunctionCall:
        """Build the parsed function call with command and execution options."""
        return ParsedFunctionCall(name=FUNCTION_NAME, arguments={
            "working_directory": working_directory,
            "command": command,
            "is_background": is_background or False,
            "safe_to_run": safe_to_run or False,
        })
