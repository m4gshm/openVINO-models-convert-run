"""Call builder for the run_configuration tool."""
from agent.client.veai.tool import Tool
from agent.parser import ParsedFunctionCall

FUNCTION_NAME = "run_configuration"


class RunConfiguration(Tool):
    """Builds run_configuration tool calls from repaired arguments."""

    @property
    def name(self) -> str:
        """Tool name as declared in tool definitions."""
        return FUNCTION_NAME

    @staticmethod
    # pylint: disable=too-many-arguments,too-many-positional-arguments  # signature mirrors the IDE run_configuration tool API
    def new_call(target_file: str,
                 configuration_name: str,
                 timeout: int | None,
                 line_number: int,
                 configuration_run_arguments: list[str] | None,
                 configuration_environment_variables: list[str] | None,
                 files_to_collect_coverage: list[str] | None) -> ParsedFunctionCall:
        """Build the parsed function call with run configuration fields."""
        return ParsedFunctionCall(name=FUNCTION_NAME, arguments={
            "target_file": target_file,
            "configuration_name": configuration_name,
            "timeout": timeout,
            "line_number": line_number,
            "configuration_run_arguments": configuration_run_arguments,
            "configuration_environment_variables": configuration_environment_variables,
            "files_to_collect_coverage": files_to_collect_coverage
        })
