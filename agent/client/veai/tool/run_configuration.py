from agent.client.veai.tool import Tool
from agent.parser import ParsedFunctionCall

function_name = "run_configuration"


class RunConfiguration(Tool):
    @property
    def name(self) -> str:
        return function_name

    @staticmethod
    def new_call(target_file: str,
                 configuration_name: str,
                 timeout: int,
                 line_number: int,
                 configuration_run_arguments: list[str],
                 configuration_environment_variables: list[str],
                 files_to_collect_coverage: list[str]) -> ParsedFunctionCall:
        return ParsedFunctionCall(name=function_name, arguments={
            "target_file": target_file,
            "configuration_name": configuration_name,
            "timeout": timeout,
            "line_number": line_number,
            "configuration_run_arguments": configuration_run_arguments,
            "configuration_environment_variables": configuration_environment_variables,
            "files_to_collect_coverage": files_to_collect_coverage
        })
