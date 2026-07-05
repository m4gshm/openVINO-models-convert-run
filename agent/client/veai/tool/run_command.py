from agent.client.veai.tool import Tool
from agent.parser import ParsedFunctionCall

function_name = "run_command"


class RunCommand(Tool):
    @property
    def name(self) -> str:
        return function_name

    @staticmethod
    def new_call(working_directory: str | None, command: str | None, safe_to_run: bool | None = False,
                 is_background: bool | None = False) -> ParsedFunctionCall:
        return ParsedFunctionCall(name=function_name, arguments={
            "working_directory": working_directory,
            "command": command,
            "is_background": is_background or False,
            "safe_to_run": safe_to_run or False,
        })
