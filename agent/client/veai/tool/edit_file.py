from typing import Any

from agent.client.veai.tool import Tool
from agent.parser import ParsedFunctionCall

function_name = "edit_file"


class EditFile(Tool):
    @staticmethod
    def name() -> str:
        return function_name

    @staticmethod
    def new_call(target_file: str, edits: Any, allow_multiple_matches=True) -> ParsedFunctionCall:
        arguments = {
            "allow_multiple_matches": allow_multiple_matches
        }
        if target_file:
            arguments["target_file"] = target_file
        if edits:
            arguments["edits"] = edits
        return ParsedFunctionCall(name=function_name, arguments=arguments)
