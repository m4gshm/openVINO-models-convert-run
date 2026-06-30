from typing import Any

from agent.client.veai.tool import Tool
from agent.parser import ParsedFunctionCall

function_name = "edit_file"


class EditFile(Tool):
    @property
    def name(self) -> str:
        return function_name

    @staticmethod
    def new_call(target_file: str, edits: Any, allow_multiple_matches=True) -> ParsedFunctionCall:
        return ParsedFunctionCall(name=function_name, arguments={
            "target_file": target_file,
            "edits": edits,
            "allow_multiple_matches": allow_multiple_matches,
        })
