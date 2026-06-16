import json
from typing import Any

from common.openai_model import FunctionCall
from veai.tool import Tool

function_name = "edit_file"


class EditFile(Tool):
    @property
    def name(self) -> str:
        return function_name

    @staticmethod
    def new_call(target_file: str, edits: dict[str, Any]) -> FunctionCall:
        arguments: dict[str, Any] = {
            "target_file": target_file,
            "edits": edits,
            "allow_multiple_matches": True,
        }
        arguments_str = json.dumps(arguments, ensure_ascii=False)
        return FunctionCall(name=function_name, arguments=arguments_str)
