import json
from typing import Any

from agent.client.veai.tool import Tool
from agent.openai.chat_completions_api import FunctionCall

function_name = "edit_file"


class EditFile(Tool):
    @property
    def name(self) -> str:
        return function_name

    @staticmethod
    def new_call(target_file: str, edits: Any, allow_multiple_matches=True) -> FunctionCall:
        args: dict[str, Any] = {
            "target_file": target_file,
            "edits": edits,
            "allow_multiple_matches": allow_multiple_matches,
        }
        arguments_str = json.dumps(args, ensure_ascii=False)
        return FunctionCall(name=function_name, arguments=arguments_str)
