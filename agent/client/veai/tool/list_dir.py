import json
from typing import Any

from agent.client.veai.tool import Tool
from agent.openai.chat_completions_api import FunctionCall

function_name = "list_dir"


class ListDir(Tool):
    @property
    def name(self) -> str:
        return function_name

    @staticmethod
    def new_call(directory_path: str, depth: int) -> FunctionCall:
        arguments: dict[str, Any] = {
            "directory_path": directory_path,
            "depth": depth,
        }
        arguments_str = json.dumps(arguments, ensure_ascii=False)
        return FunctionCall(name=function_name, arguments=arguments_str)
