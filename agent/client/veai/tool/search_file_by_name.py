import json
from typing import Any

from agent.client.veai.tool import Tool
from agent.openai.chat_completions_api import FunctionCall

function_name = "search_file_by_name"


class SearchFileByName(Tool):
    @property
    def name(self) -> str:
        return function_name

    @staticmethod
    def new_call(glob_pattern: str, search_directory: str) -> FunctionCall:
        arguments: dict[str, Any] = {
            "glob_pattern": glob_pattern,
            "search_directory": search_directory,
        }

        arguments_str = json.dumps(arguments, ensure_ascii=False)
        return FunctionCall(name=function_name, arguments=arguments_str)
