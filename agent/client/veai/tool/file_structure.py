import json
from typing import Any

from agent.client.veai.tool import Tool
from agent.openai.chat_completions_api import FunctionCall

function_name = "file_structure"


class FileStructure(Tool):
    @property
    def name(self) -> str:
        return function_name

    @staticmethod
    def new_call(target_file: str) -> FunctionCall:
        arguments: dict[str, Any] = {
            "target_file": target_file,
        }

        arguments_str = json.dumps(arguments, ensure_ascii=False)
        return FunctionCall(name=function_name, arguments=arguments_str)
