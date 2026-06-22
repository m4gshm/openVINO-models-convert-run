import json
from typing import Any

from agent.client.veai.tool import Tool
from agent.openai.chat_completions_api import FunctionCall

function_name = "write_file"


class WriteFile(Tool):
    @property
    def name(self) -> str:
        return function_name

    @staticmethod
    def new_call(target_file: str, content: str, allow_overwrite=True) -> FunctionCall:
        arguments: dict[str, Any] = {
            "target_file": target_file,
            "content": content,
            "allow_overwrite": allow_overwrite,
        }
        arguments_str = json.dumps(arguments, ensure_ascii=False)
        return FunctionCall(name=function_name, arguments=arguments_str)
