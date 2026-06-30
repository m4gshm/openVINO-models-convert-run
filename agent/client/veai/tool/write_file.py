from typing import Any

from agent.client.veai.tool import Tool
from agent.openai.chat_completions_api import FunctionCall
from agent.parser import ParsedFunctionCall

function_name = "write_file"


class WriteFile(Tool):
    @property
    def name(self) -> str:
        return function_name

    @staticmethod
    def new_call(target_file: str, content: str, allow_overwrite=True) -> ParsedFunctionCall:
        return ParsedFunctionCall(name=function_name, arguments={
            "target_file": target_file,
            "content": content,
            "allow_overwrite": allow_overwrite,
        })
