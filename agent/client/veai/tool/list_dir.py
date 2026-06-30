import json
from typing import Any

from agent.client.veai.tool import Tool
from agent.parser import ParsedFunctionCall

function_name = "list_dir"


class ListDir(Tool):
    @property
    def name(self) -> str:
        return function_name

    @staticmethod
    def new_call(directory_path: str, depth: int) -> ParsedFunctionCall:
        return ParsedFunctionCall(name=function_name, arguments={
            "directory_path": directory_path,
            "depth": depth,
        })
