from typing import Any

from agent.client.veai.tool import Tool
from agent.parser import ParsedFunctionCall

function_name = "search_file_by_name"


class SearchFileByName(Tool):
    @property
    def name(self) -> str:
        return function_name

    @staticmethod
    def new_call(glob_pattern: str, search_directory: str) -> ParsedFunctionCall:
        return ParsedFunctionCall(name=function_name, arguments={
            "glob_pattern": glob_pattern,
            "search_directory": search_directory,
        })
