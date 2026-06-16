import json
from typing import Any

from common.openai_model import FunctionCall
from veai.tool import Tool

function_name = "read_file"


class ReadFile(Tool):
    @property
    def name(self) -> str:
        return function_name

    @staticmethod
    def new_call(target_file: str, start_line: int = 1, end_line: int = 1000) -> FunctionCall:
        arguments: dict[str, Any] = {
            "target_file": target_file,
            "start_line": start_line,
            "end_line": end_line,
        }
        arguments_str = json.dumps(arguments, ensure_ascii=False)
        return FunctionCall(name=function_name, arguments=arguments_str)
