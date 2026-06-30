import json
from typing import Any

from agent.client.veai.tool import Tool
from agent.openai.chat_completions_api import FunctionCall
from agent.parser import ParsedFunctionCall

function_name = "search_for_text"


class SearchForText(Tool):
    @property
    def name(self) -> str:
        return function_name

    @staticmethod
    def new_call(target_path_or_url: str, text_snippet: str, is_case_sensitive: bool = False) -> ParsedFunctionCall:
        return ParsedFunctionCall(name=function_name, arguments={
            "target_path_or_url": target_path_or_url,
            "text_snippet": text_snippet,
            "is_case_sensitive": is_case_sensitive,
        })
