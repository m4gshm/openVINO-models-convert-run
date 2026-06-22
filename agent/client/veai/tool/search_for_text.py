import json
from typing import Any

from agent.client.veai.tool import Tool
from agent.openai.chat_completions_api import FunctionCall

function_name = "search_for_text"


class SearchForText(Tool):
    @property
    def name(self) -> str:
        return function_name

    @staticmethod
    def new_call(target_path_or_url: str, text_snippet: str, is_case_sensitive: bool = False) -> FunctionCall:
        arguments: dict[str, Any] = {
            "target_path_or_url": target_path_or_url,
            "text_snippet": text_snippet,
            "is_case_sensitive": is_case_sensitive,
        }
        arguments_str = json.dumps(arguments, ensure_ascii=False)
        return FunctionCall(name=function_name, arguments=arguments_str)
