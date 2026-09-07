from agent.client.veai.tool import Tool
from agent.parser import ParsedFunctionCall

function_name = "search_file_by_name"


class SearchFileByName(Tool):
    @property
    def name(self) -> str:
        return function_name

    @staticmethod
    def new_call(glob_pattern: str | None, search_directory: str | None) -> ParsedFunctionCall:
        arguments = {}
        if not glob_pattern is None:
            arguments["glob_pattern"] = glob_pattern
        if not search_directory is None:
            arguments["search_directory"] = search_directory
        return ParsedFunctionCall(name=function_name, arguments=arguments)
