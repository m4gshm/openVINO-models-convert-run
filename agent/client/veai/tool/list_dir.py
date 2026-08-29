from pydantic import BaseModel

from agent.client.veai.tool import Tool
from agent.parser import ParsedFunctionCall

function_name = "list_dir"


class ListDir(Tool):
    @staticmethod
    def name() -> str:
        return function_name

    @staticmethod
    def new_call(directory_path: str, depth: int) -> ParsedFunctionCall:
        return ParsedFunctionCall(name=function_name, arguments={
            "directory_path": directory_path,
            "depth": depth,
        })

    @staticmethod
    def new_result(directory_tree: list[str]) -> ListDirContent:
        return ListDirContent(result="success with json content", content=TreeContent(directory_tree="\n".join(directory_tree)))

    @staticmethod
    def new_result_json(directory_tree: list[str]) -> str:
        dump = ListDir.new_result(directory_tree=directory_tree).model_dump_json()
        return dump


class ListDirContent(BaseModel):
    result: str
    content: TreeContent


class TreeContent(BaseModel):
    directory_tree: str
