"""Call builders and result models for the list_dir tool."""
from pydantic import BaseModel

from agent.client.veai.tool import Tool
from agent.parser import ParsedFunctionCall

FUNCTION_NAME = "list_dir"


class TreeContent(BaseModel):
    """Directory tree payload returned by list_dir."""

    directory_tree: str


class ListDirContent(BaseModel):
    """Tool-result envelope: status plus the tree content."""

    result: str
    content: TreeContent


class ListDir(Tool):
    """Builds list_dir tool calls and result payloads."""

    @staticmethod
    def name() -> str:
        """Tool name as declared in tool definitions."""
        return FUNCTION_NAME

    @staticmethod
    def new_call(directory_path: str, depth: int) -> ParsedFunctionCall:
        """Build the parsed function call with directory and depth."""
        return ParsedFunctionCall(name=FUNCTION_NAME, arguments={
            "directory_path": directory_path,
            "depth": depth,
        })

    @staticmethod
    def new_result(directory_tree: list[str]) -> ListDirContent:
        """Wrap directory tree lines into a tool result model."""
        return ListDirContent(result="success with json content",
                              content=TreeContent(directory_tree="\n".join(directory_tree)))

    @staticmethod
    def new_result_json(directory_tree: list[str]) -> str:
        """Serialize the tool result model to a JSON string."""
        dump = ListDir.new_result(directory_tree=directory_tree).model_dump_json()
        return dump
