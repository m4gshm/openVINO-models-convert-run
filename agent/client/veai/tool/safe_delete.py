from pydantic import BaseModel

from agent.client.veai.tool import Tool
from agent.parser import ParsedFunctionCall

function_name = "safe_delete"


class SafeDeleteTargets(BaseModel):
    path: str
    name: str
    line: int


class SafeDelete(Tool):
    @property
    def name(self) -> str:
        return function_name

    @staticmethod
    def new_call(targets: list[dict], force: bool, dry_run: bool) -> ParsedFunctionCall:
        return ParsedFunctionCall(name=function_name, arguments={
            "targets": targets,
            "force": force,
            "dry_run": dry_run,
        })
