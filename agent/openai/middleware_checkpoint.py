import uuid

from agent.openai.chat_api import ROLE_TOOL
from agent.openai.chat_completions_api import ChatCompletionMessageParam

MIDDLEWARE_CHECKPOINT = "middleware_checkpoint"


def is_middleware_checkpoint(message: ChatCompletionMessageParam) -> bool:
    is_tool = message.role == ROLE_TOOL
    if not is_tool:
        return False
    tool_call_id = message.tool_call_id
    return tool_call_id and tool_call_id.startswith(MIDDLEWARE_CHECKPOINT)


def new_middleware_call_id() -> str:
    return MIDDLEWARE_CHECKPOINT + "_" + str(uuid.uuid4())
