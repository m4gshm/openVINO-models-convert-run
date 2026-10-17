"""Client-side helpers: user context types and agent marker detection."""
from typing import Callable

from agent.client.tool_select_model import ToolSelectOptions
from agent.multimodal.content_parts import content_text
from agent.openai.chat_completions_api import ChatCompletionMessageParam

__all__ = ["ToolSelectOptions", "is_agent", "check_agent"]


def is_agent(messages: list[ChatCompletionMessageParam], marker: str) -> bool:
    """Check that the first message content contains the marker string."""
    def check(content: str) -> bool:
        return marker in content

    return check_agent(messages, check)


def check_agent(messages: list[ChatCompletionMessageParam], check: Callable[..., bool]) -> bool:
    """Return True when check() passes on the first message content."""
    first_message = messages[0] if messages else None
    if not first_message:
        return False
    return check(content_text(first_message.content))
