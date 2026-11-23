"""Helpers marking tool-call messages as middleware checkpoints."""
import uuid

from agent.multimodal.content_parts import content_text
from agent.openai.chat_api import ROLE_TOOL
from agent.openai.chat_completions_api import ChatCompletionMessageParam

MIDDLEWARE_CHECKPOINT = "middleware_checkpoint"

# opencode ends the tool reply of an answered question with '"="<answer>". You can now continue ...'
_ANSWER_ASSIGNMENT = '="'


def is_middleware_checkpoint(message: ChatCompletionMessageParam) -> bool:
    """Return True when the message is a tool reply produced by a middleware checkpoint."""
    is_tool = message.role == ROLE_TOOL
    if not is_tool:
        return False
    tool_call_id = message.tool_call_id
    return tool_call_id and tool_call_id.startswith(MIDDLEWARE_CHECKPOINT)


def checkpoint_answer(message: ChatCompletionMessageParam) -> str:
    """The answer the user gave in an answered checkpoint tool reply.

    The clients echo the whole question text before the answer (``<question>"="<answer>. You can
    now continue ...``), so a plain substring search would match words of the echoed question
    itself — the answer must be read out of the last ``="`` assignment.
    """
    text = content_text(message.content)
    if _ANSWER_ASSIGNMENT not in text:
        return text
    tail = text.rsplit(_ANSWER_ASSIGNMENT, 1)[1]
    return tail.split('"', 1)[0]


def new_middleware_call_id() -> str:
    """Return a fresh unique middleware checkpoint tool-call id."""
    return MIDDLEWARE_CHECKPOINT + "_" + str(uuid.uuid4())
