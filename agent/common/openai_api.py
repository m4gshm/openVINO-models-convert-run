import itertools
import time
import uuid
from typing import Optional, List

from common.openai_model import ToolCall, FunctionCall, ChatCompletionChoice, ChatCompletionMessage, \
    OpenAICompletionResponse, CHAT_COMPLETION_CHUNK, CHAT_COMPLETION
from common.roles import ROLE_ASSISTANT

tool_call_counter = itertools.count(start=0)


def new_response(chat_completion_message: ChatCompletionMessage, stream: bool = True,
                 response_id: str | None = None, finish_reason: str | None = None,
                 model: str | None = None) -> OpenAICompletionResponse:
    if not response_id:
        response_id = str(uuid.uuid4())
    return OpenAICompletionResponse(object=(CHAT_COMPLETION_CHUNK if stream else CHAT_COMPLETION),
                                    id=response_id, created=int(time.perf_counter()),
                                    model=model, choices=[new_chat_completion_choice(chat_completion_message,
                                                                                     finish_reason, stream)])


def new_chunk_response(role: str, response_id: str | None = None, content: str | None = None, thinking: bool = False,
                       tool_calls: Optional[List[ToolCall]] = None,
                       finish_reason: str | None = None, model: str | None = None) -> OpenAICompletionResponse:
    delta = new_delta(role=role, content=content, thinking=thinking, tool_calls=tool_calls)
    return new_response(chat_completion_message=delta, stream=True, response_id=response_id,
                        finish_reason=finish_reason, model=model)


def new_tool_call(function: FunctionCall, call_id: str | None = None, ts: int | None = None) -> ToolCall:
    if not call_id:
        call_id = generate_tool_call_id(function.name, ts)
    return ToolCall(id=call_id, function=function)


def generate_tool_call_id(func_name: str, ts: int | None = None) -> str:
    if ts is None:
        ts = int(time.time())
    return f"call_{ts}_{next(tool_call_counter)}_{func_name}"


def new_message(content: str | None = None, reasoning_content: str | None = None,
                tool_calls: list[ToolCall] | None = None) -> ChatCompletionMessage:
    message = ChatCompletionMessage()
    message.role = ROLE_ASSISTANT
    message.content = content
    message.reasoning_content = reasoning_content
    message.tool_calls = tool_calls
    return message


def new_delta(role: str, content: str | None = None, thinking: bool = False,
              tool_calls: list[ToolCall] | None = None) -> ChatCompletionMessage:
    delta = ChatCompletionMessage()
    delta.role = role
    if thinking:
        delta.reasoning_content = content
    else:
        delta.content = content

    if tool_calls:
        delta.tool_calls = tool_calls
    return delta


def new_chat_completion_choice(chat_completion_message: ChatCompletionMessage, finish_reason: str | None = None,
                               stream: bool | None = True) -> ChatCompletionChoice:
    is_stream = stream == True
    return ChatCompletionChoice(delta=(chat_completion_message if is_stream else None),
                                message=(chat_completion_message if not is_stream else None),
                                finish_reason=finish_reason)
