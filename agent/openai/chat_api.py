import itertools
import time
import uuid
from typing import Optional, List, Literal

from agent.common.roles import ROLE_ASSISTANT
from agent.openai.chat_completions_api import ChatCompletionMessage, ToolCall, CHAT_COMPLETION_CHUNK, \
    CompletionResponse, CHAT_COMPLETION, FunctionCall, ChatCompletionChoice

EMPTY_CONTENT = ' '

tool_call_counter = itertools.count(start=0)


def new_message(content: str | None = None, reasoning_content: str | None = None,
                tool_calls: list[ToolCall] | None = None) -> ChatCompletionMessage:
    if not content or reasoning_content or tool_calls:
        content = EMPTY_CONTENT
    message = ChatCompletionMessage()
    message.role = ROLE_ASSISTANT
    message.content = content
    message.reasoning_content = reasoning_content
    message.tool_calls = tool_calls
    return message


def new_delta(role: str | None, content: str | None = None, thinking: bool = False,
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


def new_stop_response(response_id: str | None = None, model: str | None = None,
                      finish_reason: Literal["stop", "length", "tool_calls"] = "stop",
                      content: str | None = None) -> CompletionResponse:
    return new_response(response_id=response_id, model=model, finish_reason=finish_reason,
                        message=new_message(content=content))


def new_response(message: ChatCompletionMessage,
                 stream: bool = True,
                 response_id: str | None = None,
                 finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None,
                 model: str | None = None) -> CompletionResponse:
    if not response_id:
        response_id = str(uuid.uuid4())
    return CompletionResponse(object=(CHAT_COMPLETION_CHUNK if stream else CHAT_COMPLETION),
                              id=response_id, created=int(time.perf_counter()),
                              model=model, choices=[new_chat_completion_choice(message,
                                                                               finish_reason, stream)])


def new_chunk_response(role: str | None, response_id: str | None = None, content: str | None = None,
                       thinking: bool = False,
                       tool_calls: Optional[List[ToolCall]] = None,
                       finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None,
                       model: str | None = None) -> CompletionResponse:
    delta = new_delta(role=role, content=content, thinking=thinking, tool_calls=tool_calls)
    return new_response(message=delta, stream=True, response_id=response_id,
                        finish_reason=finish_reason, model=model)


def new_tool_call(function: FunctionCall, call_id: str | None = None, ts: int | None = None) -> ToolCall:
    if not call_id:
        call_id = generate_tool_call_id(function.name, ts)
    return ToolCall(id=call_id, function=function)


def generate_tool_call_id(func_name: str, ts: int | None = None) -> str:
    if ts is None:
        ts = int(time.time())
    return f"call_{ts}_{next(tool_call_counter)}_{func_name}"


def new_chat_completion_choice(message: ChatCompletionMessage,
                               finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None,
                               stream: bool | None = True) -> ChatCompletionChoice:
    is_stream = stream == True
    return ChatCompletionChoice(delta=(message if is_stream else None),
                                message=(message if not is_stream else None),
                                finish_reason=finish_reason)
