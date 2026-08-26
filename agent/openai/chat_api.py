import itertools
import time
import uuid
from typing import Optional, List, Literal

import openai
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletionChunk, ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction

from agent.openai.chat_completions_api import CHAT_COMPLETION_CHUNK, CHAT_COMPLETION

EMPTY_CONTENT = ' '

tool_call_counter = itertools.count(start=0)

Role = Literal["developer", "system", "user", "assistant", "tool"]
FinishReason = Literal["stop", "length", "tool_calls", "content_filter"]

ROLE_TOOL = "tool"
ROLE_ASSISTANT = "assistant"
ROLE_USER = "user"


def new_stop_response(role: Role | None, response_id: str | None = None, model: str = "",
                      finish_reason: FinishReason = "stop",
                      content: str | None = None) -> ChatCompletionChunk:
    return new_chat_completion_chunk(role=role, content=content, response_id=response_id, model=model,
                                     finish_reason=finish_reason)


def new_chat_completion(content: str,
                        reasoning_content: str | None = None,
                        tool_calls: Optional[List[ChoiceDeltaToolCall]] = None,
                        response_id: str | None = None,
                        finish_reason: FinishReason = "stop",
                        model: str = "") -> ChatCompletion:
    if not response_id:
        response_id = str(uuid.uuid4())
    return ChatCompletion(object=CHAT_COMPLETION, id=response_id, created=int(time.perf_counter()),
                          model=model, choices=[new_choice_message(content=content,
                                                                   reasoning_content=reasoning_content,
                                                                   finish_reason=finish_reason,
                                                                   tool_calls=tool_calls)])


def new_chat_completion_chunk(role: Role | None, response_id: str | None = None, content: str | None = None,
                              thinking: bool = False, tool_calls: Optional[List[ChoiceDeltaToolCall]] = None,
                              finish_reason: Optional[FinishReason] = None, model: str = "") -> ChatCompletionChunk:
    if not response_id:
        response_id = str(uuid.uuid4())
    return ChatCompletionChunk(object=CHAT_COMPLETION_CHUNK, id=response_id, created=int(time.perf_counter()),
                               model=model, choices=[new_choice_delta(content=content if not thinking else None,
                                                                      reasoning_content=content if thinking else None,
                                                                      role=role,
                                                                      finish_reason=finish_reason,
                                                                      tool_calls=tool_calls)])


def new_tool_call(function: ChoiceDeltaToolCallFunction,
                  call_id: str | None = None,
                  ts: int | None = None) -> ChoiceDeltaToolCall:
    if not call_id:
        call_id = generate_tool_call_id(function.name, ts)
    return ChoiceDeltaToolCall(id=call_id, function=function, type="function", index=0)


def generate_tool_call_id(func_name: str, ts: int | None = None) -> str:
    if ts is None:
        ts = int(time.time())
    return f"call_{ts}_{next(tool_call_counter)}_{func_name}"


def new_choice_delta(role: Role | None = None,
                     content: str | None = None,
                     reasoning_content: str | None = None,
                     tool_calls: Optional[List[ChoiceDeltaToolCall]] = None,
                     finish_reason: Optional[FinishReason] = None):
    delta = new_delta(role, content, reasoning_content, tool_calls)
    return openai.types.chat.chat_completion_chunk.Choice(delta=delta, finish_reason=finish_reason, index=0)


def new_delta(role: Role | None, content: str | None,
              reasoning_content: str | None, tool_calls: list[ChoiceDeltaToolCall] | None) -> ChoiceDelta:
    delta = ChoiceDelta(role=role, content=content, tool_calls=tool_calls)
    if reasoning_content:
        delta.model_extra["reasoning_content"] = reasoning_content
    return delta


def new_choice_message(content: str | None = None,
                       reasoning_content: str | None = None,
                       tool_calls: Optional[List[ChoiceDeltaToolCall]] = None,
                       finish_reason: FinishReason = "stop"):
    message = ChatCompletionMessage(role=ROLE_ASSISTANT, content=content, tool_calls=tool_calls)
    if reasoning_content:
        message.model_extra["reasoning_content"] = reasoning_content
    return openai.types.chat.chat_completion.Choice(message=message, finish_reason=finish_reason, index=0)
