"""Conversion between the Ollama API payloads and the OpenAI-compatible ones.

The Ollama protocol differs from the OpenAI one in three ways: the model list has its
own shape, the chat request carries its own options and the streaming response is a
sequence of JSON lines (NDJSON) instead of SSE. This module holds the pure conversions;
the HTTP layer lives in :mod:`agent.ollama.ollama_handlers`.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Optional

from openai.types.chat import ChatCompletionChunk

from agent.ollama.ollama_api import OllamaChatRequest, OllamaGenerateRequest, OllamaMessage, \
    OllamaModelDetails, OllamaModelEntry, OllamaOptions, OllamaTagsResponse
from agent.openai.chat_completions_api import ChatCompletionFunctionToolParam, \
    ChatCompletionMessageParam, ChatCompletionRequest

log = logging.getLogger(__name__)

DEFAULT_IMAGE_MEDIA_TYPE = "image/png"


def now_iso() -> str:
    """Current time in the ISO 8601 format the Ollama API uses."""
    return datetime.now(timezone.utc).isoformat()


def ollama_options_to_request_params(options: Optional[OllamaOptions]) -> dict[str, Any]:
    """Map the Ollama options onto the matching parameters of an OpenAI request."""
    if options is None:
        return {}
    params: dict[str, Any] = {}
    if options.temperature is not None:
        params["temperature"] = options.temperature
    if options.top_p is not None:
        params["top_p"] = options.top_p
    if options.num_predict is not None:
        params["max_tokens"] = options.num_predict
    if options.stop:
        params["stop"] = list(options.stop)
    return params


def image_content_part(image_base64: str) -> dict[str, Any]:
    """Build an OpenAI image content part out of a base64 payload of an Ollama message."""
    return {"type": "image_url",
            "image_url": {"url": f"data:{DEFAULT_IMAGE_MEDIA_TYPE};base64,{image_base64}"}}


def ollama_message_to_openai(message: OllamaMessage) -> ChatCompletionMessageParam:
    """Convert one Ollama message into an OpenAI chat message."""
    content: Any = message.content
    if message.images:
        parts: list[dict[str, Any]] = []
        if message.content:
            parts.append({"type": "text", "text": message.content})
        parts.extend(image_content_part(image) for image in message.images)
        content = parts
    return ChatCompletionMessageParam(role=message.role, content=content)


def ollama_messages_to_openai(messages: Iterable[OllamaMessage]) -> list[ChatCompletionMessageParam]:
    """Convert the messages of an Ollama chat request into OpenAI chat messages."""
    return [ollama_message_to_openai(message) for message in messages]


def ollama_tool_to_openai(tool: dict[str, Any]) -> ChatCompletionFunctionToolParam:
    """Convert one Ollama tool into an OpenAI function tool.

    The Ollama clients describe the parameters loosely: the JSON schema of the function
    may miss the ``type`` and the ``properties``, which the OpenAI schema requires.
    """
    function = dict(tool.get("function") or {})
    parameters = dict(function.get("parameters") or {})
    parameters.setdefault("type", "object")
    parameters.setdefault("properties", {})
    function["parameters"] = parameters
    return ChatCompletionFunctionToolParam.model_validate({"type": "function", "function": function})


def ollama_chat_to_request(body: OllamaChatRequest) -> ChatCompletionRequest:
    """Convert an Ollama chat request into an OpenAI chat completion request."""
    params = ollama_options_to_request_params(body.options)
    tools = [ollama_tool_to_openai(tool) for tool in (body.tools or [])]
    return ChatCompletionRequest(model=body.model,
                                 messages=ollama_messages_to_openai(body.messages),
                                 stream=body.stream,
                                 tools=tools,
                                 **params)


def ollama_generate_to_request(body: OllamaGenerateRequest) -> ChatCompletionRequest:
    """Convert an Ollama generate request into an OpenAI chat completion request.

    The Ollama generate API takes a raw prompt; it is served here as a single user
    message (preceded by the system prompt when the request carries one).
    """
    messages: list[ChatCompletionMessageParam] = []
    if body.system:
        messages.append(ChatCompletionMessageParam(role="system", content=body.system))
    content: Any = body.prompt
    if body.images:
        parts: list[dict[str, Any]] = []
        if body.prompt:
            parts.append({"type": "text", "text": body.prompt})
        parts.extend(image_content_part(image) for image in body.images)
        content = parts
    messages.append(ChatCompletionMessageParam(role="user", content=content))
    params = ollama_options_to_request_params(body.options)
    return ChatCompletionRequest(model=body.model, messages=messages, stream=body.stream, **params)


def tool_calls_of(delta: Any) -> list[dict[str, Any]]:
    """Convert the tool call deltas of a chunk into the Ollama tool call shape."""
    calls: list[dict[str, Any]] = []
    for call in getattr(delta, "tool_calls", None) or []:
        function = getattr(call, "function", None)
        arguments = getattr(function, "arguments", None) or "{}"
        try:
            parsed = json.loads(arguments)
        except (TypeError, ValueError):
            parsed = arguments
        calls.append({"function": {"name": getattr(function, "name", "") or "",
                                  "arguments": parsed}})
    return calls


def done_reason_of(finish_reason: Optional[str]) -> str:
    """Map an OpenAI finish reason onto the Ollama done reason."""
    if finish_reason == "length":
        return "length"
    return "stop"


def chat_chunk_to_ollama(chunk: ChatCompletionChunk, model: str, created_at: str) -> dict[str, Any]:
    """Convert one OpenAI chat chunk into an Ollama chat response line."""
    message: dict[str, Any] = {"role": "assistant", "content": ""}
    done = False
    done_reason = None
    if chunk.choices:
        choice = chunk.choices[0]
        delta = choice.delta
        if delta.content:
            message["content"] = delta.content
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            message["thinking"] = reasoning
        tool_calls = tool_calls_of(delta)
        if tool_calls:
            message["tool_calls"] = tool_calls
        if choice.finish_reason:
            done = True
            done_reason = done_reason_of(choice.finish_reason)
    line: dict[str, Any] = {"model": model, "created_at": created_at, "message": message, "done": done}
    if done_reason:
        line["done_reason"] = done_reason
    return line


def chat_chunks_to_ollama(chunks: Iterable[ChatCompletionChunk], model: str,
                          created_at: str) -> Iterator[dict[str, Any]]:
    """Convert a stream of OpenAI chat chunks into Ollama chat response lines."""
    for chunk in chunks:
        yield chat_chunk_to_ollama(chunk, model, created_at)


def generate_chunk_to_ollama(chunk: ChatCompletionChunk, model: str, created_at: str,
                             prompt: str) -> dict[str, Any]:
    """Convert one OpenAI chat chunk into an Ollama generate response line."""
    line = chat_chunk_to_ollama(chunk, model, created_at)
    message = line.pop("message", {})
    response: dict[str, Any] = {"model": model, "created_at": created_at,
                                "prompt": prompt, "response": message.get("content", ""),
                                "done": line.get("done", False)}
    if "done_reason" in line:
        response["done_reason"] = line["done_reason"]
    tool_calls = message.get("tool_calls")
    if tool_calls:
        response["tool_calls"] = tool_calls
    return response


def aggregate_ollama_chat(lines: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the Ollama chat response lines into a single non-streaming response."""
    content = ""
    thinking = ""
    tool_calls: list[dict[str, Any]] = []
    done = False
    done_reason = None
    model = ""
    created_at = ""
    for line in lines:
        model = line.get("model", model)
        created_at = line.get("created_at", created_at)
        message = line.get("message") or {}
        content += message.get("content", "")
        thinking += message.get("thinking", "") or ""
        tool_calls.extend(message.get("tool_calls") or [])
        if line.get("done"):
            done = True
            done_reason = line.get("done_reason")
    result: dict[str, Any] = {"model": model, "created_at": created_at, "done": done}
    message = {"role": "assistant", "content": content}
    if thinking:
        message["thinking"] = thinking
    if tool_calls:
        message["tool_calls"] = tool_calls
    result["message"] = message
    if done_reason:
        result["done_reason"] = done_reason
    return result


def aggregate_ollama_generate(lines: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the Ollama generate response lines into a single non-streaming response."""
    response_text = ""
    done = False
    done_reason = None
    model = ""
    created_at = ""
    prompt = ""
    for line in lines:
        model = line.get("model", model)
        created_at = line.get("created_at", created_at)
        prompt = line.get("prompt", prompt)
        response_text += line.get("response", "")
        if line.get("done"):
            done = True
            done_reason = line.get("done_reason")
    result: dict[str, Any] = {"model": model, "created_at": created_at, "prompt": prompt,
                               "response": response_text, "done": done}
    if done_reason:
        result["done_reason"] = done_reason
    return result


def tags_of_models(models: Iterable[Any]) -> OllamaTagsResponse:
    """Build the Ollama model list out of the models of the OpenAI model list."""
    created_at = now_iso()
    entries: list[OllamaModelEntry] = []
    for model in models:
        model_id = getattr(model, "id", None) or str(model)
        details = OllamaModelDetails()
        entries.append(OllamaModelEntry(name=model_id, model=model_id, modified_at=created_at,
                                        size=0, digest="", details=details))
    return OllamaTagsResponse(models=entries)
