"""HTTP handlers of the Ollama-compatible API.

The endpoints translate an Ollama request into the OpenAI-compatible request of the
engine, run it, and translate the response back into the Ollama shape. The streaming
responses are JSON lines (NDJSON), the way the Ollama API streams them.
"""
import asyncio
import json
import logging
from typing import Iterable, Protocol

from openai.types.chat import ChatCompletionChunk
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from agent.ollama.ollama_api import OllamaChatRequest, OllamaGenerateRequest, OllamaShowRequest, \
    OllamaShowResponse, OllamaTagsResponse, OllamaVersionResponse
from agent.ollama.ollama_convert import aggregate_ollama_chat, aggregate_ollama_generate, \
    chat_chunk_to_ollama, chat_chunks_to_ollama, generate_chunk_to_ollama, now_iso, \
    ollama_chat_to_request, ollama_generate_to_request, tags_of_models
from agent.openai.chat_completions_api import ChatCompletionRequest
from agent.openai.engine_rest_common import ControllerConfig
from agent.openai.models_api import ModelsListResponse

log = logging.getLogger(__name__)

NDJSON_MEDIA_TYPE = "application/x-ndjson"


class OllamaControllerProtocol(Protocol):
    """The part of the engine controller the Ollama API needs."""

    config: ControllerConfig

    async def models(self) -> ModelsListResponse:
        """The list of the served models."""

    def validate_messages(self, messages, tools) -> ChatCompletionChunk | None:
        """An intervention chunk for a problematic request, else None."""

    async def chat_chunks(self, body: ChatCompletionRequest,
                           request: Request) -> Iterable[ChatCompletionChunk]:
        """The chunks of a chat completion request."""


def ndjson_stream(lines: Iterable[dict]) -> StreamingResponse:
    """Wrap a sequence of JSON objects into an NDJSON streaming response."""
    def generate():
        """Serialize every line as one JSON object followed by a newline."""
        for line in lines:
            yield json.dumps(line) + "\n"

    return StreamingResponse(generate(), media_type=NDJSON_MEDIA_TYPE)


class OllamaApi:
    """Ollama-compatible endpoints on top of an OpenAI-compatible engine controller."""

    def __init__(self, controller: OllamaControllerProtocol):
        """Bind the endpoints to the engine controller."""
        self.controller = controller

    @property
    def model_name(self) -> str:
        """Name of the served model."""
        return self.controller.config.model_name or ""

    async def tags(self) -> OllamaTagsResponse:
        """GET /api/tags: the list of the served models."""
        models = await self.controller.models()
        return tags_of_models(models.data)

    async def show(self, body: OllamaShowRequest) -> OllamaShowResponse:
        """POST /api/show: the details of a served model."""
        models = await self.controller.models()
        names = [model.id for model in models.data if model.id]
        if body.model and names and body.model not in names:
            return OllamaShowResponse(model=body.model)
        return OllamaShowResponse(model=body.model or (names[0] if names else ""))

    async def version(self) -> OllamaVersionResponse:
        """GET /api/version: the version of the API."""
        return OllamaVersionResponse()

    async def chat(self, body: OllamaChatRequest, request: Request):
        """POST /api/chat: a chat completion in the Ollama shape."""
        chat_request = ollama_chat_to_request(body)
        invalid = self.controller.validate_messages(chat_request.messages, chat_request.tools)
        lines: Iterable[dict]
        if invalid is not None:
            lines = [self._line_of_chunk(invalid)]
        else:
            chunks = await self.controller.chat_chunks(chat_request, request)
            lines = chat_chunks_to_ollama(chunks, self.model_name, now_iso())
        if body.stream:
            return ndjson_stream(lines)
        # the aggregation consumes the whole generation: keep it off the event loop
        content = await asyncio.to_thread(aggregate_ollama_chat, lines)
        return JSONResponse(content=content)

    async def generate(self, body: OllamaGenerateRequest, request: Request):
        """POST /api/generate: a completion of a raw prompt in the Ollama shape."""
        chat_request = ollama_generate_to_request(body)
        created_at = now_iso()
        chunks = await self.controller.chat_chunks(chat_request, request)

        def lines():
            """Convert the chat chunks into generate response lines."""
            for chunk in chunks:
                yield generate_chunk_to_ollama(chunk, self.model_name, created_at, body.prompt)

        if body.stream:
            return ndjson_stream(lines())
        # the aggregation consumes the whole generation: keep it off the event loop
        content = await asyncio.to_thread(aggregate_ollama_generate, lines())
        return JSONResponse(content=content)

    def _line_of_chunk(self, chunk) -> dict:
        """Convert a single chunk into an Ollama chat response line."""
        return chat_chunk_to_ollama(chunk, self.model_name, now_iso())
