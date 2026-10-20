"""Tests of the Ollama HTTP handlers: the endpoints of the Ollama-compatible API."""
import asyncio
import json
import unittest
from typing import Iterable

from openai.types.chat import ChatCompletionChunk
from starlette.requests import Request

from agent.ollama.ollama_api import OllamaChatRequest, OllamaGenerateRequest, OllamaMessage, \
    OllamaOptions, OllamaShowRequest
from agent.ollama.ollama_handlers import OllamaApi
from agent.openai.chat_api import ROLE_ASSISTANT, new_chat_completion_chunk
from agent.openai.chat_completions_api import ChatCompletionRequest
from agent.openai.engine_rest_common import ControllerConfig
from agent.openai.models_api import ModelObject, ModelsListResponse


class FakeController:
    """A controller that records the requests and answers with fixed chunks."""

    def __init__(self, chunks: list[ChatCompletionChunk] | None = None):
        """Store the chunks the fake engine answers with."""
        self.config = ControllerConfig(model_name="test-model", max_prompt_len=4096,
                                       model_architectures=set())
        self.chunks = chunks if chunks is not None else [
            new_chat_completion_chunk(role=ROLE_ASSISTANT, content="hello", model="test-model"),
            new_chat_completion_chunk(role=ROLE_ASSISTANT, content="", finish_reason="stop",
                                      model="test-model"),
        ]
        self.requests: list[ChatCompletionRequest] = []

    async def models(self) -> ModelsListResponse:
        """The single served model."""
        return ModelsListResponse(data=[ModelObject(id="test-model")])

    def validate_messages(self, messages, tools) -> ChatCompletionChunk | None:  # pylint: disable=unused-argument
        """No validation problem in the fake engine."""
        return None

    async def chat_chunks(self, body: ChatCompletionRequest,
                           request: Request) -> Iterable[ChatCompletionChunk]:  # pylint: disable=unused-argument
        """Record the request and answer with the fixed chunks."""
        self.requests.append(body)
        return iter(self.chunks)


def new_request() -> Request:
    """A request object the fake controller does not use."""
    return Request({"type": "http", "method": "POST", "path": "/api/chat",
                    "headers": [], "query_string": b""})


class OllamaTagsHandlerCase(unittest.TestCase):
    """Tests of GET /api/tags."""

    def test_tags_list_the_served_model(self):
        """The model list of the engine becomes the Ollama model list."""
        api = OllamaApi(FakeController())
        response = asyncio_run(api.tags())
        self.assertEqual(1, len(response.models))
        self.assertEqual("test-model", response.models[0].name)


class OllamaShowHandlerCase(unittest.TestCase):
    """Tests of POST /api/show."""

    def test_show_answers_the_model_name(self):
        """The show endpoint answers with the name of the model."""
        api = OllamaApi(FakeController())
        response = asyncio_run(api.show(OllamaShowRequest(model="test-model")))
        self.assertEqual("test-model", response.model)


class OllamaChatHandlerCase(unittest.TestCase):
    """Tests of POST /api/chat."""

    def test_non_streaming_answers_one_object(self):
        """A non-streaming chat request answers with a single JSON object."""
        api = OllamaApi(FakeController())
        response = asyncio_run(api.chat(OllamaChatRequest(model="test-model", stream=False,
                                                           messages=[OllamaMessage(role="user",
                                                                                      content="hi")]),
                                        new_request()))
        content = json.loads(response.body)
        self.assertEqual("hello", content["message"]["content"])
        self.assertTrue(content["done"])

    def test_streaming_answers_json_lines(self):
        """A streaming chat request answers with one JSON object per line."""
        api = OllamaApi(FakeController())
        response = asyncio_run(api.chat(OllamaChatRequest(model="test-model", stream=True,
                                                           messages=[OllamaMessage(role="user",
                                                                                      content="hi")]),
                                        new_request()))
        lines = asyncio_run(read_lines(response))
        self.assertEqual(2, len(lines))
        self.assertEqual("hello", lines[0]["message"]["content"])
        self.assertTrue(lines[1]["done"])

    def test_the_request_is_translated(self):
        """The Ollama request reaches the engine as an OpenAI one."""
        controller = FakeController()
        api = OllamaApi(controller)
        asyncio_run(api.chat(OllamaChatRequest(model="test-model", stream=False,
                                               messages=[OllamaMessage(role="user", content="hi")],
                                               options=OllamaOptions(temperature=0.4)),
                             new_request()))
        self.assertEqual(1, len(controller.requests))
        self.assertEqual(0.4, controller.requests[0].temperature)
        self.assertEqual("hi", controller.requests[0].messages[0].content)


class OllamaGenerateHandlerCase(unittest.TestCase):
    """Tests of POST /api/generate."""

    def test_non_streaming_answers_the_prompt(self):
        """A non-streaming generate request answers with the generated text."""
        api = OllamaApi(FakeController())
        response = asyncio_run(api.generate(OllamaGenerateRequest(model="test-model", prompt="hi",
                                                                   stream=False), new_request()))
        content = json.loads(response.body)
        self.assertEqual("hello", content["response"])
        self.assertEqual("hi", content["prompt"])

    def test_streaming_answers_json_lines(self):
        """A streaming generate request answers with one JSON object per line."""
        api = OllamaApi(FakeController())
        response = asyncio_run(api.generate(OllamaGenerateRequest(model="test-model", prompt="hi",
                                                                   stream=True), new_request()))
        lines = asyncio_run(read_lines(response))
        self.assertEqual("hello", lines[0]["response"])
        self.assertTrue(lines[1]["done"])


def asyncio_run(coro):
    """Run a coroutine in a fresh event loop."""
    return asyncio.run(coro)


async def read_lines(response) -> list[dict]:
    """Read the JSON lines of a streaming response."""
    lines = []
    async for line in response.body_iterator:
        lines.append(json.loads(line))
    return lines


if __name__ == '__main__':
    unittest.main()
