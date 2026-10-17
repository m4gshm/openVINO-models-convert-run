import asyncio
import base64
import unittest

from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCallFunction

from agent.multimodal.image_decode_test import new_png, rgb_pixels
from agent.multimodal.modality import Modalities, Modality
from agent.multimodal.request_media import prepare_messages
from agent.openai.chat_api import ROLE_ASSISTANT, new_chat_completion_chunk, new_stop_response, new_tool_call
from agent.openai.chat_completions_api import ChatCompletionMessageParam, ChatCompletionRequest
from agent.openai.completions_api import CompletionRequest, CompletionResponse
from agent.openai.engine_rest_common import ControllerConfig, is_stream_request, make_union, new_chat_history


def new_config(**kwargs) -> ControllerConfig:
    return ControllerConfig(max_prompt_len=4096, model_architectures=set(), **kwargs)


class ControllerConfigCase(unittest.TestCase):
    def test_text_only_by_default(self):
        config = new_config()
        self.assertEqual(["text"], config.modalities_of().names())

    def test_text_is_always_reported(self):
        config = new_config(modalities={"image"})
        self.assertTrue(config.modalities_of().supports(Modality.TEXT))
        self.assertTrue(config.modalities_of().supports(Modality.IMAGE))

    def test_media_limits_come_from_config(self):
        config = new_config(max_media_bytes=1024, max_media_items=3,
                            allow_local_media_files=True, media_http_timeout=1.5)
        limits = config.media_limits()
        self.assertEqual(1024, limits.max_bytes)
        self.assertEqual(3, limits.max_items)
        self.assertTrue(limits.allow_local_files)
        self.assertEqual(1.5, limits.http_timeout)


class NewChatHistoryCase(unittest.TestCase):
    def test_request_models_are_accepted(self):
        messages = [ChatCompletionMessageParam(role="user", content="hi")]
        stored = new_chat_history(messages).get_messages()
        self.assertEqual("user", stored[0]["role"])
        self.assertEqual("hi", stored[0]["content"])

    def test_tools_are_registered(self):
        tools = [{"type": "function", "function": {"name": "read_file"}}]
        history = new_chat_history([{"role": "user", "content": "hi"}], tools)
        self.assertEqual(1, len(history.get_tools()))

    def test_media_tags_reach_the_history(self):
        png = "data:image/png;base64," + base64.b64encode(new_png(rgb_pixels(2, 2))).decode()
        messages = [{"role": "user", "content": [{"type": "text", "text": "describe "},
                                                 {"type": "image_url", "image_url": {"url": png}}]}]
        prepared = prepare_messages(messages, Modalities.of(Modality.IMAGE))
        history = new_chat_history(prepared.messages)
        stored = history.get_messages()
        self.assertEqual("describe <ov_genai_image_0>", stored[0]["content"])
        self.assertEqual(1, len(prepared.media.images))


class IsStreamRequestCase(unittest.TestCase):
    def test_completions_request_without_stream_flag(self):
        # /v1/completions default is None; it used to crash with `None | True` TypeError
        body = CompletionRequest(prompt="hi", max_tokens=256, temperature=0, stop=[])
        self.assertIsNone(body.stream)
        self.assertFalse(is_stream_request(body))

    def test_chat_request_defaults_to_streaming(self):
        self.assertTrue(is_stream_request(ChatCompletionRequest()))

    def test_explicit_flags_are_respected(self):
        self.assertTrue(is_stream_request(ChatCompletionRequest(stream=True)))
        self.assertFalse(is_stream_request(CompletionRequest(prompt="hi", stream=False)))


class MakeUnionCase(unittest.TestCase):
    def test_plain_chunks_without_reasoning_extra(self):
        # reasoning_content is an optional extra field: deltas of regular tokens (and error
        # stop responses) do not carry it — direct attribute access used to raise AttributeError
        chunks = [new_chat_completion_chunk(role=ROLE_ASSISTANT, content="Hel", model="m"),
                  new_chat_completion_chunk(role=ROLE_ASSISTANT, content="lo", model="m"),
                  new_stop_response(role=ROLE_ASSISTANT, content="boom", finish_reason="stop", model="m")]
        finish_reason, content, reasoning_content, tool_calls = make_union(iter(chunks))
        self.assertEqual("stop", finish_reason)
        self.assertEqual("Helloboom", content)
        self.assertEqual("", reasoning_content)
        self.assertEqual([], tool_calls)

    def test_thinking_and_tool_calls_are_aggregated(self):
        tool_call = new_tool_call(ChoiceDeltaToolCallFunction(name="read_file", arguments="{}"))
        chunks = [new_chat_completion_chunk(role=ROLE_ASSISTANT, content="why", thinking=True, model="m"),
                  new_chat_completion_chunk(role=ROLE_ASSISTANT, tool_calls=[tool_call], model="m",
                                            finish_reason="tool_calls")]
        finish_reason, content, reasoning_content, tool_calls = make_union(iter(chunks))
        self.assertEqual("tool_calls", finish_reason)
        self.assertEqual("", content)
        self.assertEqual("why", reasoning_content)
        self.assertEqual([tool_call], tool_calls)


class NewHttpResponseCase(unittest.TestCase):
    @staticmethod
    def chunks():
        return iter([new_chat_completion_chunk(role="assistant", content="hel"),
                     new_chat_completion_chunk(role="assistant", content="lo", finish_reason="stop")])

    def test_non_streaming_chat_response_is_aggregated(self):
        from openai.types.chat import ChatCompletion

        from agent.openai.engine_rest_common import new_http_response_chat

        response = asyncio.run(new_http_response_chat(False, self.chunks()))
        assert isinstance(response, ChatCompletion)
        self.assertEqual("hello", response.choices[0].message.content)
        self.assertEqual("stop", response.choices[0].finish_reason)

    def test_non_streaming_completions_response_keeps_text_schema(self):
        from agent.openai.engine_rest_common import new_http_response_completions

        response = asyncio.run(new_http_response_completions(False, self.chunks()))
        assert isinstance(response, CompletionResponse)
        self.assertEqual("hello", response.choices[0].text)
        self.assertEqual("stop", response.choices[0].finish_reason)

    def test_streaming_returns_sse_response(self):
        from starlette.responses import StreamingResponse

        from agent.openai.engine_rest_common import new_http_response_chat

        response = asyncio.run(new_http_response_chat(True, self.chunks()))
        assert isinstance(response, StreamingResponse)
        self.assertEqual("text/event-stream", response.media_type)


if __name__ == '__main__':
    unittest.main()
