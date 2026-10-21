"""Tests of engine_rest_common: config, chat history, stream flag, aggregation and probes."""
import asyncio
import base64
import unittest
from typing import Any, cast
from unittest.mock import patch

from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCallFunction
from starlette.responses import StreamingResponse

from agent.multimodal.image_decode_test import new_png, rgb_pixels
from agent.multimodal.modality import Modalities, Modality
from agent.multimodal.request_media import prepare_messages
from agent.openai import GenerateOpts
from agent.openai.chat_api import ROLE_ASSISTANT, new_chat_completion_chunk, new_stop_response, new_tool_call
from agent.openai.chat_completions_api import ChatCompletionMessageParam, ChatCompletionRequest
from agent.openai import completions_api
from agent.openai.completions_api import CompletionRequest, CompletionResponse
from agent.openai.engine_rest_common import (ControllerConfig, GenerationRequest, body_stop_strings,
                                             disconnect_probe,
                                             is_stream_request, make_union, new_chat_history,
                                             new_generation_config, new_http_response_chat,
                                             new_http_response_completions)


def new_config(**kwargs) -> ControllerConfig:
    """Controller config with sane defaults, overridable via kwargs."""
    return ControllerConfig(max_prompt_len=4096, model_architectures=set(), **kwargs)


class ControllerConfigCase(unittest.TestCase):
    """Tests of ControllerConfig modality and media limit helpers."""
    def test_text_only_by_default(self):
        """A config without declared modalities reports text only."""
        config = new_config()
        self.assertEqual(["text"], config.modalities_of().names())

    def test_text_is_always_reported(self):
        """Text is reported alongside any declared media modality."""
        config = new_config(modalities={"image"})
        self.assertTrue(config.modalities_of().supports(Modality.TEXT))
        self.assertTrue(config.modalities_of().supports(Modality.IMAGE))

    def test_media_limits_come_from_config(self):
        """Media size/count/local-file/http limits are taken from the config."""
        config = new_config(max_media_bytes=1024, max_media_items=3,
                            allow_local_media_files=True, media_http_timeout=1.5)
        limits = config.media_limits()
        self.assertEqual(1024, limits.max_bytes)
        self.assertEqual(3, limits.max_items)
        self.assertTrue(limits.allow_local_files)
        self.assertEqual(1.5, limits.http_timeout)


class NewChatHistoryCase(unittest.TestCase):
    """Tests of chat history building from OpenAI-style messages."""
    def test_request_models_are_accepted(self):
        """Pydantic request messages are converted to plain history dicts."""
        messages = [ChatCompletionMessageParam(role="user", content="hi")]
        stored = new_chat_history(messages).get_messages()
        self.assertEqual("user", stored[0]["role"])
        self.assertEqual("hi", stored[0]["content"])

    def test_tools_are_registered(self):
        """Tools passed with the request land in the history."""
        tools = [{"type": "function", "function": {"name": "read_file"}}]
        history = new_chat_history([{"role": "user", "content": "hi"}], tools)
        self.assertEqual(1, len(history.get_tools()))

    def test_media_tags_reach_the_history(self):
        """Prepared media parts become ov_genai tags in the stored content."""
        png = "data:image/png;base64," + base64.b64encode(new_png(rgb_pixels(2, 2))).decode()
        messages = [{"role": "user", "content": [{"type": "text", "text": "describe "},
                                                 {"type": "image_url", "image_url": {"url": png}}]}]
        prepared = prepare_messages(messages, Modalities.of(Modality.IMAGE))
        history = new_chat_history(prepared.messages)
        stored = history.get_messages()
        self.assertEqual("describe <ov_genai_image_0>", stored[0]["content"])
        self.assertEqual(1, len(prepared.media.images))


class IsStreamRequestCase(unittest.TestCase):
    """Tests of the stream flag reader."""
    def test_completions_request_without_stream_flag(self):
        """A completions request without the stream flag is not streaming."""
        # /v1/completions default is None; it used to crash with `None | True` TypeError
        body = CompletionRequest(prompt="hi", max_tokens=256, temperature=0, stop=[])
        self.assertIsNone(body.stream)
        self.assertFalse(is_stream_request(body))

    def test_chat_request_defaults_to_streaming(self):
        """A chat request defaults to streaming."""
        self.assertTrue(is_stream_request(ChatCompletionRequest()))

    def test_explicit_flags_are_respected(self):
        """Explicit stream flags win over the defaults."""
        self.assertTrue(is_stream_request(ChatCompletionRequest(stream=True)))
        self.assertFalse(is_stream_request(CompletionRequest(prompt="hi", stream=False)))


class MakeUnionCase(unittest.TestCase):
    """Tests of chunk aggregation into a single response."""
    def test_plain_chunks_without_reasoning_extra(self):
        """Deltas without the reasoning extra aggregate without AttributeError."""
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
        """Reasoning content and tool calls are collected across chunks."""
        tool_call = new_tool_call(ChoiceDeltaToolCallFunction(name="read_file", arguments="{}"))
        chunks = [new_chat_completion_chunk(role=ROLE_ASSISTANT, content="why", thinking=True, model="m"),
                  new_chat_completion_chunk(role=ROLE_ASSISTANT, tool_calls=[tool_call], model="m",
                                            finish_reason="tool_calls")]
        finish_reason, content, reasoning_content, tool_calls = make_union(iter(chunks))
        self.assertEqual("tool_calls", finish_reason)
        self.assertEqual("", content)
        self.assertEqual("why", reasoning_content)
        self.assertEqual([tool_call], tool_calls)


class GenerationConfigCase(unittest.TestCase):
    """Tests of the translation of the request and the server options into a GenerationConfig."""

    def test_server_defaults_are_applied(self):
        """A request without options inherits the defaults of the server."""
        config = new_generation_config(GenerateOpts(temperature=0.3, top_p=0.9, max_new_tokens=128),
                                       GenerationRequest())
        self.assertAlmostEqual(0.3, config.temperature)
        self.assertAlmostEqual(0.9, config.top_p)
        self.assertEqual(128, config.max_new_tokens)

    def test_request_overrides_the_server_defaults(self):
        """A request option wins over the matching default of the server."""
        config = new_generation_config(GenerateOpts(temperature=0.3),
                                       GenerationRequest(temperature=0.9))
        self.assertAlmostEqual(0.9, config.temperature)

    def test_unset_request_option_keeps_the_server_default(self):
        """A request option left unset does not override the server default."""
        config = new_generation_config(GenerateOpts(temperature=0.3),
                                       GenerationRequest(top_p=0.5))
        self.assertAlmostEqual(0.3, config.temperature)
        self.assertAlmostEqual(0.5, config.top_p)

    def test_max_completion_tokens_override_the_default(self):
        """The token limit of the request wins over the default one."""
        config = new_generation_config(GenerateOpts(max_new_tokens=128),
                                       GenerationRequest(max_completion_tokens=64))
        self.assertEqual(64, config.max_new_tokens)

    def test_stop_phrases_never_reach_the_generation_config(self):
        """Stop phrases are applied by the token handler filter, not by the pipeline config.

        generation_config.stop_strings misbehaves in OpenVINO GenAI: it cuts the
        generation instantly (empty result) for prompts that mention template
        special tokens like <|fim_middle|>, even when the stop never appears
        in the generated text.
        """
        config = new_generation_config(GenerateOpts(), GenerationRequest(stop=["a", "b"]))
        self.assertFalse(config.stop_strings)

    def test_completion_stop_reaches_the_body_helper(self):
        """The stop of a /v1/completions body reaches the token handler filter."""
        body = completions_api.CompletionRequest(prompt="p", stop=["|end|", "<|fim_pad|>"])
        self.assertEqual(("|end|", "<|fim_pad|>"), body_stop_strings(body))

    def test_body_without_stop_gives_no_stop_strings(self):
        """A request body without stop phrases yields no filter entries."""
        self.assertEqual((), body_stop_strings(completions_api.CompletionRequest(prompt="p")))
        self.assertEqual((), body_stop_strings(completions_api.CompletionRequest(prompt="p", stop=[])))

    def test_frequency_penalty_of_zero_is_applied(self):
        """A frequency penalty of zero is a value, not a missing one."""
        config = new_generation_config(GenerateOpts(frequency_penalty=0.5),
                                       GenerationRequest(frequency_penalty=0.0))
        self.assertAlmostEqual(0.0, config.frequency_penalty)

    def test_logprobs_enable_the_logprobs_of_the_config(self):
        """A request asking for the logprobs enables them."""
        config = new_generation_config(GenerateOpts(), GenerationRequest(logprobs=True))
        self.assertEqual(1, config.logprobs)


class DisconnectProbeCase(unittest.TestCase):
    """Tests of the throttled disconnect probe."""

    def test_throttles_underlying_probe_within_interval(self):
        """The underlying probe runs at most once per interval window."""
        calls = []

        def fake_is_disconnected(request):
            calls.append(request)
            return False

        with patch("agent.openai.engine_rest_common.is_disconnected", fake_is_disconnected):
            probe = disconnect_probe(cast(Any, object()))
            for _ in range(50):
                self.assertFalse(probe())
        self.assertEqual(1, len(calls))  # one roundtrip per interval window, not per token

    def test_probes_again_after_interval_expired(self):
        """The probe runs again once the interval has expired."""
        calls = []

        def fake_not_disconnected(_request):
            calls.append(1)
            return False

        with patch("agent.openai.engine_rest_common.is_disconnected", fake_not_disconnected):
            probe = disconnect_probe(cast(Any, object()), interval=0.0)
            for _ in range(3):
                self.assertFalse(probe())
        self.assertEqual(3, len(calls))

    def test_disconnect_latches_true_without_further_probes(self):
        """Once disconnected, the probe latches True without further transport calls."""
        calls = []

        def fake_disconnected(_request):
            calls.append(1)
            return True

        with patch("agent.openai.engine_rest_common.is_disconnected", fake_disconnected):
            probe = disconnect_probe(cast(Any, object()), interval=0.0)
            self.assertTrue(probe())
            self.assertTrue(probe())
        self.assertEqual(1, len(calls))  # disconnected is final: no more event-loop roundtrips


class NewHttpResponseCase(unittest.TestCase):
    """Tests of response construction for streaming and non-streaming modes."""
    @staticmethod
    def chunks():
        """Fresh iterator of two assistant chunks."""
        return iter([new_chat_completion_chunk(role="assistant", content="hel"),
                     new_chat_completion_chunk(role="assistant", content="lo", finish_reason="stop")])

    def test_non_streaming_chat_response_is_aggregated(self):
        """Chunks are aggregated into a single ChatCompletion when streaming is off."""
        response = asyncio.run(new_http_response_chat(False, self.chunks()))
        assert isinstance(response, ChatCompletion)
        self.assertEqual("hello", response.choices[0].message.content)
        self.assertEqual("stop", response.choices[0].finish_reason)

    def test_non_streaming_completions_response_keeps_text_schema(self):
        """Non-streaming completions keep the text-completions schema."""
        response = asyncio.run(new_http_response_completions(False, self.chunks()))
        assert isinstance(response, CompletionResponse)
        self.assertEqual("hello", response.choices[0].text)
        self.assertEqual("stop", response.choices[0].finish_reason)

    def test_streaming_returns_sse_response(self):
        """Streaming mode returns a text/event-stream response."""
        response = asyncio.run(new_http_response_chat(True, self.chunks()))
        assert isinstance(response, StreamingResponse)
        self.assertEqual("text/event-stream", response.media_type)


if __name__ == '__main__':
    unittest.main()
