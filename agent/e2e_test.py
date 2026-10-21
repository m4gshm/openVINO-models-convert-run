"""Tests of the e2e helpers (SSE parsing) plus the opt-in real-model e2e suite.

The SSE-parser unit tests run in the plain `task test` pass; the real-model
tests carry the `e2e` marker and run only via `-m e2e` (default pytest config
deselects them so `task test` never downloads the model or starts a server).
"""
# pylint: disable=redefined-outer-name  # the session fixture and its test parameters share a name
from collections.abc import Iterator

import pytest

from agent.e2e import (EXPECTED_CHAT_PHRASE, FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX, FetchFailure,
                       ServerHarness, chat_completion, ensure_model, parse_sse_response,
                       text_completion_fim)

TEST_FRAMES = [
    'data: {"choices":[{"delta":{"content":"2+"}}]}',
    "",
    'data: {"choices":[{"delta":{"content":"2"}}]}',
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
    "data: [DONE]",
]


def test_parse_sse_accumulates_deltas_and_finish_reason():
    """Deltas concatenate in frame order; finish_reason and the sentinel are reported."""
    content, finish_reason, done = parse_sse_response(iter(TEST_FRAMES))
    assert content == "2+2"
    assert finish_reason == "stop"
    assert done is True


def test_parse_sse_without_sentinel_reports_not_done():
    """A stream cut before data: [DONE] is distinguishable from a complete one."""
    content, finish_reason, done = parse_sse_response(TEST_FRAMES[:-1])
    assert content == "2+2"
    assert finish_reason == "stop"
    assert done is False


def test_parse_sse_of_empty_stream():
    """An empty stream yields empty content, no finish reason and no sentinel."""
    content, finish_reason, done = parse_sse_response([])
    assert content == ""
    assert finish_reason is None
    assert done is False


def test_parse_sse_of_length_finish():
    """A length-bound generation reports finish_reason length without the sentinel."""
    frames = ['data: {"choices":[{"delta":{"content":"ab"}}]}',
              'data: {"choices":[{"delta":{},"finish_reason":"length"}]}']
    content, finish_reason, done = parse_sse_response(frames)
    assert content == "ab"
    assert finish_reason == "length"
    assert done is False


@pytest.fixture(scope="session")
def base_url() -> Iterator[str]:
    """One server instance serving the fetched model for the whole e2e session."""
    try:
        model_dir = ensure_model()
    except FetchFailure as fetch_error:
        pytest.skip(f"e2e model unavailable: {fetch_error}")
    harness = ServerHarness(model_dir)
    try:
        harness.start()
        yield harness.base_url
    finally:
        harness.stop()


def _assert_contains(actual: str, expected: str, what: str) -> None:
    """Fail with both texts in the report so a mismatch is diagnosable alone."""
    assert expected in actual, \
        f"{what} does not contain the expected phrase\nexpected: {expected!r}\nactual: {actual!r}"


@pytest.mark.e2e
def test_chat_non_streaming(base_url: str) -> None:
    """A stream=false chat request answers one assistant choice carrying the expected phrase."""
    response, _ = chat_completion(stream=False, base_url=base_url)
    choices = response["choices"]
    assert len(choices) == 1, f"expected one choice, got {len(choices)}"
    choice = choices[0]
    assert choice["message"]["role"] == "assistant"
    assert choice["finish_reason"] in ("stop", "length"), choice["finish_reason"]
    assert response["model"].startswith("Qwen3.5-2B")
    _assert_contains(choice["message"]["content"] or "", EXPECTED_CHAT_PHRASE, "chat content")


@pytest.mark.e2e
def test_chat_streaming(base_url: str) -> None:
    """A stream=true chat request ends with the sentinel and carries the expected phrase."""
    streamed, sentinel = chat_completion(stream=True, base_url=base_url)
    assert sentinel == "done", "stream ended without the data: [DONE] sentinel frame"
    assert streamed["finish_reason"] in ("stop", "length"), streamed["finish_reason"]
    _assert_contains(streamed["content"], EXPECTED_CHAT_PHRASE, "streamed content")


FIM_CONTROL_TOKENS = ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")


@pytest.mark.e2e
def test_fim_completion(base_url: str) -> None:
    """A FIM completion returns the missing middle line without leaking control tokens."""
    completion_text = text_completion_fim(prefix=FIM_PREFIX, suffix=FIM_SUFFIX, base_url=base_url)
    choices = completion_text["choices"]
    assert len(choices) == 1, f"expected one choice, got {len(choices)}"
    text = choices[0]["text"]
    for control_token in FIM_CONTROL_TOKENS:
        assert control_token not in text, f"FIM control token {control_token} leaked into the completion"
    _assert_contains(" ".join(text.split()), " ".join(FIM_MIDDLE.split()), "FIM completion")


@pytest.mark.e2e
def test_fim_max_tokens_bound(base_url: str) -> None:
    """A FIM completion with a small max_tokens budget terminates with finish_reason length."""
    completion_text = text_completion_fim(prefix=FIM_PREFIX, suffix=FIM_SUFFIX,
                                          base_url=base_url, max_tokens=5)
    choice = completion_text["choices"][0]
    assert choice["finish_reason"] == "length", choice["finish_reason"]
