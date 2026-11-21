"""Proxy-mode e2e tests: OVMS + agent proxy path.

The proxy-mode tests carry the ``e2e-proxy`` marker and run only via
``-m e2e-proxy`` (default pytest config deselects them so ``task test``
never starts OVMS or a server).
"""
# pylint: disable=redefined-outer-name  # the session fixture and its test parameters share a name
# pylint: disable=duplicate-code  # the assertions intentionally mirror the native e2e suite
from collections.abc import Iterator

import pytest

from agent.e2e import (EXPECTED_CHAT_PHRASE, FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX, MODELS_DIR,
                       FetchFailure, OvmssHarness, ProxyServerHarness, chat_completion,
                       ensure_model, ovms_binary_path, parse_sse_response, text_completion_fim)

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
def proxy_urls() -> Iterator[tuple[str, str]]:
    """One OVMS + proxy pair for the whole e2e-proxy session."""
    if ovms_binary_path() is None:
        pytest.skip("OVMS binary unavailable (automatic download is Windows-only); set OVMS_BINARY")
    try:
        model = ensure_model()
    except FetchFailure as e:
        pytest.skip(str(e))
    ovms = OvmssHarness(model_path=model)
    try:
        ovms.start()
    except RuntimeError as e:
        pytest.skip(f"OVMS failed to start: {e}")
    # Then start proxy pointing to OVMS
    harness = ProxyServerHarness(ovms)
    try:
        harness.start()
        yield (harness.base_url, ovms.base_url)
    finally:
        harness.stop()


def _assert_contains(actual: str, expected: str, what: str) -> None:
    """Fail with both texts in the report so a mismatch is diagnosable alone."""
    assert expected in actual, \
        f"{what} does not contain the expected phrase\nexpected: {expected!r}\nactual: {actual!r}"


@pytest.mark.e2e_proxy
def test_proxy_chat_non_streaming(proxy_urls: tuple[str, str]) -> None:
    """A stream=false chat request through proxy answers with the expected phrase."""
    base_url, _ = proxy_urls
    response, _ = chat_completion(stream=False, base_url=base_url)
    choices = response["choices"]
    assert len(choices) == 1, f"expected one choice, got {len(choices)}"
    choice = choices[0]
    assert choice["message"]["role"] == "assistant"
    assert choice["finish_reason"] in ("stop", "length"), choice["finish_reason"]
    _assert_contains(choice["message"]["content"] or "", EXPECTED_CHAT_PHRASE, "chat content")


@pytest.mark.e2e_proxy
def test_proxy_chat_streaming(proxy_urls: tuple[str, str]) -> None:
    """A stream=true chat request through proxy ends with the sentinel."""
    base_url, _ = proxy_urls
    streamed, sentinel = chat_completion(stream=True, base_url=base_url)
    assert sentinel == "done", "stream ended without the data: [DONE] sentinel frame"
    assert streamed["finish_reason"] in ("stop", "length"), streamed["finish_reason"]
    _assert_contains(streamed["content"], EXPECTED_CHAT_PHRASE, "streamed content")


# A text-only FIM-specialized model: OVMS serves Qwen3.5-2B as a VLM that rejects /completions.
FIM_MODEL_DIR = MODELS_DIR / "Mellum-4b-base-int4-sym-g128-se-awq"
FIM_MELLUM = (FIM_MODEL_DIR.name, "<fim_prefix>", "<fim_suffix>", "<fim_middle>")


@pytest.fixture(scope="session")
def fim_urls() -> Iterator[tuple[str, str]]:
    """One OVMS + proxy pair serving the text-only FIM model."""
    if not FIM_MODEL_DIR.is_dir():
        pytest.skip(f"FIM model is not fetched locally: {FIM_MODEL_DIR}")
    ovms = OvmssHarness(model_path=FIM_MODEL_DIR)
    ovms.start()
    harness = ProxyServerHarness(ovms)
    try:
        harness.start()
        yield (harness.base_url, ovms.base_url)
    finally:
        harness.stop()


@pytest.mark.e2e_proxy
def test_proxy_fim_completion(fim_urls: tuple[str, str]) -> None:
    """A FIM completion through proxy returns the missing middle line."""
    base_url, _ = fim_urls
    completion_text = text_completion_fim(prefix=FIM_PREFIX, suffix=FIM_SUFFIX, base_url=base_url,
                                          fim=FIM_MELLUM)
    choices = completion_text["choices"]
    assert len(choices) == 1, f"expected one choice, got {len(choices)}"
    text = choices[0]["text"]
    for control_token in FIM_MELLUM[1:]:
        assert control_token not in text, f"FIM control token {control_token} leaked into the completion"
    _assert_contains(" ".join(text.split()), " ".join(FIM_MIDDLE.split()), "FIM completion")


@pytest.mark.e2e_proxy
def test_proxy_fim_max_tokens_bound(fim_urls: tuple[str, str]) -> None:
    """A token budget tighter than the answer terminates with finish_reason length."""
    base_url, _ = fim_urls
    completion_text = text_completion_fim(prefix=FIM_PREFIX, suffix=FIM_SUFFIX,
                                          base_url=base_url, max_tokens=2, fim=FIM_MELLUM)
    choice = completion_text["choices"][0]
    assert choice["finish_reason"] == "length", choice["finish_reason"]
