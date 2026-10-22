"""Concurrency and logging tests of the VLM controller request lock."""
import logging
import threading
import time
import unittest

from openvino_genai import GenerationConfig

from agent.inference.token_handler import StopSignal, TokenHandlerConfig
from agent.multimodal.media_inputs import MediaInputs
from agent.openai import GenerateOpts, engine_rest_vlm
from agent.openai.chat_api import ROLE_ASSISTANT, new_chat_completion_chunk
from agent.openai.engine_rest_common import ControllerConfig
from agent.openai.engine_rest_vlm import VlmController

# Both the engine logs to the module logger and the controller logs to the inference logger.
LOG_NAMES = ("agent.openai.engine_rest_vlm", "agent.inference")


class FakePipe:  # pylint: disable=too-few-public-methods  # duck-typed stub
    """Duck-typed pipe: VlmController only calls get_tokenizer on init."""

    def get_tokenizer(self):
        """Return a tokenizer-less pipe handle."""
        return None


class FakeTokenHandler:
    """Just enough of TokenHandler for the streamer and the consumer loop."""

    def __init__(self, stop_signal: StopSignal | None = None):
        self.stop_signal = stop_signal
        self.is_stop_flag = False

    def is_stop(self) -> bool:
        """Report stop only when the test set the flag, like a disconnected client."""
        return self.is_stop_flag

    def handle_tokens(self, _tokens):
        """Return one fixed assistant chunk for any token batch, with the configured stop signal."""
        return [new_chat_completion_chunk(role=ROLE_ASSISTANT, content="ok", model="m")], self.stop_signal

    def get_stat_info(self):
        """Return a placeholder stat line."""
        return "fake"


class LockedController(VlmController):
    """Overridden generate_result emulates a slow pipeline: the first generate blocks
    until `release` is set, like a long prefill that cannot honor a stop flag."""

    def __init__(self):
        super().__init__(ControllerConfig(max_prompt_len=4096, model_architectures=set(), model_name="m"),
                         parser=None,  # type: ignore[arg-type]  # LockedController never reaches parsing
                         pipe=FakePipe(), handler_config=TokenHandlerConfig(),
                         stop_signal=threading.Event(), generate_opts=GenerateOpts())
        self.order: list[str] = []
        self.lock_violations: list[str] = []
        self.first_generate_started = threading.Event()
        self.release = threading.Event()

    def generate_result(self, prompt, generation_config, streamer, media):
        is_first = not self.first_generate_started.is_set()
        self.order.append("generate-1-start" if is_first else "generate-2-start")
        if not self.request_lock.locked():
            self.lock_violations.append(f"{self.order[-1]} runs without request_lock")
        if is_first:
            self.first_generate_started.set()
        streamer.write([1])
        if is_first:
            self.release.wait(10)
        self.order.append("generate-1-end" if is_first else "generate-2-end")
        return "fake result"


def consume(controller: VlmController, sink: list, ready: threading.Event,  # pylint: disable=too-many-arguments,too-many-positional-arguments  # test helper options bundle
            close_after: int | None = None, closed: threading.Event | None = None,
            token_handler: FakeTokenHandler | None = None,
            stop_flag_after_first: bool = False):
    """Run the chunk generator to the end, optionally closing it early like a disconnected client."""
    handler = token_handler or FakeTokenHandler()
    gen = controller.chunk_generator(prompt="p", generation_config=GenerationConfig(),
                                     token_handler=handler,  # type: ignore[arg-type]  # duck-typed
                                     media=MediaInputs.empty())
    ready.set()
    for index, chunk in enumerate(gen):
        sink.append(chunk)
        if close_after is not None and index + 1 >= close_after:
            gen.close()  # the client went away mid-generation
            if closed:
                closed.set()
            break
        if stop_flag_after_first and index == 0:
            handler.is_stop_flag = True  # the consumer loop abandons the request after one chunk


class CaptureHandler(logging.Handler):
    """Collects log records from any thread so tests can assert on log messages."""

    def __init__(self):
        super().__init__()
        self.records: list[tuple[str, str]] = []  # (levelname, message)
        self.saved_levels: dict[str, int] = {}  # logger levels before the capture forced INFO
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord):
        with self._lock:
            self.records.append((record.levelname, record.getMessage()))

    def messages(self) -> list[tuple[str, str]]:
        """Return a snapshot of the captured records."""
        with self._lock:
            return list(self.records)

    def wait_for_message(self, fragment: str, timeout: float = 10) -> bool:
        """Block until a record containing the fragment appears, or time out."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if any(fragment in message for _, message in self.messages()):
                return True
            time.sleep(0.05)
        return False


def attach_capture() -> CaptureHandler:
    """Attach a capture handler to all engine loggers and force INFO, so INFO messages are emitted."""
    capture = CaptureHandler()
    for name in LOG_NAMES:
        logger = logging.getLogger(name)
        capture.saved_levels[name] = logger.level
        logger.setLevel(logging.INFO)
        logger.addHandler(capture)
    return capture


def detach_capture(capture: CaptureHandler):
    """Remove the capture handler from all engine loggers and restore their levels."""
    for name in LOG_NAMES:
        logger = logging.getLogger(name)
        logger.removeHandler(capture)
        logger.setLevel(capture.saved_levels.get(name, logging.NOTSET))


class VlmControllerRequestLockCase(unittest.TestCase):
    """Tests that request_lock serializes generate_result calls."""
    def test_generate_is_not_reentered_until_the_running_one_returns(self):
        """A long-running generate blocks the second call until the first returns."""
        controller = LockedController()

        first_chunks: list = []
        first_closed = threading.Event()
        first = threading.Thread(
            target=consume, args=(controller, first_chunks, threading.Event()),
            kwargs={"close_after": 1, "closed": first_closed}, daemon=True)
        first.start()
        self.assertTrue(controller.first_generate_started.wait(10), "first generate did not start")

        # an abandoned first request must keep the pipeline busy: generate-1 is still running,
        # so a second request must not reach generate even after the first generator is closed
        second_chunks: list = []
        second = threading.Thread(
            target=consume, args=(controller, second_chunks, threading.Event()), daemon=True)
        second.start()
        time.sleep(0.3)
        self.assertEqual(["generate-1-start"], controller.order)

        controller.release.set()  # generate-1 finally returns
        self.assertTrue(first_closed.wait(10), "first generator was not closed in time")

        deadline = time.time() + 10
        while "generate-2-start" not in controller.order and time.time() < deadline:
            time.sleep(0.05)
        second.join(10)

        self.assertEqual(["generate-1-start", "generate-1-end", "generate-2-start", "generate-2-end"],
                         controller.order)
        self.assertFalse(controller.lock_violations)
        self.assertEqual(1, len(first_chunks))
        self.assertEqual(2, len(second_chunks))  # the data chunk + the natural-end finish chunk
        self.assertEqual("stop", second_chunks[-1].choices[0].finish_reason)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())


class VlmFinishReasonLoggingCase(unittest.TestCase):
    """NONE finish reasons are expected when the streamer stopped the generation, unexpected otherwise."""

    def test_streamer_stopped_generation_is_not_reported_as_unexpected(self):
        """All the streamer stop paths keep the completion at INFO with an explaining message."""
        for signal in (StopSignal.STOP, StopSignal.TOOL_CALL, StopSignal.CANCEL):
            with self.subTest(stop_signal=signal):
                controller = LockedController()
                controller.release.set()  # generate returns right after the streamer word
                capture = attach_capture()
                try:
                    chunks: list = []
                    consume(controller, chunks, threading.Event(),
                            token_handler=FakeTokenHandler(stop_signal=signal))
                    self.assertTrue(capture.wait_for_message("inference stopped by the streamer"),
                                    "streamer-stop completion was not logged")
                finally:
                    detach_capture(capture)
                self.assertNotIn("unexpected status", (message for _, message in capture.messages()))

    def test_abandoned_generate_wait_reports_progress_not_error(self):
        """A generate abandoned while queued behind another one waits out with INFO progress lines.

        The pipeline cannot honor a stop during prefill and the queued call cannot even start,
        so the cleanup wait must never log the old 'waiting inference completion error'.
        """
        controller = LockedController()
        original_timeout = engine_rest_vlm.RESULT_WAIT_TIMEOUT_SECONDS
        engine_rest_vlm.RESULT_WAIT_TIMEOUT_SECONDS = 0.2
        capture = attach_capture()
        try:
            first = threading.Thread(
                target=consume, args=(controller, [], threading.Event()),
                kwargs={"stop_flag_after_first": True}, daemon=True)
            first.start()
            self.assertTrue(controller.first_generate_started.wait(10), "first generate did not start")

            # the abandoned request is still locked: the wait loop reports progress instead of erroring
            self.assertTrue(capture.wait_for_message("waiting for inference to complete"),
                            "the post-abandonment wait was never logged")

            controller.release.set()  # generate-1 finally returns, freeing the pipeline
            first.join(10)
            self.assertFalse(first.is_alive())

            messages = capture.messages()
            self.assertFalse([level for level, _ in messages if level == "ERROR"], messages)
            self.assertFalse(any("waiting inference completion error" in message for _, message in messages),
                             messages)
        finally:
            engine_rest_vlm.RESULT_WAIT_TIMEOUT_SECONDS = original_timeout
            detach_capture(capture)
            controller.release.set()  # unblock a leftover generate if an assertion failed midway


if __name__ == '__main__':
    unittest.main()
