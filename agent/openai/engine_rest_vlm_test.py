import threading
import time
import unittest

from openvino_genai import GenerationConfig

from agent.inference.token_handler import TokenHandlerConfig
from agent.multimodal.media_inputs import MediaInputs
from agent.openai import GenerateOpts
from agent.openai.chat_api import ROLE_ASSISTANT, new_chat_completion_chunk
from agent.openai.engine_rest_common import ControllerConfig
from agent.openai.engine_rest_vlm import VlmController


class FakePipe:
    """Duck-typed pipe: VlmController only calls get_tokenizer on init."""

    def get_tokenizer(self):
        return None


class FakeTokenHandler:
    """Just enough of TokenHandler for the streamer and the consumer loop."""

    def is_stop(self) -> bool:
        return False

    def handle_tokens(self, _tokens):
        return [new_chat_completion_chunk(role=ROLE_ASSISTANT, content="ok", model="m")], None

    def get_stat_info(self):
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


def consume(controller: VlmController, sink: list, ready: threading.Event,
            close_after: int | None = None, closed: threading.Event | None = None):
    gen = controller.chunk_generator(prompt="p", generation_config=GenerationConfig(),
                                     token_handler=FakeTokenHandler(),  # type: ignore[arg-type]  # duck-typed
                                     media=MediaInputs.empty())
    ready.set()
    for index, chunk in enumerate(gen):
        sink.append(chunk)
        if close_after is not None and index + 1 >= close_after:
            gen.close()  # the client went away mid-generation
            if closed:
                closed.set()
            break


class VlmControllerRequestLockCase(unittest.TestCase):
    def test_generate_is_not_reentered_until_the_running_one_returns(self):
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
        self.assertEqual(1, len(second_chunks))
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())


if __name__ == '__main__':
    unittest.main()
