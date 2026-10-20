"""Tests of the interrupt mode: a new /v1/completions request stops the running one."""
import asyncio
import threading
import time
import unittest
from argparse import ArgumentParser
from typing import cast
from unittest.mock import patch

from starlette.requests import Request

from agent.inference.token_handler import TokenHandlerConfig
from agent.openai import GenerateOpts
from agent.openai.chat_api import ROLE_ASSISTANT, new_chat_completion_chunk
from agent.openai.completions_api import CompletionRequest
from agent.openai.engine_rest_common import BaseOVController, ControllerConfig
from agent.openai.interrupt import InterruptRegistry
from agent.parser import Parser


class InterruptRegistryCase(unittest.TestCase):
    """Tests of the registry that keeps the stop event of the active request."""

    def test_new_request_interrupts_the_previous_one(self):
        """The start of a new request sets the stop event of the running one."""
        registry = InterruptRegistry()
        first = registry.start()
        second = registry.start()
        self.assertTrue(first.is_set(), "the previous request was not interrupted")
        self.assertFalse(second.is_set(), "the new request must not be interrupted")

    def test_finished_request_is_not_interrupted(self):
        """A request that already finished is not interrupted by the next one."""
        registry = InterruptRegistry()
        first = registry.start()
        registry.finish(first)
        second = registry.start()
        self.assertFalse(first.is_set(), "a finished request must not be interrupted")
        self.assertFalse(second.is_set())

    def test_finish_keeps_the_newer_request_active(self):
        """Finishing an interrupted request does not release the slot of the newer one."""
        registry = InterruptRegistry()
        first = registry.start()
        second = registry.start()
        registry.finish(first)
        self.assertFalse(registry.is_active(first))
        self.assertTrue(registry.is_active(second))

    def test_third_request_interrupts_the_second_one(self):
        """Only the running request is interrupted, the first one stays interrupted."""
        registry = InterruptRegistry()
        first = registry.start()
        second = registry.start()
        third = registry.start()
        self.assertTrue(first.is_set())
        self.assertTrue(second.is_set())
        self.assertFalse(third.is_set())


class FakeInputIds:  # pylint: disable=too-few-public-methods  # duck-typed stub
    """Minimal stand-in for the numpy array of encoded token ids."""

    size = 8


class FakeEncoding:  # pylint: disable=too-few-public-methods  # duck-typed stub
    """Minimal stand-in for the openvino_genai encoding result."""

    input_ids = FakeInputIds()


class FakeTokenizer:  # pylint: disable=too-few-public-methods  # duck-typed stub
    """Tokenizer stub: the controller only counts the prompt tokens with it."""

    def encode(self, _prompt):
        """Return a fake encoding with a fixed token amount."""
        return FakeEncoding()


class FakeController(BaseOVController):
    """Controller emitting chunks until the request is stopped or released."""

    def __init__(self, config: ControllerConfig):
        """Wire the controller with a fake tokenizer and no parser-specific state."""
        super().__init__(config, parser=Parser(), tokenizer=FakeTokenizer(),
                         handler_config=TokenHandlerConfig(), generate_opts=GenerateOpts(),
                         stop_signal=threading.Event())
        self.release = threading.Event()
        self.generating = threading.Event()
        self.interrupted = threading.Event()

    def chunk_generator(self, prompt, generation_config, token_handler, media):
        """Emit chunks until the request is interrupted or the test releases it."""
        while not token_handler.is_stop() and not self.release.is_set():
            self.generating.set()
            yield new_chat_completion_chunk(role=ROLE_ASSISTANT, content="x")
            time.sleep(0.02)
        if token_handler.is_stop():
            self.interrupted.set()


def new_config(**kwargs) -> ControllerConfig:
    """Controller config with the interrupt mode turned on."""
    return ControllerConfig(max_prompt_len=4096, model_architectures=set(), model_name="m", **kwargs)


def run_completions(controller: FakeController) -> list:
    """Call the completions endpoint and return the produced chunks."""
    body = CompletionRequest(prompt="hi", max_tokens=16)
    # the client connection is faked by the patched is_disconnected probe
    request = cast(Request, None)
    with patch("agent.openai.engine_rest_common.is_disconnected", return_value=False):
        response = asyncio.run(controller.completions(body, request))
    return response.choices[0].text


class InterruptCompletionsCase(unittest.TestCase):
    """Tests of the interrupt mode of the /v1/completions endpoint."""

    def test_new_request_interrupts_the_running_one(self):
        """The second request stops the generation of the first one."""
        controller = FakeController(new_config(interrupt_completions=True))
        first_result: list = []
        first = threading.Thread(target=lambda: first_result.append(run_completions(controller)), daemon=True)
        first.start()
        self.assertTrue(controller.generating.wait(10), "the first request did not start")

        second_result: list = []
        second = threading.Thread(target=lambda: second_result.append(run_completions(controller)), daemon=True)
        second.start()

        first.join(10)
        self.assertFalse(first.is_alive(), "the interrupted request did not finish")
        self.assertTrue(controller.interrupted.is_set(), "the first request was not interrupted")

        controller.release.set()
        second.join(10)
        self.assertFalse(second.is_alive(), "the second request did not finish")
        self.assertTrue(second_result and second_result[0], "the second request produced no text")

    def test_running_request_is_kept_when_the_mode_is_off(self):
        """Without the interrupt mode the running request is not stopped by a new one."""
        controller = FakeController(new_config(interrupt_completions=False))
        first = threading.Thread(target=lambda: run_completions(controller), daemon=True)
        first.start()
        self.assertTrue(controller.generating.wait(10), "the first request did not start")

        second = threading.Thread(target=lambda: run_completions(controller), daemon=True)
        second.start()
        time.sleep(0.5)

        self.assertFalse(controller.interrupted.is_set(), "the first request was interrupted unexpectedly")
        self.assertTrue(first.is_alive(), "the first request was stopped unexpectedly")

        controller.release.set()
        first.join(10)
        second.join(10)

    def test_interrupt_slot_is_released_when_the_request_ends(self):
        """A finished request does not interrupt the next one."""
        controller = FakeController(new_config(interrupt_completions=True))
        controller.release.set()
        run_completions(controller)
        self.assertFalse(controller.interrupt_registry.has_active(),
                         "the interrupt slot was not released")


class InterruptCliCase(unittest.TestCase):
    """Tests of the command line switch of the interrupt mode."""

    def test_mode_is_on_by_default(self):
        """The interrupt mode is enabled unless it is switched off."""
        parser = ArgumentParser()
        from agent.server_openvino import add_openvino_args  # pylint: disable=import-outside-toplevel
        add_openvino_args(parser)
        args = parser.parse_args(["--model", "m"])
        self.assertEqual("on", args.interrupt_completions)

    def test_mode_can_be_switched_off(self):
        """The interrupt mode is switched off from the command line."""
        parser = ArgumentParser()
        from agent.server_openvino import add_openvino_args  # pylint: disable=import-outside-toplevel
        add_openvino_args(parser)
        args = parser.parse_args(["--model", "m", "--interrupt_completions", "off"])
        self.assertEqual("off", args.interrupt_completions)

    def test_mode_can_be_switched_on(self):
        """The interrupt mode is switched on from the command line."""
        parser = ArgumentParser()
        from agent.server_openvino import add_openvino_args  # pylint: disable=import-outside-toplevel
        add_openvino_args(parser)
        args = parser.parse_args(["--model", "m", "--interrupt_completions", "on"])
        self.assertEqual("on", args.interrupt_completions)


if __name__ == '__main__':
    unittest.main()
