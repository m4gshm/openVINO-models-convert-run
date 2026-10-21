"""Tests of the token handler: the stop-phrase filter and tool call argument fixing."""
import unittest
from importlib.resources import files

from agent.client.veai.tool_call_fixer import veai_fix_incorrect_arguments
from agent.inference.token_handler import StopSignal, TokenHandler, TokenHandlerConfig
from agent.parser import ParsedFunctionCall, Parser

TEST_RESOURCES = "test_resources"


class CompletionStopPhrasesCases(unittest.TestCase):
    """Stop-word handling of the non-chat (completion) token stream."""

    def handler(self, stop_strings: tuple[str, ...]) -> TokenHandler:
        """A completion-mode handler fed by a plain parser. Tokens go through handler.processor."""
        return TokenHandler(tokenizer=None, prompt="", parser=Parser(), init_chat_events=False,
                            is_stop=None,
                            config=TokenHandlerConfig(stop_strings=stop_strings), tool_fixer=None)

    def chunks_text(self, result) -> list[str]:  # type: ignore[no-untyped-def]  # the handler's chunk union
        """Content of every chunk of the handler result."""
        return [chunk.choices[0].delta.content or "" for chunk in result if chunk.choices]

    def test_stop_token_ends_generation(self):
        """A generated token equal to a stop phrase is not emitted and stops the stream."""
        handler = self.handler(("«STOP»",))
        result, stop_signal = handler.processor.process_tokens(["fine ", "«STOP»", "never"])

        self.assertEqual(["fine "], self.chunks_text(result))
        self.assertEqual(StopSignal.STOP, stop_signal)

    def test_stop_without_stops_streams_everything(self):
        """Without stop phrases the handler streams all tokens."""
        handler = self.handler(())
        result, stop_signal = handler.processor.process_tokens(["fine ", "«STOP»", "never"])

        self.assertEqual(["fine ", "«STOP»", "never"], self.chunks_text(result))
        self.assertIsNone(stop_signal)

    def test_stop_is_exact_token_match_only(self):
        """A token containing but not equal to the stop phrase streams on."""
        handler = self.handler(("STOP",))
        result, stop_signal = handler.processor.process_tokens(["STOPword"])

        self.assertEqual(["STOPword"], self.chunks_text(result))
        self.assertIsNone(stop_signal)


class TokenHandlerCases(unittest.TestCase):
    """Checks the repair of the tool call arguments of the token handler."""

    def test_read_file_bad_options(self):
        """Check read file bad options."""
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "read_file_bad_options.json")
        tool_call_json = tool_call_file.read_text(encoding="utf-8")
        tool_call = ParsedFunctionCall.model_validate_json(tool_call_json)
        fixed = veai_fix_incorrect_arguments(tool_call)
        self.assertIsNotNone(fixed)
        self.assertEqual({"target_file": "/opt/test.txt", "start_line": 1, "end_line": 500}, fixed.arguments)
