"""Tests of the token handler: the stop-phrase filter and tool call argument fixing."""
import unittest
from datetime import timedelta
from importlib.resources import files

from agent.client.veai.tool_call_fixer import veai_fix_incorrect_arguments
from agent.inference.phrase import DEFAULT_DUPLICATED_TOKENS_LIMIT
from agent.inference.token_handler import (StopSignal, TokenHandler, TokenHandlerConfig,
                                           fix_tool_calls_and_convert_to_choice_delta_tool_calls)
from agent.parser import ParsedFunctionCall, Parser, StateEvent

TEST_RESOURCES = "test_resources"


class ToolCallIndexCases(unittest.TestCase):
    """Streamed parallel tool calls must carry distinct, sequential indexes."""

    def test_indexes_follow_start_index(self):
        """Calls converted with a start offset get sequential indexes, not all zero."""
        calls = [ParsedFunctionCall(name="glob", arguments={"pattern": "a"}),
                 ParsedFunctionCall(name="read", arguments={"path": "b"})]
        deltas = fix_tool_calls_and_convert_to_choice_delta_tool_calls(None, None, calls, start_index=2)

        self.assertEqual([2, 3], [delta.index for delta in deltas])


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

    def test_stop_also_applies_in_chat_mode(self):
        """The live completion handlers are chat-mode handlers; the stop must cut them too."""
        handler = TokenHandler(tokenizer=None, prompt="", parser=Parser(), init_chat_events=True,
                               is_stop=None, config=TokenHandlerConfig(stop_strings=("END",)),
                               tool_fixer=None)
        result, stop_signal = handler.processor.process_tokens(["fine ", "END", "never"])

        self.assertEqual(["fine "], self.chunks_text(result))
        self.assertEqual(StopSignal.STOP, stop_signal)


    def test_no_role_warning_in_completion_mode(self):
        """Completion mode has no role tokens: role None must not warn nor suppress output."""
        with self.assertNoLogs("agent.inference.token_handler", level="WARNING"):
            handler = self.handler(())
            result, stop_signal = handler.processor.process_tokens(["fine ", "never"])
        self.assertEqual(["fine ", "never"], self.chunks_text(result))
        self.assertIsNone(stop_signal)

    def test_unexpected_role_still_warns_in_chat_mode(self):
        """In chat mode a non-assistant role keeps the unexpected-role warning."""
        with self.assertLogs("agent.inference.token_handler", level="WARNING") as logs:
            handler = TokenHandler(tokenizer=None, prompt="", parser=Parser(), init_chat_events=True,
                                   is_stop=None, config=TokenHandlerConfig(), tool_fixer=None)
            handler.processor.state.role = "user"
            handler.processor.process_tokens(["hi"])
        self.assertTrue([m for m in logs.output if "unexpected role user" in m])

    def test_looped_generation_streams_a_warning_and_cancels(self):
        """A looped token run ends the stream with the loop warning and a CANCEL stop."""
        handler = self.handler(())
        result, stop_signal = handler.processor.process_tokens(["x"] * (DEFAULT_DUPLICATED_TOKENS_LIMIT + 1))

        self.assertEqual(StopSignal.CANCEL, stop_signal)
        text = "".join(self.chunks_text(result))
        self.assertIn("WARNING: Generated content appears to be a loop", text)


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


class ToolCallParsingWarningsCase(unittest.TestCase):
    """Warnings emitted while a tool call is still being parsed."""

    def test_parsing_time_limit_warns_once(self):
        """Exceeding the parsing time limit warns once and does not repeat per token."""
        config = TokenHandlerConfig(tool_call_parting_duration_limit=timedelta(0),
                                    tool_call_parting_duration_warning=timedelta(days=3650))
        processor = TokenHandler(tokenizer=None, prompt="", parser=Parser(), init_chat_events=True,
                                 is_stop=None, config=config, tool_fixer=None).processor
        processor.state.start_event(StateEvent.TOOL_CALL)
        processor.tool_call_parsing_start_time = 0.0

        first, _ = processor.process_token("x", 1)
        second, _ = processor.process_token("y", 2)

        first_text = "".join(chunk.choices[0].delta.content or "" for chunk in first if chunk.choices)
        self.assertIn("Tool call parsing exceeded time limit", first_text)
        self.assertEqual([], second)
