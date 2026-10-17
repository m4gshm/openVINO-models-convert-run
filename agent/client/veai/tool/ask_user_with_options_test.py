"""Tests for ask_user_with_options tool detection."""
import unittest
from importlib.resources import files

from agent.client.veai.tool.ask_user_with_options import detect
from agent.openai.chat_completions_api import ChatCompletionFunctionToolParam

TEST_RESOURCES = "test_resources"


class AskUserWithOptionsTestCase(unittest.TestCase):
    """Checks detection of ask_user_with_options tool definitions."""

    def test_parse_bad_json(self):
        """A well-formed ask_user_with_options definition must be detected."""
        tool_def_file = files(__package__).joinpath(TEST_RESOURCES, "ask_user_with_options.json")
        tool_def_json = tool_def_file.read_text(encoding="utf-8")
        tool_def = ChatCompletionFunctionToolParam.model_validate_json(tool_def_json)
        ask_user_with_options = detect(tool_def)
        self.assertIsNotNone(ask_user_with_options)


if __name__ == '__main__':
    unittest.main()
