"""Tests for ParsedFunctionCall conversion to the OpenAI tool-call shape."""
import unittest

from agent.parser import ParsedFunctionCall


class MyTestCase(unittest.TestCase):
    """Checks ParsedFunctionCall serialization for OpenAI tool calls."""

    def test_to_openai_tool_call(self):
        """Arguments must serialize to a JSON string with the tool name kept."""
        openai_tool_call = ParsedFunctionCall(name="foo", arguments={"arg": "val"}).to_openai_function_call()
        self.assertEqual("foo", openai_tool_call.name)
        self.assertEqual("""{\"arg\": \"val\"}""", openai_tool_call.arguments)


if __name__ == '__main__':
    unittest.main()
