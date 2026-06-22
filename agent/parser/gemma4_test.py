import unittest

from agent.openai.chat_completions_api import ToolCall
from agent.parser.gemma4 import Gemma4ChannelParser

TEST_RESOURCES = "test_resources"

parser = Gemma4ChannelParser()


class TestAddFunction(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_function = """'<|tool_call>call:some_function{file1:<|"|>1.json<|"|>,2:<|"|>2.json<|"|>,mode:<|"|>strict<|"|>,timeout:30,empty_val:}<tool_call|>'"""

    def test_parsing(self):
        state = parser.new_state()
        calls, partial = parser.parse_tool_calls(state=state, tool_call_expression=self.first_function)
        self.assertEqual(len(calls), 1)
        function_call: ToolCall = calls[0]
        self.assertEqual("function", function_call.type)
        self.assertEqual("some_function", function_call.function.name)
        self.assertEqual(
            """{"file1": "1.json", "2": "2.json", "mode": "strict", "timeout": "30", "empty_val": ""}""",
            function_call.function.arguments)


if __name__ == '__main__':
    unittest.main()
