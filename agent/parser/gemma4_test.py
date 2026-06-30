import unittest
from importlib.resources import files

from agent.openai.chat_completions_api import ToolCall
from agent.parser.gemma4 import Gemma4ChannelParser

TEST_RESOURCES = "test_resources"

parser = Gemma4ChannelParser()


class TestAddFunction(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normal_function = """'<|tool_call>call:some_function{file1:<|"|>1.json<|"|>,2:<|"|>2.json<|"|>,mode:<|"|>strict<|"|>,timeout:30,empty_val:}<tool_call|>'"""

    def test_parsing(self):
        state = parser.new_state()
        calls, partial = parser.parse_tool_calls(state=state, tool_call_expression=self.normal_function)
        self.assertEqual(len(calls), 1)
        function_call: ToolCall = calls[0]
        self.assertEqual("function", function_call.type)
        self.assertEqual("some_function", function_call.function.name)
        self.assertEqual(
            """{"file1": "1.json", "2": "2.json", "mode": "strict", "timeout": "30", "empty_val": ""}""",
            function_call.function.arguments)

    def test_wrapped_file_structure_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/file_structure_wrapped.txt")
        tool_call_text = tool_cal_file.read_text(encoding="utf-8")
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first: ToolCall = calls[0]
        self.assertEqual("function", first.type)
        self.assertEqual("file_function", first.function.name)
        self.assertEqual("""{"file_path": "dir/dir2/Foo.txt", "start_at": "1"}""", first.function.arguments)
        self.assertFalse(partial)


    def test_read_file_windows_path_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/read_file_windows_path.txt")
        tool_call_text = tool_cal_file.read_text(encoding="utf-8")
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first: ToolCall = calls[0]
        self.assertEqual("function", first.type)
        self.assertEqual("read_file", first.function.name)
        self.assertEqual("""{"start_line": "1", "end_line": "500", "file_path": "C:/src/MessageStorageImpl.java"}""", first.function.arguments)
        self.assertFalse(partial)


if __name__ == '__main__':
    unittest.main()
