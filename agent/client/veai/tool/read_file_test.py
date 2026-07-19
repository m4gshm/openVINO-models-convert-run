import unittest
from importlib.resources import files

from agent.client.veai.tool_call_fixer import fix_read_file
from agent.parser import gemma4_test, qwen3_test

TEST_RESOURCES = "test_resources"


class ReadFileTestCase(unittest.TestCase):

    def test_read_file_windows_path_delim_without_arg_name_parse(self):
        parser = gemma4_test.parser
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES,
                                                    "gemma4/read_file_windows_path_delim_without_arg_name.txt")
        tool_call_text = tool_cal_file.read_text(encoding="utf-8")
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        fixed = fix_read_file(calls[0])

        self.assertEqual({'end_line': 500,
                          'start_line': 1,
                          'target_file': 'C:/src/MessageStorageImpl.java'}, fixed.arguments)

    def test_read_file_with_unexpected_text_in_int_parameters(self):
        parser = qwen3_test.parser
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES,
                                                    "qwen3_5/read_file_invalid_line_offset.txt")
        tool_call_text = tool_cal_file.read_text(encoding="utf-8")
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        fixed = fix_read_file(calls[0])

        self.assertEqual({'end_line': 100, 'start_line': 1, 'target_file': 'C:/test.txt'}, fixed.arguments)


if __name__ == '__main__':
    unittest.main()
