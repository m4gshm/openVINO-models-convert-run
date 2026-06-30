import unittest
from importlib.resources import files

from agent.client.veai.tool_call_fixer import fix_read_file
from agent.parser.gemma4_test import parser

TEST_RESOURCES = "test_resources"


class ReadFileTestCase(unittest.TestCase):

    def test_read_file_windows_path_delim_without_arg_name_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES,
                                                    "gemma4/read_file_windows_path_delim_without_arg_name.txt")
        tool_call_text = tool_cal_file.read_text(encoding="utf-8")
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        fixed = fix_read_file(calls[0])

        self.assertEqual({'end_line': 500,
                          'start_line': 1,
                          'target_file': 'C:/src/MessageStorageImpl.java'}, fixed.arguments)


if __name__ == '__main__':
    unittest.main()
