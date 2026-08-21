import unittest
from importlib.resources import files

from agent.parser.lfm2 import Lfm2Parser

TEST_RESOURCES = "test_resources"

parser = Lfm2Parser()
state = parser.new_state()


class Lfm2TestCases(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_list_dir(self):
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "lfm2/list_dir.txt")
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        self.assertEqual("list_dir", first.name)
        self.assertEqual({'depth': 0, 'directory_path': 'C:\\1\\2\\3\\4'},
                         first.arguments)
        self.assertFalse(partial)


if __name__ == '__main__':
    unittest.main()
