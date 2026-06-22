import unittest
from importlib.resources import files

from agent.client.veai.tool_call_fixer import fix_incorrect_arguments
from agent.openai.chat_completions_api import ToolCall

TEST_RESOURCES = "test_resources"


class MyTestCase(unittest.TestCase):
    def test_read_file_bad_options(self):
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "read_file_bad_options.json")
        tool_call_json = tool_cal_file.read_text(encoding="utf-8")
        tool_call = ToolCall.model_validate_json(tool_call_json)
        fixed = fix_incorrect_arguments(tool_call)
        self.assertIsNotNone(fixed)
        self.assertEqual("""{"target_file": "/opt/test.txt", "start_line": 1, "end_line": 1000}""",
                         fixed.function.arguments)

    def test_write_file_bad_options(self):
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "write_file_bad_options.json")
        tool_call_json = tool_cal_file.read_text(encoding="utf-8")
        tool_call = ToolCall.model_validate_json(tool_call_json)
        fixed = fix_incorrect_arguments(tool_call)
        self.assertIsNotNone(fixed)
        self.assertEqual("""{"target_file": "/opt/test.txt", "content": "foo", "allow_overwrite": true}""",
                         fixed.function.arguments)


if __name__ == '__main__':
    unittest.main()
