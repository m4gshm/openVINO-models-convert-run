import unittest
from importlib.resources import files

from client.veai.tool_call_fixer import fix_incorrect_arguments
from common.openai_model import ToolCall

TEST_RESOURCES = "test_resources"


class MyTestCase(unittest.TestCase):
    def test_parse_bad_json(self):
        pass
        # tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "ask_user_with_options_bad_options.json")
        # tool_call_json = tool_cal_file.read_text(encoding="utf-8")
        # tool_call = ToolCall.model_validate_json(tool_call_json)
        # fixed = fix_incorrect_arguments(tool_call)
        # self.assertIsNotNone(fixed)
        # self.assertEqual("""{"is_multiple_choice": "false", "options": "[\\\"First\\\", \\\"Second\\\"]"}""",
        #                  fixed.function.arguments)


if __name__ == '__main__':
    unittest.main()
