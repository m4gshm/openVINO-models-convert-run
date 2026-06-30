import unittest


class MyTestCase(unittest.TestCase):
    def test_read_file_bad_options(self):
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "read_file_bad_options.json")
        tool_call_json = tool_cal_file.read_text(encoding="utf-8")
        tool_call = ParsedFunctionCall.model_validate_json(tool_call_json)
        fixed = veai_fix_incorrect_arguments(tool_call)
        self.assertIsNotNone(fixed)
        self.assertEqual({"target_file": "/opt/test.txt", "start_line": 1, "end_line": 500},
                         fixed.arguments)
