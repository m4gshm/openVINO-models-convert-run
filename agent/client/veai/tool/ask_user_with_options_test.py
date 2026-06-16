import unittest
from importlib.resources import files

from common.openai_model import ToolDefinition
from veai.tool.ask_user_with_options import detect

TEST_RESOURCES = "test_resources"


class MyTestCase(unittest.TestCase):
    def test_parse_bad_json(self):
        pass
        tool_def_file = files(__package__).joinpath(TEST_RESOURCES, "ask_user_with_options.json")
        tool_def_json = tool_def_file.read_text(encoding="utf-8")
        tool_def = ToolDefinition.model_validate_json(tool_def_json)
        ask_user_with_options = detect(tool_def)
        self.assertIsNotNone(ask_user_with_options)


if __name__ == '__main__':
    unittest.main()
