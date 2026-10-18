"""Tests of the tool call argument fixing of the token handler."""
import unittest
from importlib.resources import files

from agent.client.veai.tool_call_fixer import veai_fix_incorrect_arguments
from agent.parser import ParsedFunctionCall

TEST_RESOURCES = "test_resources"

class TokenHandlerCases(unittest.TestCase):
    """Checks the repair of the tool call arguments of the token handler."""

    def test_read_file_bad_options(self):
        """Check read file bad options."""
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "read_file_bad_options.json")
        tool_call_json = tool_call_file.read_text(encoding="utf-8")
        tool_call = ParsedFunctionCall.model_validate_json(tool_call_json)
        fixed = veai_fix_incorrect_arguments(tool_call)
        self.assertIsNotNone(fixed)
        self.assertEqual({"target_file": "/opt/test.txt", "start_line": 1, "end_line": 500}, fixed.arguments)
