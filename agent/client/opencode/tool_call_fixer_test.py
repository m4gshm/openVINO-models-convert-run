"""Tests for OpenCode tool call argument name normalization."""
import unittest

from agent.client.opencode.tool_call_fixer import opencode_fix_tool_calls
from agent.parser import ParsedFunctionCall


class TestOpenCodeToolCallFixer(unittest.TestCase):
    """Checks OpenCode tool call argument name normalization."""

    def test_edit_alternate_field_names(self):
        """Claude-style edit arguments must map to the canonical OpenCode names."""
        function = ParsedFunctionCall(
            name="edit",
            arguments={"file_path": "test.py", "old_string": "old", "new_string": "new"})
        fixed = opencode_fix_tool_calls(function)
        self.assertEqual(fixed.arguments["path"], "test.py")
        self.assertEqual(fixed.arguments["oldString"], "old")
        self.assertEqual(fixed.arguments["newString"], "new")

    def test_canonical_names_stay(self):
        """Canonical arguments must not be overwritten, aliases must be removed."""
        function = ParsedFunctionCall(
            name="edit",
            arguments={"path": "a.py", "oldString": "o", "newString": "n",
                       "file_path": "b.py", "old_text": "x"})
        fixed = opencode_fix_tool_calls(function)
        self.assertEqual(fixed.arguments["path"], "a.py")
        self.assertEqual(fixed.arguments["oldString"], "o")
        self.assertNotIn("file_path", fixed.arguments)
        self.assertNotIn("old_text", fixed.arguments)

    def test_subagent_alternate_field_names(self):
        """Subagent alternate names must map to the canonical ones."""
        function = ParsedFunctionCall(
            name="subagent", arguments={"session_id": "ses123", "agent_type": "general",
                                        "message": "do it"})
        fixed = opencode_fix_tool_calls(function)
        self.assertEqual(fixed.arguments["sessionID"], "ses123")
        self.assertEqual(fixed.arguments["agent"], "general")
        self.assertEqual(fixed.arguments["prompt"], "do it")

    def test_glob_alias(self):
        """`glob_pattern` must map to `pattern`."""
        function = ParsedFunctionCall(name="glob", arguments={"glob_pattern": "*.py"})
        fixed = opencode_fix_tool_calls(function)
        self.assertEqual(fixed.arguments["pattern"], "*.py")

    def test_normalization_is_tool_agnostic(self):
        """Alias normalization applies regardless of the tool name."""
        function = ParsedFunctionCall(name="unknown_tool", arguments={"file_path": "a.py"})
        fixed = opencode_fix_tool_calls(function)
        self.assertEqual(fixed.arguments, {"path": "a.py"})

    def test_missing_required_args_left_untouched(self):
        """Missing required arguments must not get placeholders: OpenCode reports the error."""
        function = ParsedFunctionCall(name="shell", arguments={})
        fixed = opencode_fix_tool_calls(function)
        self.assertEqual(fixed.arguments, {})


if __name__ == '__main__':
    unittest.main()
