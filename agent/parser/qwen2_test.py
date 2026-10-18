"""Tests for the qwen2 tool call parser."""
import unittest
from importlib.resources import files

from agent.client.veai.tool_call_fixer import fix_edit_file
from agent.parser.qwen2 import Qwen2Parser
from agent.parser.test_support import (
    GRADLE_REACTOR_AND_SPRING_BOOT_DEPENDENCIES,
    GRADLE_REACTOR_AND_SPRING_BOOT_DEPENDENCIES_NO_EOL,
    GRADLE_TESTCONTAINERS_DEVELOPMENT,
)

TEST_RESOURCES = "test_resources"

parser = Qwen2Parser()
state = parser.new_state()


class Qwen2TestCases(unittest.TestCase):
    """Checks the qwen2 tool call expressions of the test resources."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_edit_file(self):
        """Check the parsed and repaired arguments of the edit_file call."""
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "qwen2/veai_edit_file.txt")
        tool_call_text = tool_call_file.read_text(encoding="utf-8")
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]

        fixed = fix_edit_file(first, None)

        expected_edits = [{'new_text': GRADLE_REACTOR_AND_SPRING_BOOT_DEPENDENCIES + GRADLE_TESTCONTAINERS_DEVELOPMENT,
                           'old_text': GRADLE_REACTOR_AND_SPRING_BOOT_DEPENDENCIES_NO_EOL}]
        self.assertEqual("edit_file", first.name)
        self.assertEqual({'edits': expected_edits, 'target_file': 'C:/build.gradle.kts'}, first.arguments)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': expected_edits, 'target_file': 'C:/build.gradle.kts'},
                         fixed.arguments)
        self.assertFalse(partial)

    def test_parse_multiple_calls(self):
        """Check a stream with three tool calls, the last one not fully closed."""
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "qwen2/mutliple_tool_calls.txt")
        tool_call_text = tool_call_file.read_text(encoding="utf-8")
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        self.assertEqual(3, len(calls))
        self.assertTrue(partial)
        self.assertEqual("list_dir", calls[0].name)
        self.assertEqual({'directory_path': 'C:/'}, calls[0].arguments)

        self.assertEqual("search_file_by_name", calls[1].name)
        self.assertEqual({'glob_pattern': '**/*Test*',
                          'is_case_sensitive': True,
                          'search_directory': '.'}, calls[1].arguments)

        self.assertEqual("write_file", calls[2].name)
        self.assertEqual({'allow_overwrite': False, 'content': 'text', 'target_file': 'C:/test.txt'},
                         calls[2].arguments)


if __name__ == '__main__':
    unittest.main()
