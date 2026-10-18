"""Tests for edit_file tool-call argument fixing."""
import unittest
from importlib.resources import files
from pathlib import Path

from agent.client.user_context import UserContext, UserContextFiles
from agent.client.veai.tool_call_fixer import fix_edit_file
from agent.parser import gemma4_test
from agent.parser.test_support import (
    GRADLE_BUILD_KTS_NEW_TEXT,
    GRADLE_BUILD_KTS_OLD_TEXT,
)

TEST_RESOURCES = "test_resources"


class EditFileTestCase(unittest.TestCase):
    """Checks edit_file call repair against a real build file from resources."""

    def test_add_old_file(self):
        """new_text from file content must be rebuilt with reconstructed old_text."""
        src_file = files(__package__).joinpath(TEST_RESOURCES,
                                               "gemma4/build.gradle.kts")
        src_file_bytes = src_file.read_bytes()
        user_context = UserContext()
        user_context.files = UserContextFiles({Path("build.gradle.kts"): src_file_bytes})
        parser = gemma4_test.parser
        state = parser.new_state()
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES,
                                                     "gemma4/edit_file_no_old_text.txt")
        tool_call_text = tool_call_file.read_text()
        calls, _partial = parser.parse_tool_calls(state, tool_call_text)
        fixed = fix_edit_file(calls[0], context=user_context)

        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': GRADLE_BUILD_KTS_NEW_TEXT,
                                     'old_text': GRADLE_BUILD_KTS_OLD_TEXT}],
                          'target_file': 'build.gradle.kts'}, fixed.arguments)


if __name__ == '__main__':
    unittest.main()
