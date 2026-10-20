"""Tests of the markdown helpers shared by the inference, client and REST layers."""
import unittest

from agent.common.markdown import markdown_back_tick, markdown_bold, markdown_file_content, \
    markdown_json, markdown_tool_call_loop_error


class MarkdownCase(unittest.TestCase):
    """Tests of the markdown formatting helpers."""

    def test_bold_wraps_the_value(self):
        """The bold helper wraps the value into the bold markers."""
        self.assertEqual("**text**", markdown_bold("text"))

    def test_back_tick_wraps_the_value(self):
        """The back tick helper wraps the value into the inline code markers."""
        self.assertEqual("`text`", markdown_back_tick("text"))

    def test_file_content_is_a_fenced_block(self):
        """The file content helper renders a fenced code block."""
        self.assertEqual("```\ntext\n```", markdown_file_content("text"))

    def test_file_content_carries_the_mime_type(self):
        """The mime type of the fenced block is the language of the block."""
        self.assertEqual("```json\n{}\n```", markdown_file_content("{}", "json"))

    def test_json_uses_the_json_language(self):
        """The json helper renders a json block."""
        self.assertEqual("```json\n{}\n```", markdown_json("{}"))

    def test_loop_error_without_a_cause_is_a_plain_block(self):
        """A loop error without a cause is rendered as a code block only."""
        self.assertEqual("```\nerror\n```", markdown_tool_call_loop_error(None, "error"))

    def test_loop_error_with_a_cause_is_marked_as_a_warning(self):
        """A loop error with a cause starts with a bold warning."""
        rendered = markdown_tool_call_loop_error("cause", "error")
        self.assertTrue(rendered.startswith("**WARNING: cause**"))
        self.assertIn("error", rendered)


if __name__ == '__main__':
    unittest.main()
