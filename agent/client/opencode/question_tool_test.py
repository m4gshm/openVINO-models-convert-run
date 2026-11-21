"""Tests for OpenCode 'question' tool detector."""
import unittest

from agent.client.opencode.question_tool import detect, QuestionTool, FUNCTION_NAME
from agent.openai.chat_completions_api import (
    ChatCompletionFunctionToolParam,
    FunctionDefinition,
    FunctionDefinitionParameters,
)


class TestOpenCodeQuestionDetector(unittest.TestCase):
    """Checks OpenCode 'question' tool detection."""

    def _make_tool_def(self) -> ChatCompletionFunctionToolParam:
        """Build a valid OpenCode 'question' tool definition."""
        return ChatCompletionFunctionToolParam(
            type="function",
            function=FunctionDefinition(
                name=FUNCTION_NAME,
                description="Ask the user a question with options",
                parameters=FunctionDefinitionParameters(
                    type="object",
                    required={"questions"},
                    properties={
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string"},
                                    "header": {"type": "string"},
                                    "options": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "label": {"type": "string"},
                                                "description": {"type": "string"},
                                            },
                                        },
                                    },
                                    "multiple": {"type": "boolean"},
                                },
                            },
                        },
                    },
                ),
            ),
        )

    def test_detect_matches_question_schema(self):
        """Must detect OpenCode 'question' tool."""
        tool_def = self._make_tool_def()
        result = detect(tool_def)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, QuestionTool)
        assert result is not None
        self.assertEqual(result.name, FUNCTION_NAME)

    def test_detect_wrong_name(self):
        """Must not detect a different function name."""
        tool_def = self._make_tool_def()
        tool_def.function.name = "ask_user_with_options"
        result = detect(tool_def)
        self.assertIsNone(result)

    def test_detect_no_questions_param(self):
        """Must not detect when 'questions' param is missing."""
        tool_def = self._make_tool_def()
        tool_def.function.parameters.properties = {}
        result = detect(tool_def)
        self.assertIsNone(result)

    def test_new_call_builds_valid_args(self):
        """new_call must produce a valid ParsedFunctionCall."""
        detector = QuestionTool()
        call = detector.new_call("Pick one", ["a", "b"], True)
        self.assertEqual(call.name, FUNCTION_NAME)
        self.assertEqual(len(call.arguments["questions"]), 1)
        q = call.arguments["questions"][0]
        self.assertEqual(q["question"], "Pick one")
        self.assertEqual(len(q["options"]), 2)
        self.assertEqual(q["options"][0]["label"], "a")
        self.assertTrue(q["multiple"])


if __name__ == '__main__':
    unittest.main()