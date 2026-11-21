"""Tests for AgentType enum and DetectedSelectOptions."""
import unittest

from agent.client.tool_select_model import AgentType, DetectedSelectOptions, ToolSelectOptions
from agent.client.veai.tool.ask_user_with_options import AskUserWithOptions


class TestAgentType(unittest.TestCase):
    """Checks AgentType enum values."""

    def test_veai_value(self):
        self.assertEqual(AgentType.VEAI.value, "veai")

    def test_opencode_value(self):
        self.assertEqual(AgentType.OPENCODE.value, "opencode")


class TestDetectedSelectOptions(unittest.TestCase):
    """Checks DetectedSelectOptions holds agent + tool."""

    def test_holds_both_values(self):
        """Must store agent type and detected tool."""
        detector = AskUserWithOptions()
        result = DetectedSelectOptions(agent=AgentType.VEAI, tool=detector)
        self.assertEqual(result.agent, AgentType.VEAI)
        self.assertIsInstance(result.tool, ToolSelectOptions)
        self.assertEqual(result.tool.name, "ask_user_with_options")


if __name__ == '__main__':
    unittest.main()