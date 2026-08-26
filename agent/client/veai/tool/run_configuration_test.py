import unittest
from importlib.resources import files

from agent.client.veai.tool_call_fixer import fix_run_configuration
from agent.openai.chat_completions_api import FunctionDefinition
from agent.parser.qwen3_test import parser

function_name = "run_configuration"

TEST_RESOURCES = "test_resources"


class RunConfigurationTestCase(unittest.TestCase):

    def test_read_file_windows_path_delim_without_arg_name_parse(self):
        tool_call_definition_file = files(__package__).joinpath(TEST_RESOURCES,
                                                                "run_configuration_definition.json")
        tool_call_definition_json = tool_call_definition_file.read_text()
        function_def = FunctionDefinition.model_validate_json(tool_call_definition_json)
        state = parser.new_state()
        state.supported_functions = {function_def.name: function_def}
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES,
                                                     "qwen3_5/run_configuration.txt")
        tool_call_text = tool_call_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        fixed = fix_run_configuration(calls[0])

        self.assertEqual({'configuration_environment_variables': [],
                          'configuration_name': 'build',
                          'configuration_run_arguments': [],
                          'files_to_collect_coverage': [],
                          'line_number': 0,
                          'target_file': 'build.gradle.kts',
                          'timeout': 1}, fixed.arguments)
