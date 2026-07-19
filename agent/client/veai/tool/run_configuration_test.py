import unittest
from importlib.resources import files

from agent.client.veai.tool_call_fixer import fix_run_configuration
from agent.parser.qwen3_test import parser

function_name = "run_configuration"

TEST_RESOURCES = "test_resources"


class RunConfigurationTestCase(unittest.TestCase):

    def test_read_file_windows_path_delim_without_arg_name_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES,
                                                    "qwen3_5/run_configuration.txt")
        tool_call_text = tool_cal_file.read_text(encoding="utf-8")
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        fixed = fix_run_configuration(calls[0])

        self.assertEqual({'configuration_environment_variables': [],
                          'configuration_name': 'build',
                          'configuration_run_arguments': [],
                          'files_to_collect_coverage': [],
                          'line_number': 0,
                          'target_file': 'build.gradle.kts',
                          'timeout': 1}, fixed.arguments)
