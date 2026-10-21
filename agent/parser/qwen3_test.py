"""Tests of the qwen3 tool call parser."""
import unittest
from importlib.resources import files

from agent.client.user_context import UserContext, OS
from agent.client.veai.tool_call_fixer import fix_list_dir, fix_edit_file, fix_write_file, fix_safe_delete, \
    fix_search_for_text, fix_search_file_by_name, fix_read_file, fix_run_command
from agent.openai.chat_completions_api import FunctionDefinition, ChatCompletionFunctionToolParam, \
    FunctionDefinitionParameters
from agent.openai.engine_rest_common import get_function_parameters_by_name
from agent.parser.qwen3 import EXPECTED_PROPERTY_TYPE, Qwen3MoeParser
from agent.parser import test_support
from agent.parser.test_support import parse_first_call


def new_windows_user_context() -> UserContext:
    """Build a Windows user context for the argument fixers."""
    context = UserContext()
    context.os_type = OS.Windows
    return context


USER_CONTEXT = new_windows_user_context()

TEST_RESOURCES = "test_resources"

parser = Qwen3MoeParser(test_support.FakeTokenizer())
state = parser.new_state()


def get_required_function_parameters(tool_json_file: str) -> dict[str, FunctionDefinitionParameters]:
    """Read the function parameters a tool declares in a test resource json."""
    tool_call_desc = files(__package__).joinpath(TEST_RESOURCES, tool_json_file)
    json_data = tool_call_desc.read_text()
    supported_function = FunctionDefinition.model_validate_json(json_data, strict=False, extra='allow')
    tools = [ChatCompletionFunctionToolParam(function=supported_function, type="function")]
    _, function_parameters = get_function_parameters_by_name(tools, True, True)
    return function_parameters


class Qwen3TestCases(unittest.TestCase):
    """Checks the tool call expressions of the qwen3 test resources."""

    def __init__(self, *args, **kwargs):
        """Prepare the fixtures of the test case."""
        super().__init__(*args, **kwargs)
        self.first_function_no_close_tag = """<tool_call>
<function=read_file>
<parameter=end_line>
75
</parameter>
<parameter=start_line>
19
</parameter>
<parameter=target_file>
Target.py
</parameter>
</function>"""
        self.first_function = self.first_function_no_close_tag + "</tool_call>"
        self.second_function = """<tool_call>
<function=ls>
<parameter=directory>
/tmp
</parameter>
</function>
</tool_call>"""
        self.function_with_invalid_json_parameter = """<tool_call>
<function=select>
<parameter=options>
[1,2,"3"],
</parameter>
</function>
</tool_call>"""

    def test_without_close_tag(self):
        """Check without close tag."""
        calls, _partial = parser.parse_tool_calls(state, self.first_function_no_close_tag)
        self.assertEqual(len(calls), 1)
        function_call = calls[0]
        self.assertEqual("read_file", function_call.name)
        self.assertEqual({"end_line": "75", "start_line": "19", "target_file": "Target.py"},
                         function_call.arguments)

    def test_two_functions(self):
        """Check two functions."""
        calls, partial = parser.parse_tool_calls(state, self.first_function + self.second_function)
        self.assertFalse(partial)
        self.assertEqual(len(calls), 2)

        first = calls[0]
        self.assertEqual("read_file", first.name)
        self.assertEqual({"end_line": "75", "start_line": "19", "target_file": "Target.py"},
                         first.arguments)

        second = calls[1]
        self.assertEqual("ls", second.name)
        self.assertEqual({"directory": "/tmp"}, second.arguments)

    def test_two_functions_where_first_without_close_tag(self):
        """Check two functions where first without close tag."""
        calls, _partial = parser.parse_tool_calls(state, self.first_function_no_close_tag + self.second_function)
        self.assertEqual(len(calls), 2)

        first = calls[0]
        self.assertEqual("read_file", first.name)
        self.assertEqual({"end_line": "75", "start_line": "19", "target_file": "Target.py"},
                         first.arguments)

        second = calls[1]
        self.assertEqual("ls", second.name)
        self.assertEqual({"directory": "/tmp"}, second.arguments)

    def test_functions_with_invalid_json_parameter(self):
        """Check functions with invalid json parameter."""
        function_name = "select"
        supported_functions: dict[str, FunctionDefinitionParameters] = {
            function_name: FunctionDefinitionParameters(properties={"options": {EXPECTED_PROPERTY_TYPE: "array"}})
        }
        typed_state = parser.new_state(supported_functions=supported_functions)
        calls, partial = parser.parse_tool_calls(typed_state, self.function_with_invalid_json_parameter)

        first = calls[0]
        self.assertEqual("select", first.name)
        self.assertEqual({"options": [1, 2, "3"]}, first.arguments)
        self.assertFalse(partial)

    def test_partial_tool_call(self):
        """Check partial tool call."""
        first, partial = parse_first_call(parser, state, __package__, "partially_generated_tool_call.txt")
        self.assertEqual("write_file", first.name)
        self.assertEqual({"allow_overwrite": "false", "content": "no finished parameter"}, first.arguments)
        self.assertTrue(partial)

    def test_search_file_by_name(self):
        """Check search file by name."""
        first, partial = parse_first_call(parser, state, __package__, "qwen3/search_file_by_name.txt")
        fixed = fix_search_file_by_name(first, USER_CONTEXT)
        self.assertEqual("search_file_by_name", fixed.name)
        self.assertEqual({'glob_pattern': 'Properties.*',
                          'search_directory': 'consumer/config'},
                         fixed.arguments)
        self.assertFalse(partial)

    def test_list_dir(self):
        """Check list dir."""
        first, partial = parse_first_call(parser, state, __package__, "qwen3/list_dir_1.txt")
        fixed = fix_list_dir(first, USER_CONTEXT)
        self.assertEqual("list_dir", fixed.name)
        self.assertEqual({'depth': '5',
                          'directory_path': 'java/idempotent-consumer-jdbc/src/test/resources'},
                         fixed.arguments)
        self.assertFalse(partial)

    def test_edit_file(self):
        """Check edit file."""
        edit_state = parser.new_state(supported_functions=get_required_function_parameters("qwen3/edit_file_tool.json"))
        first, partial = parse_first_call(parser, edit_state, __package__, "qwen3/edit_file_1.txt")
        fixed = fix_edit_file(first, USER_CONTEXT)
        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': 'dependencies {\r\n'
                                                 '    api(project(":idempotent-consumer"))\r\n'
                                                 '    api(project(":storage-api-reactive"))\r\n'
                                                 '    api(project(":postgres-jdbc"))\r\n'
                                                 '\r\n'
                                                 '    '
                                                 'implementation("io.projectreactor:reactor-core")\r\n'
                                                 '\r\n'
                                                 '    implementation("org.postgresql:postgresql")\r\n'
                                                 '\r\n'
                                                 '    '
                                                 'implementation("org.springframework.boot:spring-boot-starte'
                                                   'r-jooq")\r\n'
                                                 '    '
                                                 'implementation("org.springframework.boot:spring-boot-autoco'
                                                   'nfigure")\r\n'
                                                 '\r\n'
                                                 '    implementation("org.jooq:jooq")\r\n'
                                                 '    '
                                                 'implementation("org.jooq:jooq-postgres-extensions")\r\n'
                                                 '\r\n'
                                                 '    // Test containers for integration tests\r\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:testcontainers")\r\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:postgresql")\r\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:junit-jupiter")\r\n'
                                                 '}',
                                     'old_text': 'dependencies {\r\n'
                                                 '    api(project(":idempotent-consumer"))\r\n'
                                                 '    api(project(":storage-api-reactive"))\r\n'
                                                 '    api(project(":postgres-jdbc"))\r\n'
                                                 '\r\n'
                                                 '    '
                                                 'implementation("io.projectreactor:reactor-core")\r\n'
                                                 '\r\n'
                                                 '    implementation("org.postgresql:postgresql")\r\n'
                                                 '\r\n'
                                                 '    '
                                                 'implementation("org.springframework.boot:spring-boot-starte'
                                                   'r-jooq")\r\n'
                                                 '    '
                                                 'implementation("org.springframework.boot:spring-boot-autoco'
                                                   'nfigure")\r\n'
                                                 '\r\n'
                                                 '    implementation("org.jooq:jooq")\r\n'
                                                 '    '
                                                 'implementation("org.jooq:jooq-postgres-extensions")\r\n'
                                                 '}'}],
                          'target_file': 'C:/alex/github/m4gshm/distributed-transactions-practice/'
                                        'java/idempotent-consumer-jdbc/build.gradle.kts'},
                         fixed.arguments)
        self.assertFalse(partial)

    def test_write_file(self):
        """Check write file."""
        first, partial = parse_first_call(parser, state, __package__, "qwen3/write_file.txt")
        fixed = fix_write_file(first, USER_CONTEXT)
        self.assertEqual("write_file", fixed.name)
        self.assertEqual({'allow_overwrite': True,
                          'content': '[project]\n'
                                     'name = "openVINO-models-convert-run"\n'
                                     'version = "0.1.0"\n'
                                     'description = "OpenVINO models conversion and run utility"\n'
                                     'authors = [\n'
                                     '    {name = "Your Name", email = "your.email@example.com"}\n'
                                     ']',
                          'target_file': 'agent/pyproject.toml'},
                         fixed.arguments)
        self.assertFalse(partial)

    def test_run_command(self):
        """Check run command."""
        first, partial = parse_first_call(parser, state, __package__, "qwen3/run_command.txt")
        fixed = fix_run_command(first, USER_CONTEXT)
        self.assertEqual("run_command", fixed.name)
        self.assertEqual({'command': 'powershell -ExecutionPolicy Bypass -File _check_runtime.ps1',
                          'is_background': 'False',
                          'safe_to_run': 'False',
                          'working_directory': ''},
                         fixed.arguments)
        self.assertFalse(partial)

    def test_run_command_2(self):
        """Check run command 2."""
        first, partial = parse_first_call(parser, state, __package__, "qwen3/run_command_2.txt")
        fixed = fix_run_command(first, USER_CONTEXT)
        self.assertEqual("run_command", fixed.name)
        self.assertEqual({'command': 'Get-ChildItem -Recurse "C:/tmp" -ErrorAction SilentlyContinue | '
                                     "Where-Object { $_.FullName -match 'test' } | Select-Object "
                                     'FullName',
                          'is_background': 'False',
                          'safe_to_run': 'True',
                          'working_directory': '.'},
                         fixed.arguments)
        self.assertFalse(partial)

    def test_read_file(self):
        """Check read file."""
        first, partial = parse_first_call(parser, state, __package__, "qwen3/read_file.txt")
        fixed = fix_write_file(first, USER_CONTEXT)
        self.assertEqual("read_file", fixed.name)
        self.assertEqual({'end_line': 500,
                          'start_line': 1,
                          'target_file': 'C:/file.txt'},
                         fixed.arguments)
        self.assertFalse(partial)

    def test_read_file_2(self):
        """Check read file 2."""
        first, partial = parse_first_call(parser, state, __package__, "qwen3/read_file_2.txt")
        fixed = fix_read_file(first, USER_CONTEXT)
        self.assertEqual("read_file", fixed.name)
        self.assertEqual({'end_line': 100,
                          'start_line': 1,
                          'target_file': 'C:/MessageStorageImpl.java'},
                         fixed.arguments)
        self.assertFalse(partial)

    def test_move(self):
        """Check move."""
        move_state = parser.new_state(supported_functions=get_required_function_parameters("qwen3/move_tool.json"))
        first, partial = parse_first_call(parser, move_state, __package__, "qwen3/move.txt")
        self.assertEqual("move", first.name)
        self.assertEqual({'dry_run': 'true',
                          'sources': [
                              'java/idempotent-consumer-jdbc/src/test/java/io/github/m4gshm/idempotent/cons'
                                'umer/MessageStorageImplTest.java'],
                          'target_dir': 'java/idempotent-consumer-jdbc/src/integrationTest/java/io/github/m4gshm/'
                                        'idempotent/consumer'},
                         first.arguments)
        self.assertFalse(partial)

    def test_safe_delete(self):
        """Check safe delete."""
        delete_parameters = get_required_function_parameters("qwen3/safe_delete_tool.json")
        delete_state = parser.new_state(supported_functions=delete_parameters)
        first, partial = parse_first_call(parser, delete_state, __package__, "qwen3/safe_delete.txt")
        fixed = fix_safe_delete(first, USER_CONTEXT)
        self.assertEqual("safe_delete", fixed.name)
        self.assertEqual({'dry_run': False,
                          'force': False,
                          'targets': [{'line': 1,
                                       'name': 'MessageStorageTestConfig',
                                       'path': 'C:/Test.java'}]},
                         fixed.arguments)
        self.assertFalse(partial)

    def test_search_for_text(self):
        """Check search for text."""
        first, partial = parse_first_call(parser, state, __package__, "qwen3/search_for_text.txt")
        fixed = fix_search_for_text(first, USER_CONTEXT)
        self.assertEqual("search_for_text", fixed.name)
        self.assertEqual({'is_case_sensitive': False,
                          'target_path_or_url': '.',
                          'text_snippet': 'Testcontainers'},
                         fixed.arguments)
        self.assertFalse(partial)

    def test_search_for_text_2(self):
        """Check search for text 2."""
        first, partial = parse_first_call(parser, state, __package__, "qwen3/search_for_text_2.txt")
        fixed = fix_search_for_text(first, USER_CONTEXT)
        self.assertEqual("search_for_text", fixed.name)
        self.assertEqual({'is_case_sensitive': False,
                          'target_path_or_url': 'C:\\',
                          'text_snippet': 'testcontainers'},
                         fixed.arguments)
        self.assertFalse(partial)


    def test_search_for_text_3(self):
        """Check search for text 3."""
        first, partial = parse_first_call(parser, state, __package__, "qwen3/search_for_text_3.txt")
        fixed = fix_search_for_text(first, USER_CONTEXT)
        self.assertEqual("search_for_text", fixed.name)
        self.assertEqual({'is_case_sensitive': False,
                          'target_path_or_url': 'C:\\java',
                          'text_snippet': 'TestPropertySource'},
                         fixed.arguments)
        self.assertFalse(partial)

if __name__ == '__main__':
    unittest.main()
