import unittest
from importlib.resources import files

from agent.client.user_context import UserContext
from agent.client.veai.tool_call_fixer import fix_list_dir, fix_edit_file, fix_write_file
from agent.openai.chat_completions_api import FunctionDefinition
from agent.parser.qwen3 import EXPECTED_PARAMETERS_PROPERTIES, EXPECTED_PROPERTY_TYPE, Qwen3MoeParser

USER_CONTEXT = UserContext()
TEST_RESOURCES = "test_resources"

parser = Qwen3MoeParser()
state = parser.new_state()


class Qwen3TestCases(unittest.TestCase):

    def __init__(self, *args, **kwargs):
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
        calls, partial = parser.parse_tool_calls(state, self.first_function_no_close_tag)
        self.assertEqual(len(calls), 1)
        function_call = calls[0]
        self.assertEqual("read_file", function_call.name)
        self.assertEqual({"end_line": "75", "start_line": "19", "target_file": "Target.py"},
                         function_call.arguments)

    def test_two_functions(self):
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
        calls, partial = parser.parse_tool_calls(state, self.first_function_no_close_tag + self.second_function)
        self.assertEqual(len(calls), 2)

        first = calls[0]
        self.assertEqual("read_file", first.name)
        self.assertEqual({"end_line": "75", "start_line": "19", "target_file": "Target.py"},
                         first.arguments)

        second = calls[1]
        self.assertEqual("ls", second.name)
        self.assertEqual({"directory": "/tmp"}, second.arguments)

    def test_functions_with_invalid_json_parameter(self):
        function_name = "select"
        state = parser.new_state()
        state.supported_functions = {
            function_name: FunctionDefinition(name=function_name, parameters={
                EXPECTED_PARAMETERS_PROPERTIES: {"options": {EXPECTED_PROPERTY_TYPE: "array"}}
            })
        }
        calls, partial = parser.parse_tool_calls(state, self.function_with_invalid_json_parameter)

        first = calls[0]
        self.assertEqual("select", first.name)
        self.assertEqual({"options": [1, 2, "3"]}, first.arguments)
        self.assertFalse(partial)

    def test_partial_tool_call(self):
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "partially_generated_tool_call.txt")
        tool_call_text = tool_call_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        self.assertEqual("write_file", first.name)
        self.assertEqual({"allow_overwrite": "false", "content": "no finished parameter"}, first.arguments)
        self.assertTrue(partial)

    def test_search_file_by_name(self):
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "qwen3/search_file_by_name.txt")
        tool_call_text = tool_call_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        self.assertEqual("search_file_by_name", first.name)
        self.assertEqual({'glob_pattern': 'Properties.*',
                          'search_directory': 'consumer/config'},
                         first.arguments)
        self.assertFalse(partial)

    def test_list_dir(self):
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "qwen3/list_dir_1.txt")
        tool_call_text = tool_call_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        fixed = fix_list_dir(first, USER_CONTEXT)
        self.assertEqual("list_dir", fixed.name)
        self.assertEqual({'depth': '5',
                          'directory_path': 'java/idempotent-consumer-jdbc/src/test/resources'},
                         fixed.arguments)
        self.assertFalse(partial)

    def test_edit_file(self):
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "qwen3/edit_file_1.txt")
        tool_call_text = tool_call_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
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
                                                 'implementation("org.springframework.boot:spring-boot-starter-jooq")\r\n'
                                                 '    '
                                                 'implementation("org.springframework.boot:spring-boot-autoconfigure")\r\n'
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
                                                 'implementation("org.springframework.boot:spring-boot-starter-jooq")\r\n'
                                                 '    '
                                                 'implementation("org.springframework.boot:spring-boot-autoconfigure")\r\n'
                                                 '\r\n'
                                                 '    implementation("org.jooq:jooq")\r\n'
                                                 '    '
                                                 'implementation("org.jooq:jooq-postgres-extensions")\r\n'
                                                 '}'}],
                          'target_file': 'C:/alex/github/m4gshm/distributed-transactions-practice/java/idempotent-consumer-jdbc/build.gradle.kts'},
                         fixed.arguments)
        self.assertFalse(partial)

    def test_write_file(self):
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "qwen3/write_file.txt")
        tool_call_text = tool_call_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
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
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "qwen3/run_command.txt")
        tool_call_text = tool_call_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        fixed = fix_write_file(first, USER_CONTEXT)
        self.assertEqual("run_command", fixed.name)
        self.assertEqual({'command': 'powershell -ExecutionPolicy Bypass -File _check_runtime.ps1',
                          'is_background': 'False',
                          'safe_to_run': 'False',
                          'working_directory': ''},
                         fixed.arguments)
        self.assertFalse(partial)

    def test_read_file(self):
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "qwen3/read_file.txt")
        tool_call_text = tool_call_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        fixed = fix_write_file(first, USER_CONTEXT)
        self.assertEqual("read_file", fixed.name)
        self.assertEqual({'end_line': 500,
                          'start_line': 1,
                          'target_file': 'C:/file.txt'},
                         fixed.arguments)
        self.assertFalse(partial)

    def test_move(self):
        tool_call_desc = files(__package__).joinpath(TEST_RESOURCES, "qwen3/move_tool.json")
        json_data = tool_call_desc.read_text()
        supported_function = FunctionDefinition.model_validate_json(json_data)
        state = parser.new_state()
        state.supported_functions = {supported_function.name: supported_function}
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "qwen3/move.txt")
        tool_call_text = tool_call_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        self.assertEqual("move", first.name)
        self.assertEqual({'dry_run': 'true',
                          'sources': ['java/idempotent-consumer-jdbc/src/test/java/io/github/m4gshm/idempotent/consumer/MessageStorageImplTest.java'],
                          'target_dir': 'java/idempotent-consumer-jdbc/src/integrationTest/java/io/github/m4gshm/idempotent/consumer'},
                         first.arguments)
        self.assertFalse(partial)


if __name__ == '__main__':
    unittest.main()
