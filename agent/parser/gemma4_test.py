import unittest
from importlib.resources import files

from agent.parser.gemma4 import Gemma4ChannelParser

TEST_RESOURCES = "test_resources"

parser = Gemma4ChannelParser()


class TestAddFunction(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normal_function = """'<|tool_call>call:some_function{file1:<|"|>1.json<|"|>,2:<|"|>2.json<|"|>,mode:<|"|>strict<|"|>,timeout:30,empty_val:}<tool_call|>'"""

    def test_parsing(self):
        state = parser.new_state()
        calls, partial = parser.parse_tool_calls(state=state, tool_call_expression=self.normal_function)
        self.assertEqual(len(calls), 1)
        function_call = calls[0]
        self.assertEqual("some_function", function_call.name)
        self.assertEqual(
            {"file1": "1.json", "2": "2.json", "mode": "strict", "timeout": "30", "empty_val": ""},
            function_call.arguments)

    def test_wrapped_file_structure_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/file_structure_wrapped.txt")
        tool_call_text = tool_cal_file.read_text(encoding="utf-8")
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        self.assertEqual("file_function", first.name)
        self.assertEqual({'file_path': 'dir/dir2/Foo.txt', 'start_at': 1}, first.arguments)
        self.assertFalse(partial)

    def test_read_file_windows_path_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/read_file_windows_path.txt")
        tool_call_text = tool_cal_file.read_text(encoding="utf-8")
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        self.assertEqual("read_file", first.name)
        self.assertEqual({"start_line": "1", "end_line": "500", "file_path": "C:/src/MessageStorageImpl.java"},
                         first.arguments)
        self.assertFalse(partial)

    def test_read_file_windows_path_delim_without_arg_name_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES,
                                                    "gemma4/read_file_windows_path_delim_without_arg_name.txt")
        tool_call_text = tool_cal_file.read_text(encoding="utf-8")
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        self.assertEqual("read_file", first.name)
        self.assertEqual({}, first.arguments)
        self.assertEqual(["C:/src/MessageStorageImpl.java"], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_read_file_like_json(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/read_file_lie_json.txt")
        tool_call_text = tool_cal_file.read_text(encoding="utf-8")
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        self.assertEqual("read_file", first.name)
        self.assertEqual({'end_line': '500',
                          'start_line': '1',
                          'target_file': 'build.gradle.kts'}, first.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_write_file_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/write_file.txt")
        tool_call_text = tool_cal_file.read_text(encoding="utf-8")
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        self.assertEqual("write_file", first.name)
        self.assertEqual({'allow_overwrite': 'true',
                          'content': 'plugins {\n'
                                     '    `java-library`\n'
                                     '}\n'
                                     'apply(plugin = "io.spring.dependency-management")\n'
                                     '\n'
                                     'dependencies {\n'
                                     '    api(project(":idempotent-consumer"))\n'
                                     '    api(project(":storage-api-reactive"))\n'
                                     '    api(project(":postgres-jdbc"))\n'
                                     '\n'
                                     '    implementation("io.projectreactor:reactor-core")\n'
                                     '\n'
                                     '    implementation("org.postgresql:postgresql")\n'
                                     '\n'
                                     '    '
                                     'implementation("org.springframework.boot:spring-boot-starter-jooq")\n'
                                     '    '
                                     'implementation("org.springframework.boot:spring-boot-autoconfigure")\n'
                                     '\n'
                                     '    implementation("org.jooq:jooq")\n'
                                     '    implementation("org.jooq:jooq-postgres-extensions")\n'
                                     '\n'
                                     '    // --- Testcontainers Dependencies ---\n'
                                     '    testImplementation("org.testcontainers:junit-jupiter")\n'
                                     '    testImplementation("org.testcontainers:postgresql")\n'
                                     '    testImplementation("org.testcontainers:os" // Для '
                                     'Podman/Docker runtime\n'
                                     '    testImplementation("org.junit.jupiter:junit-jupiter-api")\n'
                                     '    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine")\n'
                                     '}',
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'},
                         first.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_write_file_like_json(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/write_file_like_json.txt")
        tool_call_text = tool_cal_file.read_text(encoding="utf-8")
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        self.assertEqual("write_file", first.name)
        self.assertEqual({'allow_overwrite': 'true',
                          'content': 'plugins {\n'
                                     '    `java-library`\n'
                                     '}\n'
                                     'apply(plugin = "io.spring.dependency-management")\n'
                                     '\n'
                                     'dependencies {\n'
                                     '    api(project(":idempotent-consumer"))\n'
                                     '    api(project(":storage-api-reactive"))\n'
                                     '    api(project(":postgres-jdbc"))\n'
                                     '\n'
                                     '    implementation("io.projectreactor:reactor-core")\n'
                                     '\n'
                                     '    implementation("org.postgresql:postgresql")\n'
                                     '\n'
                                     '    '
                                     'implementation("org.springframework.boot:spring-boot-starter-jooq")\n'
                                     '    '
                                     'implementation("org.springframework.boot:spring-boot-autoconfigure")\n'
                                     '\n'
                                     '    implementation("org.jooq:jooq")\n'
                                     '    implementation("org.jooq:jooq-postgres-extensions")\n'
                                     '\n'
                                     '    // --- Testcontainers Dependencies ---\n'
                                     '    testImplementation("org.testcontainers:junit-jupiter")\n'
                                     '    testImplementation("org.testcontainers:postgresql")\n'
                                     '    testImplementation("org.testcontainers:os") // Для '
                                     'Podman/Docker runtime\n'
                                     '    testImplementation("org.junit.jupiter:junit-jupiter-api")\n'
                                     '    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine")',
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'},
                         first.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/edit_file.txt")
        tool_call_text = tool_cal_file.read_text(encoding="utf-8")
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        self.assertEqual("edit_file", first.name)
        self.assertEqual({'allow_multiple_matches': 'false',
                          'edits': [{'new_text': 'dependencies {\r\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:junit-jupiter")\r\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:postgresql")\r\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:containers")\r\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:lombok")\r\n'
                                                 '    '
                                                 'testImplementation("org.junit.jupiter:junit-jupiter-api")\r\n'
                                                 '    '
                                                 'testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine")\r\n'
                                                 '}',
                                     'old_text': 'dependencies {\r\n'
                                                 '    api(project(":idempotent-consumer"))\r\n'
                                                 '    api(project(":storage-api-reactive"))\r\n'
                                                 '    api(project(":postgres-jdbc"))\r\n'
                                                 '    '
                                                 'implementation("io.projectreactor:reactor-core")\r\n'
                                                 '    implementation("org.postgresql:postgresql")\r\n'
                                                 '    '
                                                 'implementation("org.springframework.boot:spring-boot-starter-jooq")\r\n'
                                                 '    '
                                                 'implementation("org.springframework.boot:spring-boot-autoconfigure")\r\n'
                                                 '    implementation("org.jooq:jooq")\r\n'
                                                 '    '
                                                 'implementation("org.jooq:jooq-postgres-extensions")\r\n'
                                                 '}'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'}, first.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)


if __name__ == '__main__':
    unittest.main()
