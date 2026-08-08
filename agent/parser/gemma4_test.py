import unittest
from importlib.resources import files

from agent.client.veai.tool_call_fixer import fix_edit_file, fix_write_file
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
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        self.assertEqual("file_function", first.name)
        self.assertEqual({'file_path': 'dir/dir2/Foo.txt', 'start_at': 1}, first.arguments)
        self.assertFalse(partial)

    def test_search_for_text_space_delimited_args(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/search_for_text_space_delimited_args.txt")
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        self.assertEqual("search_for_text", first.name)
        self.assertEqual({'is_case_sensitive': 'false',
                          'target_path_or_url': 'MessageStorageImpl.java',
                          'text_snippet': '    "MessageStorageImpl"'}, first.arguments)
        self.assertFalse(partial)

    def test_read_file_windows_path_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/read_file_windows_path.txt")
        tool_call_text = tool_cal_file.read_text()
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
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        self.assertEqual("read_file", first.name)
        self.assertEqual({}, first.arguments)
        self.assertEqual(["C:/src/MessageStorageImpl.java"], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_read_file_like_json(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/read_file_like_json.txt")
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        self.assertEqual("read_file", first.name)
        self.assertEqual({'end_line': '500', 'start_line': '1', 'target_file': 'build.gradle.kts"}}'}, first.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_write_file_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/write_file.txt")
        tool_call_text = tool_cal_file.read_text()
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
                                     '    testImplementation("org.testcontainers:os")\n'
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
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        fixed = fix_write_file(first)
        self.assertEqual("write_file", fixed.name)
        self.maxDiff = None
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
                                     '    testImplementation("org.testcontainers:os")\n'
                                     '    testImplementation("org.junit.jupiter:junit-jupiter-api")\n'
                                     '    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine")\n'
                                     '}',
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'},
                         fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_write_file_like_json_2(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/write_file_like_json_2.txt")
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        fixed = fix_write_file(first)
        self.assertEqual("write_file", first.name)
        self.maxDiff = None
        self.assertEqual({'allow_overwrite': True,
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
                                     '    // Testcontainers dependencies for integration testing\n'
                                     '    testImplementation("org.testcontainers:postgresql")\n'
                                     '    testImplementation("org.testcontainers:junit-jupiter")\n'
                                     '}',
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'},
                         fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/edit_file.txt")
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        self.assertEqual("edit_file", first.name)
        self.assertEqual({'allow_multiple_matches': 'false',
                          'edits': [{},
                                    {'new_text': 'dependencies {\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:junit-jupiter")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:postgresql")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:containers")\n'
                                                 '    testImplementation("org.testcontainers:lombok")\n'
                                                 '    '
                                                 'testImplementation("org.junit.jupiter:junit-jupiter-api")\n'
                                                 '    '
                                                 'testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine")\n'
                                                 '}',
                                     'old_text': 'dependencies {\n'
                                                 '    api(project(":idempotent-consumer"))\n'
                                                 '    api(project(":storage-api-reactive"))\n'
                                                 '    api(project(":postgres-jdbc"))\n'
                                                 '    implementation("io.projectreactor:reactor-core")\n'
                                                 '    implementation("org.postgresql:postgresql")\n'
                                                 '    '
                                                 'implementation("org.springframework.boot:spring-boot-starter-jooq")\n'
                                                 '    '
                                                 'implementation("org.springframework.boot:spring-boot-autoconfigure")\n'
                                                 '    implementation("org.jooq:jooq")\n'
                                                 '    '
                                                 'implementation("org.jooq:jooq-postgres-extensions")\n'
                                                 '}'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'}, first.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file2_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/edit_file_2.txt")
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]

        fixed = fix_edit_file(first)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': True,
                          'edits': [{'new_text': '\n'
                                                 '    implementation("org.jooq:jooq")\n'
                                                 '    '
                                                 'implementation("org.jooq:jooq-postgres-extensions")\n'
                                                 '\n'
                                                 '    // Testing Dependencies (JUnit 5 & '
                                                 'Testcontainers)\n'
                                                 '    '
                                                 'testImplementation("org.junit.jupiter:junit-jupiter-api")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:postgresql")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:junit-jupiter")\n'
                                                 '}',
                                     'old_text': '    implementation("org.jooq:jooq")\n'
                                                 '    '
                                                 'implementation("org.jooq:jooq-postgres-extensions")'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'}, fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file3_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/edit_file_3.txt")
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]

        fixed = fix_edit_file(first)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': True,
                          'edits': [{'allow_multiple_matches': 'false',
                                     'new_text': 'testImplementation {\n'
                                                 '    // Testing Frameworks\n'
                                                 '    '
                                                 'testImplementation("org.junit.jupiter:junit-jupiter-api")\n'
                                                 '    '
                                                 'testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine")\n'
                                                 '\n'
                                                 '    // Spring Boot Test Utilities\n'
                                                 '    '
                                                 'testImplementation("org.springframework.boot:spring-boot-starter-test")\n'
                                                 '\n'
                                                 '    // Testcontainers Dependency for PostgreSQL\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:junit-jupiter")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:postgresql"),\n'
                                                 '\n'
                                                 '    // If the production code uses '
                                                 'reactor/netty/etc., these might be useful for '
                                                 'reactive testing\n'
                                                 '    implementation("io.projectreactor:reactor-test")\n'
                                                 '}',
                                     'old_text': '/* End of file or relevant marker */'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'}, fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file4_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/edit_file_4.txt")
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]

        fixed = fix_edit_file(first)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': True,
                          'edits': [{'new_text': 'test {\n'
                                                 '    useTestcontainers = true\n'
                                                 '\n'
                                                 '    dependencies {\n'
                                                 '        // Testing framework\n'
                                                 '        '
                                                 'testImplementation("org.junit.jupiter:junit-jupiter-api:5.10.2")\n'
                                                 '        '
                                                 'testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine")\n'
                                                 '\n'
                                                 '        // Testcontainers for PostgreSQL database\n'
                                                 '        '
                                                 'testImplementation("org.testcontainers:postgresql:1.19.8")\n'
                                                 '        '
                                                 'testImplementation("org.testcontainers:junit-jupiter:1.19.8")\n'
                                                 '\n'
                                                 '        // Optional: Logging if needed for test '
                                                 'setup\n'
                                                 '        '
                                                 'testImplementation("ch.qos.logback:logback-classic")\n'
                                                 '    }\n'
                                                 '}',
                                     'old_text': '\n'
                                                 '    '
                                                 'implementation("org.jooq:jooq-postgres-extensions")\n'
                                                 '}'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'}, fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file5_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/edit_file_5.txt")
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]

        fixed = fix_edit_file(first)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': True,
                          'edits': [{'new_text': 'null', 'old_text': '// ...'}, {'old_text': ''}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'}, fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file6_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/edit_file_6.txt")
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]

        fixed = fix_edit_file(first)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': True,
                          'edits': [{'new_text': 'testImplementation("org.testcontainers:junit-jupiter")',
                                     'old_text': 'implementation("org.jooq:jooq-postgres-extensions")'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'}, fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file7_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/edit_file_7.txt")
        tool_call_text = tool_cal_file.read_text(encoding="utf-8")
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]

        fixed = fix_edit_file(first)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': True,
                          'edits': [{'new_text': '    '
                                                 'testImplementation("org.testcontainers:junit-jupiter")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:postgresql")\n'
                                                 '}',
                                     'old_text': '}'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'}, fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file8_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/edit_file_8.txt")
        tool_call_text = tool_cal_file.read_text(encoding="utf-8")
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]

        fixed = fix_edit_file(first)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': 'false',
                          'edits': [{'new_text': '    implementation("org.jooq:jooq")\n'
                                                 '    '
                                                 'implementation("org.jooq:jooq-postgres-extensions")\n'
                                                 '\n'
                                                 '    '
                                                 'testImplementation("org.junit.jupiter:junit-jupiter-api:5.10.1")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:junit-jupiter:1.19.3")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:postgresql:1.19.3")\n'
                                                 '}',
                                     'old_text': '    implementation("org.jooq:jooq")\n'
                                                 '    '
                                                 'implementation("org.jooq:jooq-postgres-extensions")\n'
                                                 '}'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'}, fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file9_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/edit_file_9.txt")
        tool_call_text = tool_cal_file.read_text(encoding="utf-8")
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]

        fixed = fix_edit_file(first)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': True,
                          'edits': [{'new_text': '\n'
                                                 'testImplementation("org.junit.jupiter:junit-jupiter-api:5.10.2")\n'
                                                 'testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.10.2")\n'
                                                 'testImplementation("org.testcontainers:postgresql:1.19.7")\n'
                                                 'testImplementation("org.testcontainers:junit-jupiter:1.19.7")\n'
                                                 '}',
                                     'old_text': '}\n'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'}, fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)


if __name__ == '__main__':
    unittest.main()
