import unittest
from importlib.resources import files

from agent.client.user_context import UserContext
from agent.client.veai.tool_call_fixer import fix_edit_file, fix_write_file, GEMMA_4

from agent.parser.gemma4 import Gemma4ChannelParser

USER_CONTEXT = UserContext(model_architectures={GEMMA_4})
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
        self.assertEqual({'is_case_sensitive': False,
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

    def test_get_configurations_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/get_configurations.txt")
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        fixed = fix_write_file(first)
        self.assertEqual("get_configurations", fixed.name)
        self.assertEqual({'configuration_name_for_part': 'test',
                          'include_global_configurations': True,
                          'page': '1',
                          'target_file': None}, fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_write_file_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/write_file.txt")
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        self.assertEqual("write_file", first.name)
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
        self.assertEqual({'allow_multiple_matches': False,
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

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
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

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'allow_multiple_matches': False,
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

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
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

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': '}', 'old_text': '// ...'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'}, fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file6_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/edit_file_6.txt")
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]

        fixed = fix_edit_file(first, USER_CONTEXT)

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
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
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
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
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
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
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

    def test_edit_file10_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/edit_file_10.txt")
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': '    '
                                                 'testImplementation("org.junit.jupiter:junit-jupiter-api:5.10.2")\n'
                                                 '    '
                                                 'testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.10.2")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:junit-jupiter")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:postgresql")}',
                                     'old_text': '    implementation("org.jooq:jooq")\n'
                                                 '    '
                                                 'implementation("org.jooq:jooq-postgres-extensions")\n'
                                                 '}'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'}, fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file11_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/edit_file_11.txt")
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': 'test {\n'
                                                 '    runtimeClasspath = sourceSets.test.copy {\n'
                                                 '        .+ dependencies {\n'
                                                 '            implementation '
                                                 'project(":storage-api-dependencies") // If '
                                                 'applicable\n'
                                                 '            '
                                                 'testImplementation("org.springframework.boot:allowed-versions")\n'
                                                 '            '
                                                 'testImplementation("org.postgresql:postgresql")\n'
                                                 '        }\n'
                                                 '    }\n'
                                                 '}',
                                     'old_text': '}'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'}, fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file12_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/edit_file_12.txt")
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'allow_multiple_matches': False,
                                     'new_text': '    '
                                                 'testImplementation("org.springframework.boot:spring-to-test")\n'
                                                 '    '
                                                 'testImplementation("net.bnd.system.container:postgresql-junit-container") '
                                                 '// Or the modern Testcontainers suite\n'
                                                 '    // Note: If using the unified Testcontainers '
                                                 'dependency, the specific library coordinates might '
                                                 "change. We'll use the standard ones for now.\n"
                                                 '    // We need a robust dependency for the actual '
                                                 'PostgreSQL container.\n'
                                                 "    // Let's use the common Spring Testcontainers "
                                                 'dependency approach if possible.\n'
                                                 '\n'
                                                 '    // Correcting to use the standard '
                                                 'Testcontainers dependency for simplicity and '
                                                 'compatibility:\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:postgresql:X.Y.Z")]',
                                     'old_text': '\n}'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'}, fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file13_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/edit_file_13.txt")
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': '    '
                                                 'implementation("org.jooq:jooq-postgres-extensions")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:postgresql")\n'
                                                 '    testImplementation("org.springframework.boot:" + '
                                                 '":0.0.0.0"',
                                     'old_text': '\n}'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'}, fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file14_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/edit_file_14.txt")
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': '\n'
                                                 '    '
                                                 'testImplementation("org.springframework.boot:spring-boot-starter-test)")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:postgresql") '
                                                 '// Testcontainers for PostgreSQL\n'
                                                 '}\n',
                                     'old_text': '\n}\n'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'}, fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file15_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/edit_file_15.txt")
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': '    implementation("org.jooq:jooq"\n'
                                                 '    '
                                                 'implementation("org.jooq:jooq-postgres-extensions")\n'
                                                 '\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:junit-bom")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:postgresql")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:junit-jupiter")\n'
                                                 '}',
                                     'old_text': '    implementation("org.jooq: "\n'
                                                 '    '
                                                 'implementation("org.jooq:jooq-postgres-extensions")\n'
                                                 '}'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'}, fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file16_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/edit_file_16.txt")
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': '    '
                                                 "testImplementation('org.springframework.boot:spring-boot-starter-test') "
                                                 '{ \n'
                                                 "        exclude group: 'org.mockito'\n"
                                                 '    }',
                                     'old_text': '    '
                                                 "implementation('org.springframework.boot:spring-boot-autoconfigure')",
                                     }],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'},
                         fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_write_file_2_parse(self):
        state = parser.new_state()
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "gemma4/write_file_2.txt")
        tool_call_text = tool_cal_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("write_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'allow_overwrite': True,
                          'content': 'package io.github.m4gshm.idempotent.consumer;\n'
                                     '\n'
                                     'import '
                                     'io.github.m4gshm.idempotent.consumer.storage.tables.InputMessages;\n'
                                     'import '
                                     'io.github.m4gshm.idempotent.consumer.storage.tables.MessageStorageMaintenanceService;\n'
                                     'import org.junit.jupiter.api.Test;\n'
                                     'import org.junit.jupiter.api.extension.ExtendWith;\n'
                                     'import org.springframework.beans.factory.annotation.Autowired;\n'
                                     'import org.springframework.boot.test.context.SpringBootTest;\n'
                                     'import org.springframework.test.context.ActiveProfiles;\n'
                                     'import java.time.LocalDate;\n'
                                     'import java.time.OffsetDateTime;\n'
                                     'import java.time.ZoneOffset;\n'
                                     'import java.util.UUID;\n'
                                     '\n'
                                     'import io.github.m4gshm.idempotent.consumer.Message;\n'
                                     'import '
                                     'io.github.m4gshm.idempotent.consumer.MessageStorageMaintenanceService;\n'
                                     'import io.github.m4gshm.idempotent.consumer.MessageStorageImpl;\n'
                                     'import '
                                     'io.github.m4gshm.idempotent.consumer.MessageAlreadyProcessedException;\n'
                                     'import org.springframework.test.context.TestPropertySource;\n'
                                     'import org.testcontainers.containers.PostgreSQLContainer;\n'
                                     'import org.testcontainers.junit.jupiter.Container;\n'
                                     '\n'
                                     'import static org.junit.jupiter.api.Assertions.*;\n'
                                     'import static org.mockito.Mockito.*;\n'
                                     '\n'
                                     '/** \n'
                                     ' * Integration tests for MessageStorageImpl using a real database '
                                     'managed by Testcontainers.\n'
                                     ' */ \n'
                                     '@SpringBootTest \n'
                                     '@ActiveProfiles("integration") // Assume an integration profile '
                                     'with testdb settings\n'
                                     '@TestPropertySource(name = "application-integration.properties") '
                                     '// Custom properties for DB connection\n'
                                     'public class MessageStorageImplIntegrationTest {\n'
                                     '\n'
                                     '    // --- Testcontainers Setup ---\n'
                                     '    // We use a dedicated PostgreSQL container for testing\n'
                                     '    @Container\n'
                                     '    static PostgreSQLContainer<?> postgres = new '
                                     'PostgreSQLContainer<>("postgres:14-alpine") {\n'
                                     '        @Override\n'
                                     '        protected void start() {\n'
                                     '            // Optional: Run migrations here if using '
                                     'Flyway/Liquibase\n'
                                     '        }\n'
                                     '    };\n'
                                     '\n'
                                     '    private final String dbUrl = "jdbc:postgresql:" + '
                                     'postgres.getJdbcUrl() + ""; // Placeholder\n'
                                     '\n'
                                     '    // NOTE: Due to the complexity of replacing DSLContext with a '
                                     'mock in a pure SpringBootTest setup \n'
                                     '    // while maintaining a live connection, we rely on the '
                                     'infrastructure providing the correct setup.\n'
                                     '    // For a robust integration test, we inject the actual '
                                     'services required by MessageStorageImpl.\n'
                                     '\n'
                                     '    @Autowired\n'
                                     '    private MessageStorageImpl messageStorageImpl;\n'
                                     '\n'
                                     '    @Autowired\n'
                                     '    private MessageStorageMaintenanceService maintenanceService;\n'
                                     '\n'
                                     '    // Mock service for controlled testing of partition logic\n'
                                     '    @MockBean\n'
                                     '    private MessageStorageMaintenanceService '
                                     'mockMaintenanceService;\n'
                                     '\n'
                                     '    @Test\n'
                                     '    void storeUnique_ShouldStoreNewMessageSuccessfully() throws '
                                     'Exception {\n'
                                     '        // ARRANGE\n'
                                     '        Message newMessage = new '
                                     'Message(UUID.randomUUID().toString(), "subscriber-1", "msg-1", '
                                     'OffsetDateTime.now(ZoneOffset.UTC)); \n'
                                     "        // Ensure the message ID doesn't exist in the mocked DB "
                                     'structure initially\n'
                                     '        when(mockMaintenanceService.getPartitionStart(any(), '
                                     'any()))\n'
                                     '            .thenReturn(LocalDate.now());\n'
                                     '\n'
                                     '        // ACT\n'
                                     '        messageStorageImpl.storeUnique(newMessage);\n'
                                     '\n'
                                     '        // ASSERT\n'
                                     '        // In a true integration test, we would query the DB here '
                                     'to confirm existence.\n'
                                     '        // Since we are testing the impl logic, we assert no '
                                     'exceptions occurred.\n'
                                     '        assertTrue(true); // Placeholder for actual DB '
                                     'verification\n'
                                     '    }\n'
                                     '\n'
                                     '    @Test\n'
                                     '    void '
                                     'storeUnique_ShouldThrowException_WhenMessageAlreadyProcessed() {\n'
                                     '        // ARRANGE\n'
                                     '        Message existingMessage = new '
                                     'Message(UUID.randomUUID().toString(), "subscriber-1", '
                                     '"msg-duplicate", OffsetDateTime.now(ZoneOffset.UTC)); \n'
                                     '        when(mockMaintenanceService.getPartitionStart(any(), '
                                     'any()))\n'
                                     '            .thenReturn(LocalDate.now());\n'
                                     '\n'
                                     '        // Simulate the DB returning 0 rows affected (indicating '
                                     'a duplicate/ignored insert)\n'
                                     '        // In a real scenario, we might mock the Jooq execution '
                                     'or use a transaction rollback hook.\n'
                                     '\n'
                                     '        // ACT & ASSERT\n'
                                     '        assertThrows(MessageAlreadyProcessedException.class, () '
                                     '-> {\n'
                                     '            messageStorageImpl.storeUnique(existingMessage);\n'
                                     '        });\n'
                                     '    }\n'
                                     '\n'
                                     '    @Test\n'
                                     '    void storeUnique_ShouldTriggerPartitionCreation_WhenMissing() '
                                     'throws Exception {\n'
                                     '        // ARRANGE\n'
                                     '        Message newMessage = new '
                                     'Message(UUID.randomUUID().toString(), "subscriber-1", "msg-new", '
                                     'OffsetDateTime.now(ZoneOffset.UTC)); \n'
                                     '\n'
                                     '        // Mock the scenario where the partition does not exist\n'
                                     '        // We simulate the failure that leads to partition '
                                     'creation\n'
                                     '        when(mockMaintenanceService.getPartitionStart(any(), '
                                     'any())).thenThrow(new RuntimeException("no partition of '
                                     'relation")); \n'
                                     '        \n'
                                     '        // Mock the partition creation success\n'
                                     '        '
                                     'when(mockMaintenanceService.getPartitionStart(eq(Current.CURRENT), '
                                     'eq(OffsetDateTime.now(ZoneOffset.UTC))))\n'
                                     '            .thenReturn(LocalDate.now()); \n'
                                     '\n'
                                     '        // ACT\n'
                                     '        // Assuming createPartitionOnStore is enabled by default '
                                     'or configured for this test\n'
                                     '        messageStorageImpl.storeUnique(newMessage);\n'
                                     '        \n'
                                     '        // ASSERT\n'
                                     '        // Verification that the maintenance service was called '
                                     'to add the partition.\n'
                                     '        verify(mockMaintenanceService, '
                                     'atLeastOnce()).addPartition(any(), any()); \n'
                                     '    }\n'
                                     '\n'
                                     '    // NOTE: Further tests would cover partition cleanup, time '
                                     'zone handling, etc.\n'
                                     '}\n',
                          'target_file': 'java/idempotent-consumer-jdbc/src/test/java/io/github/m4gshm/idempotent/consumer/MessageStorageImplIntegrationTest.java'},
                         fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)


if __name__ == '__main__':
    unittest.main()
