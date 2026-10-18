"""Tests of the gemma4 tool call parser."""
import unittest
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

from agent.client.user_context import UserContext, OS, UserContextFiles
from agent.client.veai.tool_call_fixer import fix_edit_file, fix_write_file, GEMMA_4, fix_search_file_by_name
from agent.parser.gemma4 import Gemma4ChannelParser
from agent.parser.test_support import (
    GRADLE_BUILD_KTS_CONTENT,
    GRADLE_BUILD_KTS_CONTENT_INTEGRATION,
    GRADLE_DEPENDENCIES_AND_TESTCONTAINERS,
    parse_first_call,
)

TEST_RESOURCES = "test_resources"


def new_windows_user_context() -> UserContext:
    """Build a Windows user context holding the gradle build file of the gemma4 resources."""
    context = UserContext(model_architectures={GEMMA_4})
    context.os_type = OS.Windows

    file_content = {Path("java/build.gradle.kts"): read_bytes("build.gradle.kts.txt")}
    context.files = UserContextFiles(file_content=file_content)
    return context


def open_file(file_name: str) -> Traversable:
    """Open a test resource of the gemma4 parser tests."""
    return files(__package__).joinpath(TEST_RESOURCES, "gemma4/" + file_name)


def read_text(file_name: str):
    """Read a gemma4 test resource as text."""
    file = open_file(file_name)
    return file.read_text()


def read_bytes(file_name: str):
    """Read a gemma4 test resource as bytes."""
    file = open_file(file_name)
    return file.read_bytes()


USER_CONTEXT = new_windows_user_context()

parser = Gemma4ChannelParser()


class TestAddFunction(unittest.TestCase):
    """Checks the tool call expressions of the gemma4 test resources."""

    def __init__(self, *args, **kwargs):
        """Prepare the fixtures of the test case."""
        super().__init__(*args, **kwargs)
        self.normal_function = (
            "'<|tool_call>call:some_function{file1:<|\"|>1.json<|\"|>,2:<|\"|>2.json<|\"|>,"
            "mode:<|\"|>strict<|\"|>,timeout:30,empty_val:}<tool_call|>'"
        )

    def test_parsing(self):
        """Check parsing."""
        state = parser.new_state()
        calls, _partial = parser.parse_tool_calls(state=state, tool_call_expression=self.normal_function)
        self.assertEqual(len(calls), 1)
        function_call = calls[0]
        self.assertEqual("some_function", function_call.name)
        self.assertEqual(
            {"file1": "1.json", "2": "2.json", "mode": "strict", "timeout": "30", "empty_val": ""},
            function_call.arguments)

    def test_wrapped_file_structure_parse(self):
        """Check wrapped file structure parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/file_structure_wrapped.txt")
        self.assertEqual("file_function", first.name)
        self.assertEqual({'file_path': 'dir/dir2/Foo.txt', 'start_at': 1}, first.arguments)
        self.assertFalse(partial)

    def test_search_for_text_space_delimited_args(self):
        """Check search for text space delimited args."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/search_for_text_space_delimited_args.txt")
        self.assertEqual("search_for_text", first.name)
        self.assertEqual({'is_case_sensitive': False,
                          'target_path_or_url': 'MessageStorageImpl.java',
                          'text_snippet': '    "MessageStorageImpl"'}, first.arguments)
        self.assertFalse(partial)

    def test_read_file_windows_path_parse(self):
        """Check read file windows path parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/read_file_windows_path.txt")
        self.assertEqual("read_file", first.name)
        self.assertEqual({"start_line": "1", "end_line": "500", "file_path": "C:/src/MessageStorageImpl.java"},
                         first.arguments)
        self.assertFalse(partial)

    def test_read_file_windows_path_delim_without_arg_name_parse(self):
        """Check read file windows path delim without arg name parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/read_file_windows_path_delim_w"
                                                                        "ithout_arg_name.txt")
        self.assertEqual("read_file", first.name)
        self.assertEqual({}, first.arguments)
        self.assertEqual(["C:/src/MessageStorageImpl.java"], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_read_file_like_json(self):
        """Check read file like json."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/read_file_like_json.txt")
        self.assertEqual("read_file", first.name)
        self.assertEqual({'end_line': '500', 'start_line': '1', 'target_file': 'build.gradle.kts"}}'}, first.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_get_configurations_parse(self):
        """Check get configurations parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/get_configurations.txt")
        fixed = fix_write_file(first)
        self.assertEqual("get_configurations", fixed.name)
        self.assertEqual({'configuration_name_for_part': 'test',
                          'include_global_configurations': True,
                          'page': '1',
                          'target_file': None}, fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_write_file_parse(self):
        """Check write file parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/write_file.txt")
        self.assertEqual("write_file", first.name)
        self.assertEqual({'allow_overwrite': True,
                          'content': GRADLE_BUILD_KTS_CONTENT,
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'},
                         first.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_write_file_like_json(self):
        """Check write file like json."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/write_file_like_json.txt")
        fixed = fix_write_file(first)
        self.assertEqual("write_file", fixed.name)
        self.assertEqual({'allow_overwrite': True,
                          'content': GRADLE_BUILD_KTS_CONTENT,
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'},
                         fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_write_file_like_json_2(self):
        """Check write file like json 2."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/write_file_like_json_2.txt")
        fixed = fix_write_file(first)
        self.assertEqual("write_file", first.name)
        self.assertEqual({'allow_overwrite': True,
                          'content': GRADLE_BUILD_KTS_CONTENT_INTEGRATION,
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'},
                         fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

class EditFileCallCases(unittest.TestCase):
    """Checks the edit_file tool call expressions of the gemma4 test resources."""

    def test_edit_file_parse(self):
        """Check edit file parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file.txt")
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
                                                 'implementation("org.springframework.boot:spring-boot-autoco'
                                                   'nfigure")\n'
                                                 '    implementation("org.jooq:jooq")\n'
                                                 '    '
                                                 'implementation("org.jooq:jooq-postgres-extensions")\n'
                                                 '}'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'}, first.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file2_parse(self):
        """Check edit file2 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file_2.txt")

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
        """Check edit file3 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file_3.txt")

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
                                                 'testImplementation("org.springframework.boot:spring-boot-st'
                                                   'arter-test")\n'
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
        """Check edit file4 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file_4.txt")

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
        """Check edit file5 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file_5.txt")

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': '}', 'old_text': '// ...'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'}, fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file6_parse(self):
        """Check edit file6 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file_6.txt")

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': True,
                          'edits': [{'new_text': 'testImplementation("org.testcontainers:junit-jupiter")',
                                     'old_text': 'implementation("org.jooq:jooq-postgres-extensions")'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'}, fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file7_parse(self):
        """Check edit file7 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file_7.txt")

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
        """Check edit file8 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file_8.txt")

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
        """Check edit file9 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file_9.txt")

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
        """Check edit file10 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file_10.txt")

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
        """Check edit file11 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file_11.txt")

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
        """Check edit file12 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file_12.txt")

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'allow_multiple_matches': False,
                                     'new_text': '    '
                                                 'testImplementation("org.springframework.boot:spring-to-test")\n'
                                                 '    '
                                                 'testImplementation("net.bnd.system.container:postgresql-jun'
                                                   'it-container") '
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

class EditFileCallCasesContinued(unittest.TestCase):
    """Checks the remaining edit_file tool call expressions of the gemma4 test resources."""

    def test_edit_file13_parse(self):
        """Check edit file13 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file_13.txt")

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
        """Check edit file14 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file_14.txt")

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': '\n'
                                                 '    '
                                                 'testImplementation("org.springframework.boot:spring-boot-st'
                                                   'arter-test)")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:postgresql") '
                                                 '// Testcontainers for PostgreSQL\n'
                                                 '}\n',
                                     'old_text': '\n}\n'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'}, fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file15_parse(self):
        """Check edit file15 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file_15.txt")

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
        """Check edit file16 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file_16.txt")

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': '    '
                                                 "testImplementation('org.springframework.boot:spring-boot-st"
                                                   "arter-test') "
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

    def test_edit_file17_parse(self):
        """Check edit file17 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file_17.txt")

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': 'dependencies {\n'
                                                 '    // ... existing dependencies ...\n'
                                                 '\n'
                                                 '    // Testcontainers for PostgreSQL\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:postgresql:1.19.7")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:junit-jupiter:1.19.7")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:docker-compose:1.19.7")\n'
                                                 '\n'
                                                 '    // ... other dependencies ...\n'
                                                 '}'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'},
                         fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file18_parse(self):
        """Check edit file18 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file_18.txt")

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': GRADLE_DEPENDENCIES_AND_TESTCONTAINERS}],
                           },
                         fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file19_parse(self):
        """Check edit file19 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file_19.txt")

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'file_path': 'java/idempotent-consumer-jdbc/build.gradle.kts',
                                     'new_text': '    api(project(":postgres-jdbc"))\n'
                                                 '\n'
                                                 '    '
                                                 'testImplementation("org.junit.jupiter:junit-jupiter-api")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:junit-jupiter")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:postgresql")\n'
                                                 '\n'
                                                 '    '
                                                 'implementation("io.projectreactor:reactor-core")\n',
                                     'old_text': '    api(project(":postgres-jdbc"))\n'
                                                 '\n'
                                                 '    '
                                                 'implementation("io.projectreactor:reactor-core")\n',
                                     'start_column': 9,
                                     'start_line': 9}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'},
                         fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file20_parse(self):
        """Check edit file20 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file_20.txt")

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': 'dependencies {\n'
                                                 '    // Existing dependencies...\n'
                                                 '\n'
                                                 '    // Test dependencies (Ensuring JUnit 5 is '
                                                 'present)\n'
                                                 '    testImplementation '
                                                 "'org.junit.jupiter:junit-jupiter-api:5.10.2'\n"
                                                 '    testRuntimeOnly '
                                                 "'org.junit.jupiter:junit-jupiter-engine:5.10.2'\n"
                                                 '\n'
                                                 '    // Testcontainers dependencies for PostgreSQL '
                                                 'integration testing\n'
                                                 '    testImplementation '
                                                 "'org.testcontainers:junit-jupiter:1.19.7'\n"
                                                 '    testImplementation '
                                                 "'org.testcontainers:postgresql:1.19.7'\n"
                                                 '    // Ensure PostgreSQL driver is available for the '
                                                 'container connection\n'
                                                 '    testImplementation '
                                                 "'org.postgresql:postgresql:42.7.3' \n"
                                                 '}'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'},
                         fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file21_parse(self):
        """Check edit file21 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file_21.txt")

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': '    '
                                                 'testImplementation("org.testcontainers:postgresql:1.19.7")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:junit-jupiter:1.19.7")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:containers:1.19.7")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:postgresql-module-j'
                                                   'ava:1.19.7")',
                                     'old_text': '    '
                                                 'implementation("org.jooq:jooq-postgres-extensions")'}],
                          'target_file': 'java/idempotent-consumer-jdbc/build.gradle.kts'},
                         fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file22_parse(self):
        """Check edit file22 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file_22.txt")

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': '    '
                                                 'testImplementation("org.springframework.boot:spring-boot-st'
                                                   'arter-test")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:postgresql")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:junit-jupiter")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:local-embedded")\n'
                                                 '    // Podman is usually handled by the '
                                                 'Testcontainers ecosystem, but explicit integration '
                                                 'might require a wrapper or specific library if not '
                                                 'using standard Docker setup.\n'
                                                 '    // For Podman, usually, you rely on '
                                                 'docker/podman-compose setup outside Gradle '
                                                 'configuration or specific extensions, but for '
                                                 'simplicity in a Gradle test setup, the standard '
                                                 'Testcontainers dependency should suffice for '
                                                 'PostgreSQL. \n'
                                                 '\n'
                                                 '    // Note: Specific Podman setup might require '
                                                 'separate configuration not covered by standard '
                                                 'dependencies. \n'
                                                 '    \n',
                                     'old_text': '    '
                                                 'implementation("org.jooq:jooq-postgres-extensions")\n'}]},
                         fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_edit_file23_parse(self):
        """Check edit file23 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/edit_file_23.txt")

        fixed = fix_edit_file(first, USER_CONTEXT)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': '// --- Testcontainers dependencies for integration '
                                                 'testing ---\n'
                                                 "testImplementation('org.testcontainers:postgresql:postgresql:1.19.1' "
                                                 ')\n'
                                                 "testImplementation('org.testcontainers:junit-jupiter:junit-"
                                                   "jupiter:1.8.2' "
                                                 ')\n'}],
                          'target_file': 'java/build.gradle.kts'},
                         fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_write_file_2_parse(self):
        """Check write file 2 parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/write_file_2.txt")

        fixed = fix_write_file(first, USER_CONTEXT)

        self.assertEqual("write_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'allow_overwrite': True,
                          'content': 'package io.github.m4gshm.idempotent.consumer;\n'
                                     '\n'
                                     'import '
                                     'io.github.m4gshm.idempotent.consumer.storage.tables.InputMessages;\n'
                                     'import '
                                     'io.github.m4gshm.idempotent.consumer.storage.tables.MessageStorageMaint'
                                       'enanceService;\n'
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
                          'target_file': 'java/idempotent-consumer-jdbc/src'
                           '/test/java/io/github/m4gshm/idempotent/consumer/MessageStorageImplIntegrationTest.java'},
                         fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)

    def test_search_file_by_name_parse(self):
        """Check search file by name parse."""
        state = parser.new_state()
        first, partial = parse_first_call(parser, state, __package__, "gemma4/search_file_by_name.txt")

        fixed = fix_search_file_by_name(first, USER_CONTEXT)

        self.assertEqual("search_file_by_name", fixed.name)
        self.assertEqual({'glob_pattern': 'build.gradle.kts',
                          'search_directory': 'C:\\idempotent-consumer-jdbc'},
                         fixed.arguments)
        self.assertEqual([], first.anonymous_arguments)
        self.assertFalse(partial)


if __name__ == '__main__':
    unittest.main()
