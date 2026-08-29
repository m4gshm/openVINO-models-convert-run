import unittest
from importlib.resources import files
from pathlib import Path

from agent.client.user_context import UserContext, UserContextFiles
from agent.client.veai.tool_call_fixer import fix_edit_file
from agent.parser import gemma4_test

TEST_RESOURCES = "test_resources"


class EditFileTestCase(unittest.TestCase):

    def test_add_old_file(self):
        src_file = files(__package__).joinpath(TEST_RESOURCES,
                                               "gemma4/build.gradle.kts")
        src_file_bytes = src_file.read_bytes()
        user_context = UserContext()
        user_context.files = UserContextFiles({Path("build.gradle.kts"): src_file_bytes})
        parser = gemma4_test.parser
        state = parser.new_state()
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES,
                                                     "gemma4/edit_file_no_old_text.txt")
        tool_call_text = tool_call_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        fixed = fix_edit_file(calls[0], context=user_context)

        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': 'plugins {\n'
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
                                                 '    '
                                                 'implementation("org.jooq:jooq-postgres-extensions")\n'
                                                 '    // Testcontainers dependencies for integration '
                                                 'tests\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:postgresql")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:junit-jupiter")',
                                     'old_text': 'plugins {\n'
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
                                                 '    '
                                                 'implementation("org.jooq:jooq-postgres-extensions")'}],
                          'target_file': 'build.gradle.kts'}, fixed.arguments)


if __name__ == '__main__':
    unittest.main()
