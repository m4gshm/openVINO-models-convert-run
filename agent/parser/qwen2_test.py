import unittest
from importlib.resources import files

from agent.client.veai.tool_call_fixer import fix_edit_file
from agent.parser.qwen2 import Qwen2Parser

TEST_RESOURCES = "test_resources"

parser = Qwen2Parser()
state = parser.new_state()


class Qwen2TestCases(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_edit_file(self):
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "qwen2/veai_edit_file.txt")
        tool_call_text = tool_cal_file.read_text(encoding="utf-8")
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]

        fixed = fix_edit_file(first, None)

        expected_edits = [{'new_text': 'implementation("io.projectreactor:reactor-core")\n'
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
                                       '    // TestContainers dependencies for integration '
                                       'tests\n'
                                       '    '
                                       'developmentImplementation("org.testcontainers:testcontainers")\n'
                                       '    '
                                       'developmentImplementation("org.testcontainers:junit-jupiter")\n'
                                       '    '
                                       'developmentImplementation("org.testcontainers:postgresql")',
                           'old_text': 'implementation("io.projectreactor:reactor-core")\n'
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
                                       'implementation("org.jooq:jooq-postgres-extensions")'}]
        self.assertEqual("edit_file", first.name)
        self.assertEqual({'edits': expected_edits, 'target_file': 'C:/build.gradle.kts'}, first.arguments)

        self.assertEqual("edit_file", fixed.name)
        self.assertEqual({'allow_multiple_matches': True,
                          'edits': expected_edits, 'target_file': 'C:/build.gradle.kts'},
                         fixed.arguments)

    def test_parse_multiple_calls(self):
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "qwen2/mutliple_tool_calls.txt")
        tool_call_text = tool_cal_file.read_text(encoding="utf-8")
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        self.assertEqual(3, len(calls))
        self.assertTrue(partial)
        self.assertEqual("list_dir", calls[0].name)
        self.assertEqual({'directory_path': 'C:/'}, calls[0].arguments)

        self.assertEqual("search_file_by_name", calls[1].name)
        self.assertEqual({'glob_pattern': '**/*Test*',
                          'is_case_sensitive': True,
                          'search_directory': '.'}, calls[1].arguments)

        self.assertEqual("write_file", calls[2].name)
        self.assertEqual({'allow_overwrite': False, 'content': 'text', 'target_file': 'C:/test.txt'},
                         calls[2].arguments)


if __name__ == '__main__':
    unittest.main()
