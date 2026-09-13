import unittest
from importlib.resources import files

from agent.client.user_context import UserContext, OS
from agent.client.veai.tool_call_fixer import fix_edit_file, fix_file_structure, fix_search_file_by_name
from agent.parser import ParsedFunctionCall
from agent.parser.lfm2 import Lfm2Parser

TEST_RESOURCES = "test_resources"

user_context = UserContext()
parser = Lfm2Parser()
state = parser.new_state()


class Lfm2TestCases(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_list_dir(self):
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "lfm2/list_dir.txt")
        tool_call_text = tool_call_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        self.assertEqual("list_dir", first.name)
        self.assertEqual({'depth': 0, 'directory_path': 'C:\\1\\2\\3\\4'},
                         first.arguments)
        self.assertFalse(partial)

    def test_edit_file(self):
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "lfm2/edit_file.txt")
        tool_call_text = tool_call_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        first_fixed = fix_edit_file(first)
        self.assertEqual("edit_file", first_fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': 'dependencies {\n'
                                                 '    '
                                                 'implementation("org.springframework.boot:spring-boot-starter-data-r2dbc")\n'
                                                 '    '
                                                 'implementation("io.projectsspecifiers:runtime:org.postgresql:42.7.2")\n'
                                                 '    '
                                                 'implementation("org.testcontainers:junit-jupiter:2.19.0")\n'
                                                 '    '
                                                 'testRuntime("org.testcontainers:junit-jupiter:2.19.0")\n'
                                                 '    '
                                                 'testcontainers-dependency:testcontainers:junit-jupiter:2.19.0\n'
                                                 '}',
                                     'old_text': 'dependencies {\n    // Existing dependencies...\n}'}],
                          'target_file': 'C:/alex/github/m4gshm/distributed-transactions-practice/build.gradle.kts'},
                         first_fixed.arguments)
        self.assertFalse(partial)

    def test_edit_file_2(self):
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "lfm2/edit_file_2.txt")
        tool_call_text = tool_call_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        first_fixed = fix_edit_file(first)
        self.assertEqual("edit_file", first_fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': 'dependencies {\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:junit-jupiter:2.2.0")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:postgresql:latest")\n'
                                                 '}',
                                     'old_text': ''}],
                          'target_file': 'build.gradle.kts'},
                         first_fixed.arguments)
        self.assertFalse(partial)

    # def test_edit_file_3(self):
    #     tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "lfm2/edit_file_3.txt")
    #     tool_call_text = tool_call_file.read_text()
    #     calls, partial = parser.parse_tool_calls(state, tool_call_text)
    #     first = calls[0]
    #     first_fixed = fix_edit_file(first)
    #     self.assertEqual("edit_file", first_fixed.name)
    #     self.assertEqual({'allow_multiple_matches': False,
    #                       'edits': [{'new_text': 'dependencies {\n'
    #                                              '    '
    #                                              'testImplementation("org.testcontainers:junit-jupiter:2.2.0")\n'
    #                                              '    '
    #                                              'testImplementation("org.testcontainers:postgresql:latest")\n'
    #                                              '}',
    #                                  'old_text': ''}],
    #                       'target_file': 'build.gradle.kts'},
    #                      first_fixed.arguments)
    #     self.assertFalse(partial)

    def test_search_file_by_name(self):
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "lfm2/search_file_by_name.txt")
        tool_call_text = tool_call_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        user_context = UserContext()
        user_context.os_type = OS.Windows
        fixed = [fix_search_file_by_name(c, context=user_context) for c in calls]
        self.assertEqual("search_file_by_name", fixed[0].name)
        self.assertEqual([ParsedFunctionCall(name='search_file_by_name',
                                             arguments={'glob_pattern': '**/MessageStorageImpl*Test*.java',
                                                        'search_directory': 'C:\\alex\\github\\m4gshm\\distributed-transactions-practice'},
                                             anonymous_arguments=[]),
                          ParsedFunctionCall(name='search_file_by_name',
                                             arguments={'glob_pattern': '**/MessageStorageImpl*Test*.java',
                                                        'search_directory': 'C:\\alex\\github\\m4gshm\\distributed-transactions-practice'},
                                             anonymous_arguments=[]),
                          ParsedFunctionCall(name='search_file_by_name', arguments={
                              'search_directory': 'C:/alex/github/m4gshm/distributed-transactions-practice',
                              'glob_pattern': '**/MessageStorageImpl*Test*.java'}, anonymous_arguments=[]),
                          ParsedFunctionCall(name='search_file_by_name',
                                             arguments={'glob_pattern': '**/MessageStorageImpl*Test*.java',
                                                        'search_directory': 'C:\\alex\\github\\m4gshm\\distributed-transactions-practice'},
                                             anonymous_arguments=[]),
                          ParsedFunctionCall(name='search_file_by_name', arguments={
                              'search_directory': 'C:\\alex\\github\\m4gshm\\distributed-transactions-practice',
                              'glob_pattern': '**/MessageStorageImpl*Test*.java'}, anonymous_arguments=[])],
                         fixed)
        self.assertFalse(partial)

    def test_read_file_tuple(self):
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "lfm2/read_file_tuple.txt")
        tool_call_text = tool_call_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        self.assertEqual(len(calls), 2)
        first = calls[0]
        self.assertEqual("read_file", first.name)
        self.assertEqual({'target_file': 'C:/1/2/3/4/build.gradle.kts'},
                         first.arguments)

        second = calls[1]
        self.assertEqual("read_file", second.name)
        self.assertEqual({'target_file': 'C:/1/2/3/4/java/MessageImpl.java'},
                         second.arguments)
        self.assertFalse(partial)

    def test_file_structure(self):
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "lfm2/file_structure.txt")
        tool_call_text = tool_call_file.read_text()
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        first_fixed = fix_file_structure(first, user_context)
        self.assertEqual("file_structure", first_fixed.name)
        self.assertEqual({'depth': 4, 'directory_path': 'C:/src'},
                         first.arguments)


if __name__ == '__main__':
    unittest.main()
