"""Tests of the lfm2 tool call parser."""
import unittest

from agent.client.user_context import UserContext, OS
from agent.client.veai.tool_call_fixer import fix_edit_file, fix_file_structure, fix_search_file_by_name, \
    fix_ask_user_with_options
from agent.parser import ParsedFunctionCall
from agent.parser.lfm2 import Lfm2Parser, parse_delayed_phrase
from agent.parser import test_support
from agent.parser.test_support import parse_first_call, read_tool_call

TEST_RESOURCES = "test_resources"


def new_windows_user_context() -> UserContext:
    """Build a Windows user context for the argument fixers."""
    context = UserContext()
    context.os_type = OS.Windows
    return context


user_context = new_windows_user_context()
parser = Lfm2Parser(test_support.FakeTokenizer())
state = parser.new_state()


class Lfm2TestCases(unittest.TestCase):
    """Checks the tool call expressions of the lfm2 test resources."""

    def __init__(self, *args, **kwargs):
        """Prepare the fixtures of the test case."""
        super().__init__(*args, **kwargs)

    def test_list_dir(self):
        """Check list dir."""
        first, partial = parse_first_call(parser, state, __package__, "lfm2/list_dir.txt")
        self.assertEqual("list_dir", first.name)
        self.assertEqual({'depth': 0, 'directory_path': 'C:\\1\\2\\3\\4'},
                         first.arguments)
        self.assertFalse(partial)

    def test_edit_file(self):
        """Check edit file."""
        first, partial = parse_first_call(parser, state, __package__, "lfm2/edit_file.txt")
        first_fixed = fix_edit_file(first)
        self.assertEqual("edit_file", first_fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': 'dependencies {\n'
                                                 '    '
                                                 'implementation("org.springframework.boot:'
                                                 'spring-boot-starter-data-r2dbc")\n'
                                                 '    '
                                                 'implementation("io.projectsspecifiers:runtime:'
                                                 'org.postgresql:42.7.2")\n'
                                                 '    '
                                                 'implementation("org.testcontainers:junit-jupiter:2.19.0")\n'
                                                 '    '
                                                 'testRuntime("org.testcontainers:junit-jupiter:2.19.0")\n'
                                                 '    '
                                                 'testcontainers-dependency:testcontainers:'
                                                 'junit-jupiter:2.19.0\n'
                                                 '}',
                                     'old_text': 'dependencies {\n    // Existing dependencies...\n}'}],
                          'target_file': 'C:/alex/github/m4gshm/distributed-transactions-practice/build.gradle.kts'},
                         first_fixed.arguments)
        self.assertFalse(partial)

    def test_edit_file_2(self):
        """Check edit file 2."""
        first, partial = parse_first_call(parser, state, __package__, "lfm2/edit_file_2.txt")
        first_fixed = fix_edit_file(first)
        self.assertEqual("edit_file", first_fixed.name)
        self.assertEqual({'allow_multiple_matches': False,
                          'edits': [{'new_text': 'dependencies {\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:'
                                                 'junit-jupiter:2.2.0")\n'
                                                 '    '
                                                 'testImplementation("org.testcontainers:postgresql:latest")\n'
                                                 '}',
                                     'old_text': ''}],
                          'target_file': 'build.gradle.kts'},
                         first_fixed.arguments)
        self.assertFalse(partial)

    # def test_edit_file_3(self):
    #     first, partial = parse_first_call(parser, state, __package__, "lfm2/edit_file_3.txt")
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
        """Check search file by name."""
        calls, partial = parser.parse_tool_calls(state, read_tool_call(__package__, "lfm2/search_file_by_name.txt"))
        windows_context = UserContext()
        windows_context.os_type = OS.Windows
        fixed = [fix_search_file_by_name(c, context=windows_context) for c in calls]
        self.assertEqual("search_file_by_name", fixed[0].name)
        self.assertEqual([ParsedFunctionCall(name='search_file_by_name',
                                             arguments={'glob_pattern': '**/MessageStorageImpl*Test*.java',
                                                        'search_directory': 'C:\\alex\\github\\m4gshm\\'
                                                                           'distributed-transactions-practice'},
                                             anonymous_arguments=[]),
                          ParsedFunctionCall(name='search_file_by_name',
                                             arguments={'glob_pattern': '**/MessageStorageImpl*Test*.java',
                                                        'search_directory': 'C:\\alex\\github\\m4gshm\\'
                                                                           'distributed-transactions-practice'},
                                             anonymous_arguments=[]),
                          ParsedFunctionCall(name='search_file_by_name', arguments={
                              'search_directory': 'C:/alex/github/m4gshm/distributed-transactions-practice',
                              'glob_pattern': '**/MessageStorageImpl*Test*.java'}, anonymous_arguments=[]),
                          ParsedFunctionCall(name='search_file_by_name',
                                             arguments={'glob_pattern': '**/MessageStorageImpl*Test*.java',
                                                        'search_directory': 'C:\\alex\\github\\m4gshm\\'
                                                                           'distributed-transactions-practice'},
                                             anonymous_arguments=[]),
                          ParsedFunctionCall(name='search_file_by_name', arguments={
                              'search_directory': 'C:\\alex\\github\\m4gshm\\distributed-transactions-practice',
                              'glob_pattern': '**/MessageStorageImpl*Test*.java'}, anonymous_arguments=[])],
                         fixed)
        self.assertFalse(partial)

    def test_read_file_tuple(self):
        """Check read file tuple."""
        calls, partial = parser.parse_tool_calls(state, read_tool_call(__package__, "lfm2/read_file_tuple.txt"))
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
        """Check file structure."""
        first, _partial = parse_first_call(parser, state, __package__, "lfm2/file_structure.txt")
        first_fixed = fix_file_structure(first, user_context)
        self.assertEqual("file_structure", first_fixed.name)
        self.assertEqual({'depth': 4, 'directory_path': 'C:/src'},
                         first.arguments)

    def test_ask_user_with_options_call_json(self):
        """Check ask user with options call json."""
        tool_call_text = read_tool_call(__package__, "lfm2/ask_user_with_options_call.json")
        delay = parse_delayed_phrase(tool_call_text)
        calls = delay.tool_calls
        first = calls[0]
        first_fixed = fix_ask_user_with_options(first, user_context)
        self.assertEqual("ask_user_with_options", first_fixed.name)
        self.assertEqual({
            "question": "Based on the MessageStorageImpl class, what specific test cases would you "
                        "like to create? Please specify the key scenarios to cover.",
            'options': ["Basic unit tests for insert and duplicate key handling",
                        "Integration tests with real PostgreSQL database using testcontainers",
                        "Tests for partition creation and maintenance service interactions",
                        "Tests for the storeUnique method with various failure scenarios",
                        "All of the above"],
            "is_multiple_choice": True
        }, first.arguments)

    def test_delay_streaming_requires_chat_mode(self):
        """Without a chat conversation event a '{' token must not start the delayed streaming."""
        completion_state = parser.new_state("def f():\n    return", init_chat_events=False)
        self.assertIsNone(completion_state.get_current_event())
        self.assertFalse(parser.is_delay_streaming_start(completion_state, "{"))

    def test_delay_streaming_applies_in_chat_mode(self):
        """In a chat conversation a '{' token starts the delayed streaming as before."""
        chat_state = parser.new_state()
        self.assertTrue(parser.is_delay_streaming_start(chat_state, "{"))
        self.assertTrue(parser.is_delay_streaming_start(chat_state, " [ "))


if __name__ == '__main__':
    unittest.main()
