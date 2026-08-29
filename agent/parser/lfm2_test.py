import json
import unittest
from importlib.resources import files

from agent.client.veai.tool_call_fixer import fix_edit_file
from agent.inference.token_handler import TokenProcessor, TokenHandlerConfig
from agent.parser.lfm2 import Lfm2Parser

TEST_RESOURCES = "test_resources"

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

    def test_list_dir_probably_tool_call_parsing(self):
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "lfm2/list_dir.json")
        tool_call_json = tool_call_file.read_text(encoding="utf-8")
        tokens = json.loads(tool_call_json)

        processor = TokenProcessor(prompt="", parser=parser, init_chat_events=True, config=TokenHandlerConfig(),
                                   is_veai=True)
        process_tokens, _ = processor.process_tokens(tokens)

        first = process_tokens[0]
        first_tool_calls = first.choices[0].delta.tool_calls[0]
        function = first_tool_calls.function
        arguments = json.loads(function.arguments)
        self.assertEqual("list_dir", function.name)
        self.assertEqual({'depth': 2, 'directory_path': '.'}, arguments)

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


if __name__ == '__main__':
    unittest.main()
