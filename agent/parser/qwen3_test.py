import unittest
from importlib.resources import files

from agent.openai.chat_completions_api import FunctionDefinition
from agent.parser.qwen3 import EXPECTED_PARAMETERS_PROPERTIES, EXPECTED_PROPERTY_TYPE, Qwen3Parser

TEST_RESOURCES = "test_resources"

parser = Qwen3Parser()
state = parser.new_state()


class TestAddFunction(unittest.TestCase):

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
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "partially_generated_tool_call.txt")
        tool_call_text = tool_cal_file.read_text(encoding="utf-8")
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        self.assertEqual("write_file", first.name)
        self.assertEqual({"allow_overwrite": "false", "content": "no finished parameter"}, first.arguments)
        self.assertTrue(partial)

    def test_search_file_by_name(self):
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "qwen3/search_file_by_name.txt")
        tool_call_text = tool_cal_file.read_text(encoding="utf-8")
        calls, partial = parser.parse_tool_calls(state, tool_call_text)
        first = calls[0]
        self.assertEqual("search_file_by_name", first.name)
        self.assertEqual({'glob_pattern': 'Properties.*',
                          'search_directory': 'consumer/config'},
                         first.arguments)
        self.assertFalse(partial)


if __name__ == '__main__':
    unittest.main()
