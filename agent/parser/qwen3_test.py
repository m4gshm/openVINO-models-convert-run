import unittest

from common.openai_model import ToolCall, FunctionDefinition
from parser.qwen3 import parse_tool_calls, EXPECTED_PARAMETERS_PROPERTIES, EXPECTED_PROPERTY_TYPE


class TestAddFunction(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_function_no_close_tag = """
            <tool_call>
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
            </function>
            """
        self.first_function = self.first_function_no_close_tag + "</tool_call>"
        self.second_function = """
            <tool_call>
            <function=ls>
            <parameter=directory>
            /tmp
            </parameter>
            </function>
            </tool_call>
            """
        self.function_with_invalid_json_parameter = """
            <tool_call>
            <function=select>
            <parameter=options>
            [1,2,"3"],
            </parameter>
            </function>
            </tool_call>
            """

    def test_without_close_tag(self):
        calls = parse_tool_calls(self.first_function_no_close_tag)
        self.assertEqual(len(calls), 1)
        function_call: ToolCall = calls[0]
        self.assertEqual("function", function_call.type)
        self.assertEqual("read_file", function_call.function.name)
        self.assertEqual("""{"end_line": "75", "start_line": "19", "target_file": "Target.py"}""",
                         function_call.function.arguments)

    def test_two_functions(self):
        calls = parse_tool_calls(self.first_function + self.second_function)
        self.assertEqual(len(calls), 2)

        first: ToolCall = calls[0]
        self.assertEqual("function", first.type)
        self.assertEqual("read_file", first.function.name)
        self.assertEqual("""{"end_line": "75", "start_line": "19", "target_file": "Target.py"}""",
                         first.function.arguments)

        second: ToolCall = calls[1]
        self.assertEqual("function", second.type)
        self.assertEqual("ls", second.function.name)
        self.assertEqual("""{"directory": "/tmp"}""", second.function.arguments)

    def test_two_functions_where_first_without_close_tag(self):
        calls = parse_tool_calls(self.first_function_no_close_tag + self.second_function)
        self.assertEqual(len(calls), 2)

        first: ToolCall = calls[0]
        self.assertEqual("function", first.type)
        self.assertEqual("read_file", first.function.name)
        self.assertEqual("""{"end_line": "75", "start_line": "19", "target_file": "Target.py"}""",
                         first.function.arguments)

        second: ToolCall = calls[1]
        self.assertEqual("function", second.type)
        self.assertEqual("ls", second.function.name)
        self.assertEqual("""{"directory": "/tmp"}""", second.function.arguments)

    def test_functions_with_invalid_json_parameter(self):
        function_name = "select"
        calls = parse_tool_calls(self.function_with_invalid_json_parameter, {
            function_name: FunctionDefinition(name=function_name, parameters={
                EXPECTED_PARAMETERS_PROPERTIES: {"options": {EXPECTED_PROPERTY_TYPE: "array"}}
            })
        })

        first: ToolCall = calls[0]
        self.assertEqual("function", first.type)
        self.assertEqual("select", first.function.name)
        self.assertEqual("""{"options": [1, 2, \"3\"]}""", first.function.arguments)


if __name__ == '__main__':
    unittest.main()
