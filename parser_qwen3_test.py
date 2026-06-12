import unittest

from common_openapi_model import ToolCall
from parser_qwen3 import parse_tool_calls


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

    def test_without_close_tag(self):
        calls = parse_tool_calls(self.first_function_no_close_tag)
        self.assertEqual(len(calls), 1)
        function_call: ToolCall = calls[0]
        self.assertEqual(function_call.type, "function")
        self.assertEqual(function_call.function.name, "read_file")
        self.assertEqual(function_call.function.arguments,
                         """{"end_line": "75", "start_line": "19", "target_file": "Target.py"}""")

    def test_two_functions(self):
        calls = parse_tool_calls(self.first_function + self.second_function)
        self.assertEqual(len(calls), 2)

        first: ToolCall = calls[0]
        self.assertEqual(first.type, "function")
        self.assertEqual(first.function.name, "read_file")
        self.assertEqual(first.function.arguments,
                         """{"end_line": "75", "start_line": "19", "target_file": "Target.py"}""")

        second: ToolCall = calls[1]
        self.assertEqual(second.type, "function")
        self.assertEqual(second.function.name, "ls")
        self.assertEqual(second.function.arguments,
                         """{"directory": "/tmp"}""")

    def test_two_functions_where_first_without_close_tag(self):
        calls = parse_tool_calls(self.first_function_no_close_tag + self.second_function)
        self.assertEqual(len(calls), 2)

        first: ToolCall = calls[0]
        self.assertEqual(first.type, "function")
        self.assertEqual(first.function.name, "read_file")
        self.assertEqual(first.function.arguments,
                         """{"end_line": "75", "start_line": "19", "target_file": "Target.py"}""")

        second: ToolCall = calls[1]
        self.assertEqual(second.type, "function")
        self.assertEqual(second.function.name, "ls")
        self.assertEqual(second.function.arguments,
                         """{"directory": "/tmp"}""")


if __name__ == '__main__':
    unittest.main()
