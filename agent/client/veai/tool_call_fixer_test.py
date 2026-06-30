import unittest

from agent.parser import ParsedFunctionCall


class MyTestCase(unittest.TestCase):
    def test_to_openai_tool_call(self):
        openai_tool_call = ParsedFunctionCall(name="foo", arguments={"arg": "val"}).to_openai_function_call()
        self.assertEqual("foo", openai_tool_call.name)
        self.assertEqual("""{\"arg\": \"val\"}""", openai_tool_call.arguments)


if __name__ == '__main__':
    unittest.main()
