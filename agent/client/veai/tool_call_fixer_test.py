import unittest

from agent.inference.token_handler import to_openai_tool_call
from agent.parser import ParsedFunctionCall


class MyTestCase(unittest.TestCase):
    def test_to_openai_tool_call(self):
        openai_tool_call = to_openai_tool_call(ParsedFunctionCall(name="foo", arguments={"arg": "val"}))
        self.assertEqual("foo", openai_tool_call.function.name)
        self.assertEqual("""{\"arg\": \"val\"}""", openai_tool_call.function.arguments)


if __name__ == '__main__':
    unittest.main()
