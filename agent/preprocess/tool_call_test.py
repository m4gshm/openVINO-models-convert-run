import unittest
from importlib.resources import files

from pydantic import TypeAdapter

from agent.openai.chat_completions_api import ChatCompletionMessageParam
from agent.preprocess.tool_call import PreprocessToolCall, new_function_call_result, \
    find_tool_call_function

TEST_RESOURCES = "test_resources"


class PreprocessToolCallCase(unittest.TestCase):
    adapter = TypeAdapter(list[ChatCompletionMessageParam])

    def test_check_loop_calls(self):
        tool_cal_file = files(__package__).joinpath(TEST_RESOURCES, "loop_tool_calls.json")
        tool_call_text = tool_cal_file.read_text(encoding="utf-8")
        messages = self.adapter.validate_json(tool_call_text)
        preprocess_tool_call = PreprocessToolCall()
        loop_call, count = preprocess_tool_call.check_loop_tool_calls(messages)
        self.assertIsNotNone(loop_call)
        self.assertEqual("search_file_by_name", loop_call.name)
        self.assertEqual('{"glob_pattern": "InputMessages.*", "search_directory": "test_dir"}', loop_call.arguments)
        self.assertEqual('{"result":"warning","warning":"Directory does not exist: test_dir"}', loop_call.result)

    def test_markdown_render(self):
        tool_cal_file_src = files(__package__).joinpath(TEST_RESOURCES, "tool_call_for_markdown_rendering.json")
        tool_cal_file_expected = files(__package__).joinpath(TEST_RESOURCES, "tool_call_for_markdown_rendering.md")
        tool_call_json = tool_cal_file_src.read_text(encoding="utf-8")
        tool_call_md = tool_cal_file_expected.read_text(encoding="utf-8")
        messages = self.adapter.validate_json(tool_call_json)

        tool_call_result = messages[-1]
        function, i = find_tool_call_function(tool_call_result.tool_call_id, len(messages) - 1, messages)
        markdown = new_function_call_result(function.name, function.arguments, tool_call_result.content).render_markdown()
        self.assertEqual(tool_call_md, markdown)


if __name__ == '__main__':
    unittest.main()
