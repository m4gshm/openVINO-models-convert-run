"""Tests of the tool call loop detection and of its markdown rendering."""
import json
import unittest
from importlib.resources import files

from pydantic import TypeAdapter

from agent.openai.chat_completions_api import ChatCompletionMessageParam
from agent.preprocess.tool_call import PreprocessToolCall, new_function_call_result, \
    find_tool_call_function

TEST_RESOURCES = "test_resources"


class PreprocessToolCallCase(unittest.TestCase):
    """Checks the loop detection over a chat history of tool calls."""

    adapter = TypeAdapter(list[ChatCompletionMessageParam])

    def test_check_loop_calls(self):
        """Check check loop calls."""
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "loop_tool_calls.json")
        tool_call_text = tool_call_file.read_text(encoding="utf-8")
        messages = self.adapter.validate_json(tool_call_text)
        preprocess_tool_call = PreprocessToolCall()
        loop_call, _count = preprocess_tool_call.check_loop_tool_calls(messages)
        self.assertIsNotNone(loop_call)
        self.assertEqual("search_file_by_name", loop_call.name)
        self.assertEqual('{"glob_pattern": "InputMessages.*", "search_directory": "test_dir"}', loop_call.arguments)
        self.assertEqual('{"result":"warning","warning":"Directory does not exist: test_dir"}', loop_call.result)

    def test_check_loop_multitools_calls(self):
        """Check check loop multitools calls."""
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "loop_tool_calls_multiple.json")
        tool_call_text = tool_call_file.read_text(encoding="utf-8")
        messages = self.adapter.validate_json(tool_call_text)
        preprocess_tool_call = PreprocessToolCall()
        loop_call, _count = preprocess_tool_call.check_loop_tool_calls(messages)
        self.assertIsNotNone(loop_call)
        self.assertEqual("search_file_by_name", loop_call.name)
        self.assertEqual('{"glob_pattern": "*.java", "search_directory": "/tmp"}', loop_call.arguments)
        self.assertEqual('{"result":"success with json content","content":{"results":[]}}', loop_call.result)

    def test_markdown_render(self):
        """Check markdown render."""
        tool_call_file_src = files(__package__).joinpath(TEST_RESOURCES, "tool_call_for_markdown_rendering.json")
        tool_call_file_expected = files(__package__).joinpath(TEST_RESOURCES, "tool_call_for_markdown_rendering.md")
        tool_call_json = tool_call_file_src.read_text(encoding="utf-8")
        tool_call_md = tool_call_file_expected.read_text(encoding="utf-8")
        messages = self.adapter.validate_json(tool_call_json)

        tool_call_result = messages[-1]
        function, _i = find_tool_call_function(tool_call_result.tool_call_id, len(messages) - 1, messages)
        markdown = new_function_call_result(function.name, function.arguments, tool_call_result.content,
                                            5).render_markdown()
        self.assertEqual(tool_call_md, markdown)

    def test_already_applied_detected(self):
        """An edit that succeeded and is then repeated with the same arguments is reported."""
        messages = self.adapter.validate_json(json.dumps([
            {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "edit", "arguments": '{"oldString": "x", "newString": "y"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "Edited a.txt (1 replacement)"},
            {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function",
                "function": {"name": "edit", "arguments": '{"oldString": "x", "newString": "y"}'}}]},
            {"role": "tool", "tool_call_id": "c2",
             "content": '{"error":{"type":"tool.execution","message":"Could not find oldString in a.txt"}}'},
            {"role": "assistant", "tool_calls": [{"id": "c3", "type": "function",
                "function": {"name": "edit", "arguments": '{"oldString": "x", "newString": "y"}'}}]},
            {"role": "tool", "tool_call_id": "c3",
             "content": '{"error":{"type":"tool.execution","message":"Could not find oldString in a.txt"}}'},
        ]))
        applied = PreprocessToolCall().check_already_applied_tool_calls(messages)
        self.assertIsNotNone(applied)
        self.assertEqual("edit", applied.name)
        self.assertEqual('{"oldString": "x", "newString": "y"}', applied.arguments)
        self.assertIn("Could not find oldString", applied.result)
        self.assertEqual(2, applied.repeats)

    def test_already_applied_not_reported_on_single_failed_retry(self):
        """A single failed retry after a success is not a loop yet."""
        messages = self.adapter.validate_json(json.dumps([
            {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "edit", "arguments": '{"oldString": "x", "newString": "y"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "Edited a.txt (1 replacement)"},
            {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function",
                "function": {"name": "edit", "arguments": '{"oldString": "x", "newString": "y"}'}}]},
            {"role": "tool", "tool_call_id": "c2",
             "content": '{"error":{"type":"tool.execution","message":"Could not find oldString"}}'},
        ]))
        self.assertIsNone(PreprocessToolCall().check_already_applied_tool_calls(messages))

    def test_series_of_two_tools_in_same_order_detected(self):
        """A read->edit pair repeated in the same order with identical results is a loop series."""
        def call(cid, name, args):
            return {"role": "assistant", "tool_calls": [{"id": cid, "type": "function",
                    "function": {"name": name, "arguments": args}}]}
        def result(cid, content):
            return {"role": "tool", "tool_call_id": cid, "content": content}
        history = []
        for i in range(3):
            history += [call(f"r{i}", "read", '{"path": "a.txt"}'), result(f"r{i}", "same content"),
                        call(f"e{i}", "edit", '{"oldString": "x"}'),
                        result(f"e{i}", '{"error":{"message":"not found"}}')]
        messages = self.adapter.validate_json(json.dumps(history))
        looped, count = PreprocessToolCall().check_loop_tool_calls(messages)
        self.assertIsNotNone(looped)
        self.assertEqual(3, count)
        self.assertEqual(["read", "edit"], looped.series_names)

    def test_excluded_tool_never_counts_as_a_loop(self):
        """A tool whose name matches an exclusion glob is left out of the series entirely."""
        tool_call_file = files(__package__).joinpath(TEST_RESOURCES, "loop_tool_calls.json")
        messages = self.adapter.validate_json(tool_call_file.read_text(encoding="utf-8"))
        self.assertIsNone(PreprocessToolCall(excluded_tools=["search_*"]).check_loop_tool_calls(messages)[0])
        looped, _count = PreprocessToolCall(excluded_tools=["time*", "mcp__*clock*"]) \
            .check_loop_tool_calls(messages)
        self.assertIsNotNone(looped, "unrelated exclusions keep the detection working")

    def test_excluded_tool_does_not_mask_the_rest_of_a_series(self):
        """Filtering a repeatable tool out of a read->poll series still exposes the read loop."""
        def call(cid, name, args):
            return {"role": "assistant", "tool_calls": [{"id": cid, "type": "function",
                    "function": {"name": name, "arguments": args}}]}
        def result(cid, content):
            return {"role": "tool", "tool_call_id": cid, "content": content}
        history = []
        for i in range(3):
            history += [call(f"r{i}", "read", '{"path": "a.txt"}'), result(f"r{i}", "same content"),
                        call(f"p{i}", "poll_status", '{"job": "1"}'), result(f"p{i}", "pending")]
        messages = self.adapter.validate_json(json.dumps(history))
        looped, count = PreprocessToolCall(excluded_tools=["poll*"]).check_loop_tool_calls(messages)
        self.assertIsNotNone(looped)
        self.assertEqual("read", looped.name)
        self.assertEqual(3, count)
        self.assertEqual(["read"], looped.series_names)

    def test_already_applied_ignores_excluded_tool(self):
        """The already-applied detector skips tools matched by the exclusions."""
        history = [
            {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "edit", "arguments": '{"oldString": "x", "newString": "y"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "Edited a.txt (1 replacement)"},
            {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function",
                "function": {"name": "edit", "arguments": '{"oldString": "x", "newString": "y"}'}}]},
            {"role": "tool", "tool_call_id": "c2",
             "content": '{"error":{"type":"tool.execution","message":"Could not find oldString"}}'},
            {"role": "assistant", "tool_calls": [{"id": "c3", "type": "function",
                "function": {"name": "edit", "arguments": '{"oldString": "x", "newString": "y"}'}}]},
            {"role": "tool", "tool_call_id": "c3",
             "content": '{"error":{"type":"tool.execution","message":"Could not find oldString"}}'},
        ]
        messages = self.adapter.validate_json(json.dumps(history))
        self.assertIsNone(PreprocessToolCall(excluded_tools=["edit"]).check_already_applied_tool_calls(messages))
        applied = PreprocessToolCall(excluded_tools=["time*"]).check_already_applied_tool_calls(messages)
        self.assertIsNotNone(applied, "unrelated exclusions keep the detection working")

    def test_already_applied_not_reported_without_earlier_success(self):
        """Failing calls that never succeeded are not 'already applied'."""
        messages = self.adapter.validate_json(json.dumps([
            {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "edit", "arguments": '{"oldString": "x", "newString": "y"}'}}]},
            {"role": "tool", "tool_call_id": "c1",
             "content": '{"error":{"type":"tool.execution","message":"Could not find oldString"}}'},
        ]))
        self.assertIsNone(PreprocessToolCall().check_already_applied_tool_calls(messages))

    def test_already_applied_not_reported_when_error_word_in_plain_result(self):
        """A successful result merely containing the word "error" is not treated as a failure."""
        messages = self.adapter.validate_json(json.dumps([
            {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "grep", "arguments": '{"pattern": "error"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": 'file.py: raise ValueError("error")'},
            {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function",
                "function": {"name": "grep", "arguments": '{"pattern": "error"}'}}]},
            {"role": "tool", "tool_call_id": "c2", "content": 'file.py: raise ValueError("error")'},
        ]))
        self.assertIsNone(PreprocessToolCall().check_already_applied_tool_calls(messages))


if __name__ == '__main__':
    unittest.main()
