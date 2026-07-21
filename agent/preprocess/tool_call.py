import json
import logging

from agent.common.roles import ROLE_TOOL, ROLE_ASSISTANT
from agent.inference.token_handler import markdown_bold, markdown_json, markdown_back_tick, markdown_file_content
from agent.openai.chat_completions_api import ChatCompletionMessageParam, FunctionCall

log = logging.getLogger(__name__)


class FunctionCallResult:
    def __init__(self, name: str, arguments: str, result: str, repeats: int):
        self.name = name
        self.arguments = arguments
        self.result = result
        self.repeats = repeats

    def render_markdown(self) -> str:
        function_name = markdown_back_tick(self.name)
        function_arguments = ""
        try:
            arguments = json.loads(self.arguments)
            if not arguments:
                function_arguments = "NO_ARGS"
            else:
                for arg_name, arg_value in arguments.items():
                    if len(function_arguments) > 0:
                        function_arguments += "\n"
                    function_arguments += f"* {arg_name}: " + markdown_back_tick(arg_value)
            function_arguments = "\n" + function_arguments
        except json.decoder.JSONDecodeError as e:
            log.debug(f"function arguments parsing error: {e}")
            function_arguments = "\n" + markdown_json(self.arguments)

        result = self.result
        result_strip = result.strip()
        is_looks_like_json = result_strip.startswith("{") or result_strip.startswith("[")
        function_results = "\n" + markdown_file_content(result,
                                                        "json" if is_looks_like_json else "") if result else "NO_RESULT"
        return markdown_bold(
            f"WARNING: Tool calls loop "
            f"({self.repeats} repeat{"s" if self.repeats != 1 else ""})."
        ) + "\n\n" \
            f"Tool name: {function_name}\n\n" \
            f"Arguments: {function_arguments}\n\n" \
            f"Result: {function_results}"


def find_tool_call_function(tool_call_id: str | None, start_from: int,
                            messages: list[ChatCompletionMessageParam]) -> tuple[FunctionCall, int] | tuple[None, int]:
    i = start_from
    while i >= 0:
        prev_message = messages[i]
        if prev_message and prev_message.role == ROLE_ASSISTANT:
            tool_calls = prev_message.tool_calls
            for tool_call in (tool_calls or []):
                if tool_call_id and tool_call.id == tool_call_id:
                    return tool_call.function, i
        i -= 1
    return None, i


def new_function_call_result(function_name: str, last_tool_call_arguments: str, result: str,
                             repeats: int) -> FunctionCallResult:
    return FunctionCallResult(name=function_name, arguments=last_tool_call_arguments, result=result, repeats=repeats)


class PreprocessToolCall:
    max_repeated_tool_calls_with_the_same_result = 5
    max_messages_to_check = 5 * 4

    def check_loop_tool_calls(self, messages: list[ChatCompletionMessageParam]) -> tuple[
        FunctionCallResult | None, int]:
        if not messages:
            return None, 0
        results = dict[str, str | None]()
        function_call = dict[str, dict[str, dict[str, set[int]]]]()
        over_all_messages = False
        for i, message in enumerate(reversed(messages)):
            if not over_all_messages and i >= self.max_messages_to_check - 1:
                break
            if message.role == ROLE_TOOL:
                result_tool_call_id = message.tool_call_id
                result = message.content
                if result_tool_call_id:
                    results[result_tool_call_id] = f"{result}"
            elif message.role == ROLE_ASSISTANT:
                for tool_call in message.tool_calls or []:
                    function_tool_call_id = tool_call.id
                    function = tool_call.function
                    function_arguments = function_call.setdefault(function.name, dict[str, dict[str, set[int]]]())
                    function_results = function_arguments.setdefault(function.arguments, dict[str, set[int]]())
                    result = results.get(function_tool_call_id)
                    if result:
                        function_call_positions = function_results.setdefault(result, set())
                        function_call_positions.add(i)

                        repeats = len(function_call_positions)
                        if not over_all_messages and repeats > 2:
                            over_all_messages = True
                        if repeats >= self.max_repeated_tool_calls_with_the_same_result:
                            return new_function_call_result(function.name,
                                                            function.arguments, result, repeats), repeats
                    else:
                        log.warning(
                            f"cannot find function call for tool last_tool_call_id={function_tool_call_id},"
                            f" tool call result={message}")

            else:
                break

        return None, 0
