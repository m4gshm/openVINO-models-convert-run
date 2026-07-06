import json
import logging

from agent.common.roles import ROLE_TOOL, ROLE_ASSISTANT
from agent.inference.token_handler import markdown_bold, markdown_json, markdown_back_tick
from agent.openai.chat_completions_api import ChatCompletionMessageParam

log = logging.getLogger(__name__)


class FunctionCallResult:
    def __init__(self, name: str, arguments: str, result: str):
        self.name = name
        self.arguments = arguments
        self.result = result

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

        function_results = "\n" + markdown_json(self.result) if self.result else "NO_RESULT"
        return markdown_bold("WARNING: Tool calls loop.") + "\n\n" \
                                                            f"Tool name: {function_name}\n\n" \
                                                            f"Arguments: {function_arguments}\n\n" \
                                                            f"Result: {function_results}"


def find_tool_call_arguments(tool_call_id: str | None, start_from: int,
                             messages: list[ChatCompletionMessageParam]) -> tuple[str | None, int]:
    i = start_from
    while i >= 0:
        prev_message = messages[i]
        if prev_message and prev_message.role == ROLE_ASSISTANT:
            tool_calls = prev_message.tool_calls
            for tool_call in (tool_calls or []):
                if tool_call_id and tool_call.id == tool_call_id:
                    return tool_call.function.arguments, i
        i -= 1
    return None, i


def new_function_call_result(last_message: ChatCompletionMessageParam,
                             last_tool_call_arguments: str | None) -> FunctionCallResult:
    return FunctionCallResult(name=last_message.name, arguments=last_tool_call_arguments, result=last_message.content)


class PreprocessToolCall:
    max_repeated_tool_calls_with_the_same_result = 5
    max_messages_to_check = 5 * 4

    def check_loop_calls(self, messages: list[ChatCompletionMessageParam]) -> tuple[FunctionCallResult | None, int]:
        last_message = messages[-1] if messages else None
        if last_message and last_message.role == ROLE_TOOL and last_message.name:
            last_tool_call_id = last_message.tool_call_id
            last_tool_call_arguments, i = find_tool_call_arguments(last_tool_call_id, len(messages) - 2, messages)

            repeated = 1
            # reverse loop from prelast message
            i -= 1
            count = 0
            while i >= 0:
                count += 1
                if count >= self.max_messages_to_check:
                    break
                message = messages[i]
                if message.role != ROLE_TOOL:
                    continue
                name = message.name
                result = message.content
                tool_call_id = message.tool_call_id
                if not name or not result:
                    continue

                tool_call_arguments, i = find_tool_call_arguments(tool_call_id, i - 1, messages)

                is_equal = name == last_message.name and tool_call_arguments == last_tool_call_arguments and result == last_message.content
                if is_equal:
                    repeated += 1
                else:
                    break

                if repeated >= self.max_repeated_tool_calls_with_the_same_result:
                    return new_function_call_result(last_message, last_tool_call_arguments), repeated
                i -= 1
        return None, 0
