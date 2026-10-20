"""Tool call history inspection: locating the originating call and loop detection."""
import json
import logging

from agent.openai.chat_api import ROLE_TOOL, ROLE_ASSISTANT
from agent.common.markdown import markdown_bold, markdown_json, markdown_back_tick, markdown_file_content
from agent.multimodal.content_parts import content_text
from agent.openai.chat_completions_api import ChatCompletionMessageParam, Function
from agent.openai.middleware_checkpoint import is_middleware_checkpoint

log = logging.getLogger(__name__)

# name -> arguments -> result -> positions of the repeating calls in the history
FunctionCallIndex = dict[str, dict[str, dict[str, set[int]]]]


class FunctionCallResult:  # pylint: disable=too-few-public-methods
    """A tool call that was repeated too many times with the same result."""

    def __init__(self, name: str, arguments: str, result: str, repeats: int):
        self.name = name
        self.arguments = arguments
        self.result = result
        self.repeats = repeats

    def render_markdown(self) -> str:
        """Render the loop warning as markdown."""
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
            log.debug("function arguments parsing error: %s", e)
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
                            messages: list[ChatCompletionMessageParam]) -> tuple[Function, int] | tuple[None, int]:
    """Find the assistant tool call a tool result belongs to, searching backwards."""
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
    """Build the loop detection report of a repeated tool call."""
    return FunctionCallResult(name=function_name, arguments=last_tool_call_arguments, result=result, repeats=repeats)


class PreprocessToolCall:  # pylint: disable=too-few-public-methods
    """Detects tool calls the model repeats with an unchanged result."""

    max_repeated_tool_calls_with_the_same_result = 5
    max_messages_to_check = 5 * 4

    def check_loop_tool_calls(self, messages: list[ChatCompletionMessageParam]) -> tuple[
        FunctionCallResult | None, int]:
        """Return the first looping tool call of the history, if any."""
        if not messages:
            return None, 0
        results: dict[str, str | None] = {}
        function_call = FunctionCallIndex()
        over_all_messages = False
        for i, message in enumerate(reversed(messages)):
            if not over_all_messages and i >= self.max_messages_to_check - 1:
                break
            if message.role == ROLE_TOOL and not is_middleware_checkpoint(message):
                result_tool_call_id = message.tool_call_id
                if result_tool_call_id:
                    results[result_tool_call_id] = f"{content_text(message.content)}"
            elif message.role == ROLE_ASSISTANT:
                loop_result, many_repeats = self._register_tool_calls(
                    message, i, results, function_call, self.max_repeated_tool_calls_with_the_same_result)
                repeats = loop_result[3] if loop_result else 0
                if not over_all_messages and (many_repeats or repeats > 2):
                    over_all_messages = True
                if loop_result and repeats >= self.max_repeated_tool_calls_with_the_same_result:
                    name, arguments, result = loop_result[:3]
                    return new_function_call_result(name, arguments, result, repeats), repeats
            else:
                break

        return None, 0

    @staticmethod
    def _register_tool_calls(message: ChatCompletionMessageParam, position: int, results: dict[str, str | None],
                             function_call: FunctionCallIndex, max_repeats: int) -> tuple[
        tuple[str, str, str, int] | None, bool]:
        """Index the calls of one assistant message.

        Returns the most repeated call as ``(name, arguments, result, repeats)``
        and whether any call of the message already repeats more than twice.
        """
        best: tuple[str, str, str, int] | None = None
        best_repeats = 0
        many_repeats = False
        for tool_call in message.tool_calls or []:
            function = tool_call.function
            result = results.get(tool_call.id)
            if not result:
                log.warning("cannot find function call for tool last_tool_call_id=%s, tool call result=%s",
                            tool_call.id, message)
                continue
            function_arguments = function_call.setdefault(function.name, dict[str, dict[str, set[int]]]())
            function_call_positions = function_arguments \
                .setdefault(function.arguments, dict[str, set[int]]()) \
                .setdefault(result, set())
            function_call_positions.add(position)
            repeats = len(function_call_positions)
            many_repeats = many_repeats or repeats > 2
            if repeats > best_repeats:
                best = (function.name, function.arguments, result, repeats)
                best_repeats = repeats
            if repeats >= max_repeats:
                break
        return best, many_repeats
