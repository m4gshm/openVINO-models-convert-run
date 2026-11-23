"""Tool call history inspection: locating the originating call and loop detection."""
import fnmatch
import json
import logging

from agent.openai.chat_api import ROLE_TOOL, ROLE_ASSISTANT
from agent.common.markdown import markdown_bold, markdown_json, markdown_back_tick, markdown_file_content
from agent.multimodal.content_parts import content_text
from agent.openai.chat_completions_api import ChatCompletionMessageParam, Function
from agent.openai.middleware_checkpoint import is_middleware_checkpoint

log = logging.getLogger(__name__)

# (name, arguments, result) of one completed tool call
ToolCall = tuple[str, str, str]


class FunctionCallResult:  # pylint: disable=too-few-public-methods
    """A tool call that was repeated too many times with the same result."""

    def __init__(self, name: str, arguments: str, result: str, repeats: int,
                 series_names: list[str] | None = None):
        self.name = name
        self.arguments = arguments
        self.result = result
        self.repeats = repeats
        self.series_names = series_names

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

        series = ""
        if self.series_names and len(self.series_names) > 1:
            series = "\n\nRepeated sequence: " + " -> ".join(markdown_back_tick(n) for n in self.series_names)
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
            f"Result: {function_results}" + series


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
                             repeats: int, series_names: list[str] | None = None) -> FunctionCallResult:
    """Build the loop detection report of a repeated tool call."""
    return FunctionCallResult(name=function_name, arguments=last_tool_call_arguments, result=result,
                              repeats=repeats, series_names=series_names)


def _is_error_result(result: str) -> bool:
    """A failed tool result: valid JSON with a top-level ``error`` key (the opencode tool executor shape).

    A plain substring check would misread a successful grep/read that merely contains the word "error".
    shortcut: other clients (MCP-based veai) serialize errors differently, extend here when samples exist
    """
    try:
        parsed = json.loads(result)
    except json.decoder.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and "error" in parsed


def is_tool_call_excluded(name: str, exclusions: list[str]) -> bool:
    """True when the tool name matches one of the loop-detection exclusion globs.

    Exclusions name repeatable tools (clocks, pollers) by glob patterns — by tool name only, which
    covers built-in and MCP tools alike: their names arrive in one namespace, MCP ones prefixed
    (``mcp__server__tool``).
    """
    return any(fnmatch.fnmatch(name, pattern) for pattern in exclusions)


class PreprocessToolCall:
    """Detects tool calls the model repeats with an unchanged result or after it already succeeded."""

    max_repeated_tool_calls_with_the_same_result = 3

    def __init__(self, excluded_tools: list[str] | None = None):
        """`excluded_tools` are glob patterns of tool names never checked for loops."""
        self.excluded_tools = excluded_tools or []

    def check_loop_tool_calls(self, messages: list[ChatCompletionMessageParam]) -> tuple[
        FunctionCallResult | None, int]:
        """Return the trailing series of identical consecutive tool calls, if it repeats enough.

        A series is one tool repeated in a row, or several tools repeated in the same order
        (``read -> edit -> read -> edit ...``); every repeat must return the identical result,
        so a justified repeat that yields a new value (random, weather) never counts. Tools whose
        name matches `excluded_tools` (repeatable ones like clocks or pollers) are left out of the
        series entirely — they neither count nor mask loops of the remaining tools.
        """
        calls = self._tool_calls_with_results(messages, self.excluded_tools)
        threshold = self.max_repeated_tool_calls_with_the_same_result
        best_block, best_repeats = self._longest_series(calls, threshold)
        if not best_block:
            return None, 0
        (name, arguments, result), series_names = self._report_call(best_block)
        return new_function_call_result(name, arguments, result, best_repeats, series_names), best_repeats

    @staticmethod
    def _longest_series(calls: list[ToolCall], threshold: int) -> tuple[list[ToolCall], int]:
        """Find the block of consecutive tool calls repeated the most, if at least `threshold` times.

        The block may sit anywhere in the history, not only at its end: a series of failed repeats
        is often followed by the model finally succeeding with another tool.
        """
        best_block: list[ToolCall] = []
        best_repeats = 0
        for period in range(1, len(calls) // threshold + 1):
            start = 0
            while start + period * threshold <= len(calls):
                block = calls[start:start + period]
                repeats = 1
                while start + (repeats + 1) * period <= len(calls) \
                        and calls[start + repeats * period: start + (repeats + 1) * period] == block:
                    repeats += 1
                if repeats >= threshold and repeats > best_repeats:
                    best_block, best_repeats = block, repeats
                start += period
        return best_block, best_repeats

    @staticmethod
    def _report_call(block: list[ToolCall]) -> tuple[ToolCall, list[str]]:
        """Pick the call to report: the most frequent one, ties broken by the earliest first occurrence."""
        counts: dict[ToolCall, int] = {}
        for call in block:
            counts[call] = counts.get(call, 0) + 1
        call = max(counts, key=lambda c: (counts[c], -block.index(c)))
        series_names = [n for n, _a, _r in dict.fromkeys(block)]
        return call, series_names

    @staticmethod
    def _tool_calls_with_results(messages: list[ChatCompletionMessageParam],
                                 excluded_tools: list[str]) -> list[ToolCall]:
        """Flatten the history into ``(name, arguments, result)`` triples of completed tool calls."""
        results = {m.tool_call_id: f"{content_text(m.content)}"
                   for m in messages
                   if m.role == ROLE_TOOL and not is_middleware_checkpoint(m) and m.tool_call_id}
        ordered: list[ToolCall] = []
        for message in messages:
            if message.role == ROLE_ASSISTANT:
                for tool_call in message.tool_calls or []:
                    if is_tool_call_excluded(tool_call.function.name, excluded_tools):
                        continue
                    result = results.get(tool_call.id)
                    if result:
                        ordered.append((tool_call.function.name, tool_call.function.arguments, result))
        return ordered

    def check_already_applied_tool_calls(self, messages: list[ChatCompletionMessageParam]) -> FunctionCallResult | None:
        """Return the latest tool call whose identical earlier call succeeded but now fails at least twice.

        The mechanism is client-neutral: same arguments, an earlier success, then repeated failures —
        the typical case is an ``edit`` applied once and then retried ("Could not find oldString"-style
        errors of the opencode executor). Whether a result counts as a failure is decided by
        ``_is_error_result``, which knows only the opencode error shape; clients with a different shape
        never match, so the detector stays silent for them instead of guessing. One failed retry may be
        the model checking the state, two in a row are a loop. Tools excluded by name are never checked.
        """
        calls: dict[str, tuple[str, str]] = {}
        succeeded: dict[tuple[str, str], bool] = {}
        failed_retries: dict[tuple[str, str], int] = {}
        latest: FunctionCallResult | None = None
        for message in messages:
            if message.role == ROLE_ASSISTANT:
                for tool_call in message.tool_calls or []:
                    if is_tool_call_excluded(tool_call.function.name, self.excluded_tools):
                        continue
                    calls[tool_call.id] = (tool_call.function.name, tool_call.function.arguments)
            elif message.role == ROLE_TOOL and not is_middleware_checkpoint(message):
                key = calls.get(message.tool_call_id or "")
                if not key:
                    continue
                result = f"{content_text(message.content)}"
                if not _is_error_result(result):
                    succeeded[key] = True
                elif succeeded.get(key):
                    failed_retries[key] = failed_retries.get(key, 0) + 1
                    if failed_retries[key] >= 2:
                        latest = new_function_call_result(key[0], key[1], result, failed_retries[key])
        return latest
