"""Repair of model-generated tool calls: alternate argument names for OpenCode IDE tools.

Small local models often mix argument schemas of other toolsets they were trained on
(for example Claude's ``file_path``/``old_string``); only the names are normalized here.
Missing required arguments are left as-is on purpose: OpenCode answers with a clear
validation error the model can fix, while invented placeholders would hide the problem.
"""
import logging

from agent.parser import ParsedFunctionCall

log = logging.getLogger(__name__)

# canonical OpenCode argument name -> alternate names some models use instead
_ALIASES: dict[str, tuple[str, ...]] = {
    "path": ("file_path", "filePath"),
    "oldString": ("old_string", "old_text", "oldText"),
    "newString": ("new_string", "new_text", "newText"),
    "pattern": ("glob", "glob_pattern"),
    "sessionID": ("session_id", "sessionId"),
    "agent": ("agent_type", "agentType"),
    "prompt": ("message", "text"),
}


def opencode_fix_tool_calls(function: ParsedFunctionCall, _user_context=None) -> ParsedFunctionCall:
    """Normalize alternate argument names to the canonical OpenCode ones."""
    args = function.arguments
    if not isinstance(args, dict):
        log.warning("unexpected arguments type for tool '%s': %r", function.name, args)
        return function
    for canonical, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in args:
                value = args.pop(alias)
                if canonical not in args:
                    args[canonical] = value
    return function
