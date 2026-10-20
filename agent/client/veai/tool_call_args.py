"""Generic argument utilities of the Veai tool call repair.

The helpers here are shared by the per-tool fixers: they read the arguments of
a model-generated tool call, coerce the loosely typed values the model produces
and resolve the file paths of the session.
"""
import json
import logging
import re
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

from agent.client.user_context import UserContext, OS
from agent.openai.chat_api import ROLE_ASSISTANT

log = logging.getLogger(__name__)

TypeT = TypeVar("TypeT")

TARGET_FILE = "target_file"

# The fields a model may use instead of the canonical target_file field.
TARGET_FILE_ALIASES = ["file_path", "file", "path", "edit_scope", "directory_path"]


def get_args(function) -> dict[str, Any]:
    """Return the parsed function arguments as a dict, empty when missing."""
    return function.arguments or {}


def get_one_of(args: dict[str, Any], one_of_field: list[str]) -> str | None:
    """Return the first non-None value among the candidate argument fields."""
    for field in one_of_field:
        value = args.get(field)
        if not value is None:
            return value
    return None


def as_int_or_none(val, name: str) -> tuple[int | None, bool]:
    """Return (int, False) on success or (recovered int or None, True) when repaired."""
    result = as_type_or_none(int, val, name)
    if result is not None:
        return result, False
    if isinstance(val, str):
        match = re.search(r'-\d+|\d+', val)
        return int(match.group()) if match else None, True
    return None, False


def as_bool_or_none(val, name: str) -> bool | None:
    """Return the value as bool: bools pass, strings are mapped, anything else None."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() == "true"
    return as_type_or_none(bool, val, name)


def as_type_or_none(t: type[TypeT], val, name: str) -> TypeT | None:
    """Cast val to t, logging and returning None when the cast fails.

    A value that already has the requested type is returned as it is.
    """
    if val is None:
        return None
    if isinstance(val, t):
        return val
    try:
        caster = cast(Callable[[Any], TypeT], t)
        return caster(val)
    except ValueError:
        log.warning("%s is not an %s: '%s'", name, t, val)
        return None


def get_target_file(args, context: UserContext | None,
                    additional_fields: list[str] | None = None) -> tuple[str, bool]:
    """Resolve the target file from the primary or alternative argument fields."""
    target_file = args.get(TARGET_FILE)

    invalid = not target_file
    if invalid:
        possible_fields = list(TARGET_FILE_ALIASES)
        if additional_fields:
            possible_fields.extend(additional_fields)
        target_file = get_one_of(args, possible_fields)

    target_file, fixed = fix_windows_path(target_file, context)
    if fixed:
        invalid = True

    return target_file, invalid


def clean_file_path(target_file):
    """Drop stray braces and unescaped quotes that leaked into a file path."""
    if not target_file:
        return target_file

    cleaned_target_file = ""
    for i, s in enumerate(target_file):
        ignore = s in ("}", "]")
        if not ignore and s in ("\"", "'"):
            prev = target_file[i - 1] if i > 0 else None
            if prev != "\\":
                ignore = True

        if not ignore:
            cleaned_target_file += s
    log.debug("clean_file_path: path=%s, cleaned=%s", target_file, cleaned_target_file)
    return cleaned_target_file


def fix_windows_path(path: Any | None, context: UserContext | None) -> tuple[Any, bool]:
    """Normalize POSIX-style paths into Windows drive paths for Windows sessions."""
    fixed = False
    if path and isinstance(path, str) and context and context.os_type == OS.Windows:
        # SERA case
        if path.startswith("/"):
            fixed = True
            path = path[1:]

        p = Path(path)
        parts = p.parts
        if len(parts) > 0:
            first = parts[0]
            stripped = False
            while first == "\\" and len(parts) > 1:
                stripped = True
                first = parts[1]
                parts = parts[1:]

            win_disk_letter_size = 1
            if not (first.endswith(":/") or first.endswith(":\\")) and len(first) == win_disk_letter_size:
                new_first = first + (":/" if "/" in path else ":\\")
                new_parts = (new_first,) + parts[1:]
                path = str(Path().joinpath(*new_parts))
                fixed = True
            elif stripped:
                path = str(Path().joinpath(*parts))
                fixed = True
    return path, fixed


def find_target_file_from_prev_tool_call(args, context: UserContext | None, function_name: str,
                                         target_file: Any | None) -> Any | None:
    """Scan previous assistant tool calls of the same tool for a target_file."""
    if log.isEnabledFor(logging.DEBUG):
        log.debug("no required target_file, trying to get from previous cool call of '%s', args='%s'",
                  function_name, args)
    else:
        log.info("no required target_file, trying to get from previous cool call of '%s'", function_name)

    messages = context.messages if context else []
    for message in reversed(messages):
        if message.role == ROLE_ASSISTANT:
            for tool_call in message.tool_calls or []:
                function = tool_call.function
                if function.name == function_name:
                    try:
                        arguments = json.loads(function.arguments)
                    except JSONDecodeError:
                        log.debug("error on function arguments parsing: tool_call.id=%s, function.name=%s, "
                                  "arguments='%s'", tool_call.id, function.name, function.arguments)
                        arguments = {}

                    target_file = arguments.get(TARGET_FILE)
                    if target_file:
                        log.info("gets target file from previous tool call: tool_call.id=%s, "
                                 "function.name=%s, target_file='%s'", tool_call.id, function.name, target_file)
    return target_file
