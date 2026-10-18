"""Best-effort json parsing of damaged LLM tool call arguments."""
import json
import logging
from typing import Any

import json_repair

log = logging.getLogger(__name__)

ARRAY_START = "["
ARRAY_END = "]"
OBJECT_START = "{"
OBJECT_END = "}"


def try_to_parse_json_arguments(raw_json: str) -> Any:
    """Parse raw arguments as json, falling back to the json_repair heuristic."""
    arguments = try_to_parse_json(raw_json)
    if not arguments:
        try:
            arguments = json_repair.loads(raw_json)
            if not isinstance(arguments, dict):
                log.error("unexpected type of repaired args '%s', arguments='%s'", type(arguments), arguments)
        except Exception:  # pylint: disable=broad-exception-caught  # json_repair may raise anything, fall back to {}
            arguments = {}
    return arguments


def try_to_parse_json(raw_json: str) -> Any:
    """Parse raw json repairing it recursively, or return None on failure."""
    result, _ = _try_to_parse_json(raw_json)
    return result


# Recursive json repair state machine: one branch per json decoder error message.
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def _try_to_parse_json(raw_json: str, cycle_detect=0, last_inserted_pos: int | None = None,
                       last_inserted_sym: str | None = None, last_replaced_sym: str | None = None
                       ) -> tuple[Any, bool]:
    """Try json.loads and recursively repair the typical LLM json damage."""
    repeat = False
    if cycle_detect >= 1000:
        raise ValueError(f"try_to_parse is cycled on {raw_json}")
    log.debug("trying to parse as json: %s", raw_json)
    parsed_object: Any | None = None
    try:
        parsed_object = json.loads(raw_json, object_pairs_hook=error_on_duplicates)
    except json.decoder.JSONDecodeError as e:
        json_len = len(raw_json)
        pos = e.pos
        sym = raw_json[pos] if pos < json_len else None
        prev_pos = pos - 1
        prev_sym = raw_json[prev_pos] if json_len > 1 and prev_pos < json_len else None
        msg = e.msg
        if msg == "Expecting ',' delimiter":
            insert_sym = ','
            if pos == last_inserted_pos:
                if last_inserted_sym == OBJECT_END:
                    insert_sym = ARRAY_END
                elif last_inserted_sym == ARRAY_END:
                    insert_sym = OBJECT_END
            elif pos == json_len:
                # end of object of array
                if prev_sym == OBJECT_END:
                    insert_sym = ARRAY_END
                else:
                    insert_sym = OBJECT_END
            new_possible_json_args = raw_json[:pos] + insert_sym + raw_json[pos + 1:]
            parsed_object, repeat = _try_to_parse_json(new_possible_json_args, cycle_detect + 1, pos, insert_sym,
                                                       sym)
            if repeat is True and sym == "\\":
                # try to escape
                prefix = raw_json[:prev_pos] + "\\" + prev_sym
                new_possible_json_args = prefix + raw_json[prev_pos + 1:]
                parsed_object, repeat = _try_to_parse_json(new_possible_json_args, cycle_detect + 1)
        elif msg == "Illegal trailing comma before end of array":
            new_possible_json_args = raw_json[:pos] + raw_json[pos + 1:]
            parsed_object, repeat = _try_to_parse_json(new_possible_json_args, cycle_detect + 1)
        elif msg == "Expecting value":
            if sym is None or sym == "]" or sym == "}" and prev_sym == ",":
                new_possible_json_args = raw_json[:prev_pos] + raw_json[prev_pos + 1:]
                parsed_object, repeat = _try_to_parse_json(new_possible_json_args, cycle_detect + 1)
            else:
                parsed_object = None
        elif msg == "Expecting property name enclosed in double quotes":
            if last_inserted_sym == "," and last_replaced_sym == "\\":
                return {}, True
        elif msg == "Invalid control character at":
            insert_sym = escape(sym)
            new_possible_json_args = raw_json[:pos] + insert_sym + raw_json[pos + 1:]
            parsed_object, repeat = _try_to_parse_json(new_possible_json_args, cycle_detect + 1, pos, insert_sym, sym)
        else:
            pass

        if not parsed_object:
            try:
                parsed_object = json_repair.loads(raw_json)
            except Exception as repair_error:  # pylint: disable=broad-exception-caught  # json_repair may raise anything
                log.error("repair json error: json=%r, error='%s'", raw_json, repair_error)
                parsed_object = None

    return parsed_object, repeat


def error_on_duplicates(ordered_pairs):
    """Dict object_pairs_hook that logs and drops duplicated json keys."""
    result = {}
    for key, value in ordered_pairs:
        if key in result:
            log.warning("duplicate key detected: key=%r, value=%r", key, value)
        else:
            result[key] = value
    return result


def escape(val: str | Any):
    """Escape a raw control character literal into its json-safe form."""
    if isinstance(val, str):
        if val == "\n":
            return "\\n"
        if val == "\t":
            return "\\t"
        if val == "\r":
            return "\\r"
    return val
