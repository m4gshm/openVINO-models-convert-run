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
    arguments = try_to_parse_json(raw_json)
    if not arguments:
        try:
            arguments = json_repair.loads(raw_json)
            if not isinstance(arguments, dict):
                log.error(f"unexpected type of repaired args '{type(arguments)}', arguments='{arguments}'")
        except Exception as e:
            arguments = {}
    return arguments


def try_to_parse_json(raw_json: str, cycle_detect=0, last_inserted_pos: int | None = None,
                      last_inserted_sym: str | None = None, last_replaced_sym: str | None = None
                      ) -> tuple[Any, bool]:
    repeat = False
    if cycle_detect >= 1000:
        raise Exception(f"try_to_parse is cycled on {raw_json}")
    log.debug(f"trying to parse as json: {raw_json}")
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
            parsed_object, repeat = try_to_parse_json(new_possible_json_args, cycle_detect + 1, pos, insert_sym,
                                                      sym)
            if repeat == True and sym == "\\":
                # try to escape
                prefix = raw_json[:prev_pos] + "\\" + prev_sym
                new_possible_json_args = prefix + raw_json[prev_pos + 1:]
                parsed_object, repeat = try_to_parse_json(new_possible_json_args, cycle_detect + 1)
                pass
        elif msg == "Illegal trailing comma before end of array":
            new_possible_json_args = raw_json[:pos] + raw_json[pos + 1:]
            parsed_object, repeat = try_to_parse_json(new_possible_json_args, cycle_detect + 1)
            pass
        elif msg == "Expecting value":
            if sym is None or sym == "]" or sym == "}" and prev_sym == ",":
                new_possible_json_args = raw_json[:prev_pos] + raw_json[prev_pos + 1:]
                parsed_object, repeat = try_to_parse_json(new_possible_json_args, cycle_detect + 1)
            else:
                parsed_object = None
        elif msg == "Expecting property name enclosed in double quotes":
            if last_inserted_sym == "," and last_replaced_sym == "\\":
                return {}, True
            else:
                pass
        elif msg == "Invalid control character at":
            insert_sym = escape(sym)
            new_possible_json_args = raw_json[:pos] + insert_sym + raw_json[pos + 1:]
            parsed_object, repeat = try_to_parse_json(new_possible_json_args, cycle_detect + 1, pos, insert_sym, sym)
            pass
        else:
            pass

        if not parsed_object:
            try:
                parsed_object = json_repair.loads(raw_json)
            except Exception as e:
                log.error(f"repair json error: json='{raw_json!r}', error='{e}'")
                parsed_object = None

    return parsed_object, repeat


def error_on_duplicates(ordered_pairs):
    result = {}
    for key, value in ordered_pairs:
        if key in result:
            log.warning(f"duplicate key detected: key={key!r}, value={value!r}")
        else:
            result[key] = value
    return result


def escape(val: str | Any):
    if isinstance(val, str):
        if val == "\n":
            return "\\n"
        elif val == "\t":
            return "\\t"
        elif val == "\r":
            return "\\r"
    return val
