import itertools
import json
import logging
import re
import time
from typing import List, Any

import json_repair

from agent.common.openai_model import ToolCall, FunctionCall, FunctionDefinition

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
REASONING_START = "<think>"
REASONING_END = "</think>"

CONVERSATION_START = "<|im_start|>"
CONVERSATION_END = "<|im_end|>"

log = logging.getLogger(__name__)
tool_call_counter = itertools.count(start=0)


def think_is_over(subword: str) -> bool:
    return subword.strip() == REASONING_END


def think_is_started(subword: str) -> bool:
    return subword.strip() == REASONING_START


def is_conversation_start(text: str) -> bool:
    return CONVERSATION_START == text.strip()


def is_conversation_end(text: str) -> bool:
    return CONVERSATION_END == text.strip()


def is_possible_tool_call_start(text: str) -> bool:
    return TOOL_CALL_START == text.strip()


def is_possible_tool_call_end(text: str) -> bool:
    return TOOL_CALL_END == text.strip()


def is_prompt_start_thinking(prompt: str) -> bool:
    return prompt.endswith(REASONING_START, 0, len(prompt) - 1) if prompt.endswith(
        '\n') else prompt.endswith(REASONING_START)


def parse_tool_calls(text: str, supported_functions: dict[str, FunctionDefinition]) -> List[ToolCall]:
    """Parses Qwen3-Coder style unique XML tool call blocks."""
    tool_call_blocks = text.split(TOOL_CALL_START)
    ts = int(time.time())

    parsed_calls: list[ToolCall] = []
    for i, call_block in enumerate(tool_call_blocks):
        func_name = get_func_name(call_block)
        if func_name is None:
            # log
            continue

        function = supported_functions.get(func_name)
        if function is None:
            log.warning(f"parsed unsupported function '{func_name}'")

        parameters = function.parameters if function is not None else []
        arguments = get_arguments(call_block, parameters)

        call_id = next(tool_call_counter)
        parsed_calls.append(ToolCall(id=f"call_{ts}_{call_id}_{func_name}",
                                     function=FunctionCall(name=func_name, arguments=arguments)))
    return parsed_calls


def get_arguments(call_block: str, expected_parameters: dict[str, Any] | None = None) -> str:
    expected_properties: dict[str, Any] = expected_parameters.get("properties", {}) if expected_parameters else {}

    param_pattern = r"<parameter=(.*?)>(.*?)</parameter>"
    parameters = re.findall(param_pattern, call_block, re.DOTALL)

    arguments: dict[str, str | dict[str, Any] | list[Any]] = {}
    for param_name, param_value in parameters:
        param_name_norm: str = param_name.strip()
        param_value_norm: str = param_value.strip()
        expected_property: dict[str, Any] = expected_properties.get(param_name_norm, {})
        expected_type = expected_property.get('type', 'string') if expected_property else 'string'
        if expected_type == 'array' or expected_type == 'object':
            try:
                structured_parameter: dict[str, Any] | list[Any] = json.loads(param_value_norm)
            except json.decoder.JSONDecodeError as e:
                log.error(f"function parameter parsing error, parameter '{param_name}', value '{param_value_norm}', "
                          f"expected_type '{expected_type}': {e}")
                structured_parameter: dict[str, Any] | list[Any] = json_repair.loads(param_value_norm)
                log.info(f"repaired parameter value '{structured_parameter}'")

            arguments[param_name_norm] = structured_parameter
        else:
            arguments[param_name_norm] = param_value_norm
    if isinstance(arguments, dict):
        arguments_str = json.dumps(arguments, ensure_ascii=False)
    else:
        arguments_str = str(arguments)
    return arguments_str


def get_func_name(call_block) -> str | None:
    func_match = re.search(r"<function=(.*?)>", call_block)
    func_name = func_match.group(1).strip() if func_match else None
    return func_name
