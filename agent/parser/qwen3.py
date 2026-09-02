import logging
import re
from typing import Any, Iterable

from agent.parser import ParserState, StateEvent, ParsedFunctionCall
from agent.parser import fill_state_by_prompt_tail
from agent.parser.gemma4 import try_to_parse_json
from agent.parser.qwen_base import CLOSE_TAG_PREF, OPEN_TAG_SUF, TOOL_CALL_START, TOOL_CALL_END, QwenBaseParser

log = logging.getLogger(__name__)

PARAMETER_START_PREF = "<parameter"
PARAMETER_END = CLOSE_TAG_PREF + "parameter>"

EXPECTED_PROPERTY_TYPE = 'type'
EXPECTED_PARAMETERS_PROPERTIES = "properties"
FUNCTION_START_PREF = "<function"
FUNCTION_START = FUNCTION_START_PREF + OPEN_TAG_SUF
FUNCTION_END = CLOSE_TAG_PREF + "function>"


def parse_name(parameters_block) -> tuple[str | None, str | None]:
    pattern = f"=(.*?){OPEN_TAG_SUF}(.*)"
    match = re.search(pattern, parameters_block, re.DOTALL)
    if match:
        name = match.group(1).strip()
        tail = match.group(2).strip()
        return name, tail
    else:
        return None, None


def get_arguments(expected_parameters: dict[str, Any], arguments_block: str) -> tuple[
    dict[str, Any], bool]:
    arguments: dict[str, Any] = {}
    partial = False
    if arguments_block.startswith("parameter="):
        # try to parse as json
        arguments_block = arguments_block[len("parameter="):]
        if arguments_block.endswith(PARAMETER_END):
            arguments_block = arguments_block[:len(arguments_block) - len(PARAMETER_END)]
        json_args, _ = try_to_parse_json(arguments_block)
        partial = False
        arguments = json_args
    else:
        parameter_blocks = arguments_block.split(PARAMETER_START_PREF)
        for parameter_block in parameter_blocks:
            parameter_block = parameter_block.lstrip()
            if len(parameter_block) == 0:
                continue
            param_pattern = f"=(.*?){OPEN_TAG_SUF}(.*)"
            parameters = re.findall(param_pattern, parameter_block, re.DOTALL)
            partial = False
            for param_name, param_tail in parameters:
                param_name_norm: str = param_name.strip()
                param_tail_norm: str = param_tail.strip()

                parameter_end_i = param_tail_norm.find(PARAMETER_END)
                full_parameter = param_tail_norm[:parameter_end_i] if parameter_end_i >= 0 else None
                if not full_parameter is None:
                    next_param_i = parameter_end_i + len(PARAMETER_END)
                    next_tail = param_tail_norm[next_param_i:]
                    param_tail_norm = next_tail
                    param_value_norm = full_parameter.strip()
                else:
                    partial = True
                    param_value_norm = param_tail_norm.strip()

                expected_param = expected_parameters.get(param_name_norm) if expected_parameters else None
                expected_type = expected_param['type'] if expected_param and 'type' in expected_param else None
                is_expected_array = is_expected_type('array', expected_type)
                is_expected_object = is_expected_type('object', expected_type)
                is_like_json_array = param_value_norm.startswith("[")
                is_like_json_object = param_value_norm.startswith("{")
                if (is_expected_array and is_like_json_array) or (is_expected_object and is_like_json_object):
                    result_parameter, _ = try_to_parse_json(param_value_norm)
                    arguments[param_name_norm] = result_parameter
                else:
                    arguments[param_name_norm] = param_value_norm

    return arguments, partial


def is_expected_type(exp_type: str, possible_types: Any) -> bool | Any:
    if isinstance(possible_types, Iterable):
        return exp_type in possible_types
    return exp_type == possible_types


def parse_function_call(state: ParserState, function_block: str, partial: bool) -> tuple[ParsedFunctionCall, bool]:
    function_block_rstrip = function_block.rstrip()
    if function_block_rstrip.endswith(FUNCTION_END):
        function_block = function_block_rstrip[:-len(FUNCTION_END)]

    func_name, tail = parse_name(function_block)
    if not func_name is None:
        expected_parameters = state.get_function_parameters(func_name)
        log.debug(f"found expected_parameters: function={func_name}, expected_parameters={expected_parameters}")
        expected_properties_dict = expected_parameters if isinstance(expected_parameters, dict) else None
        arguments, partial_param = get_arguments(expected_properties_dict, tail or "")
        if partial_param:
            partial = True

        parsed_function_call = ParsedFunctionCall(name=func_name, arguments=arguments)
    else:
        parsed_function_call = None
    return parsed_function_call, partial


class Qwen3MoeParser(QwenBaseParser):
    def new_state(self, prompt: str = "", supported_functions: dict[str, dict] | None = None,
                  init_chat_events=True) -> ParserState:
        if not prompt:
            state = super().new_state(prompt, supported_functions, init_chat_events)
            if init_chat_events:
                state.start_event(StateEvent.THINK)
        else:
            state = self._new_state(supported_functions)
            fill_state_by_prompt_tail(init_chat_events, prompt, state, self.is_assistant)
        return state

    def parse_tool_calls(self, state: ParserState, tool_call_expression: str) -> tuple[list[ParsedFunctionCall], bool]:
        tool_call_expression = tool_call_expression.lstrip()
        tool_call_blocks = tool_call_expression.split(TOOL_CALL_START)

        parsed_calls: list[ParsedFunctionCall] = []
        partial = False
        for call_block in tool_call_blocks:
            if len(call_block) == 0:
                continue

            call_block_rstrip = call_block.rstrip()
            if call_block_rstrip.endswith(TOOL_CALL_END):
                call_block = call_block_rstrip[:-len(TOOL_CALL_END)]

            call_block = call_block.lstrip()
            function_blocks = call_block.split(FUNCTION_START_PREF)
            for function_block in function_blocks:
                if len(function_block) == 0:
                    continue
                parsed_function_call, partial = parse_function_call(state, function_block, partial)

                if parsed_function_call:
                    parsed_calls.append(parsed_function_call)
        return parsed_calls, partial
