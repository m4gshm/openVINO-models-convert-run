import json
import logging
import re
from typing import Any

import json_repair

from agent.common.roles import ROLE_ASSISTANT
from agent.parser import ParserState, StateEvent, ParsedFunctionCall
from agent.parser.qwen_base import CLOSE_TAG_PREF, OPEN_TAG_SUF, TOOL_CALL_START, TOOL_CALL_END, QwenBaseParser

log = logging.getLogger(__name__)

PARAMETER_START_PREF = "<parameter"
PARAMETER_END = CLOSE_TAG_PREF + "parameter>"

EXPECTED_PROPERTY_TYPE = 'type'
EXPECTED_PARAMETERS_PROPERTIES = "properties"
FUNCTION_START_PREF = "<function"
FUNCTION_START = FUNCTION_START_PREF + OPEN_TAG_SUF
FUNCTION_END = CLOSE_TAG_PREF + "function>"

THINK_START = "<think>"
IM_START = "<|im_start|>"


def parse_name(parameters_block) -> tuple[str | None, str | None]:
    pattern = f"=(.*?){OPEN_TAG_SUF}(.*)"
    match = re.search(pattern, parameters_block, re.DOTALL)
    if match:
        name = match.group(1).strip()
        tail = match.group(2).strip()
        return name, tail
    else:
        return None, None


def get_arguments(arguments_block: str, expected_parameters: dict[str, Any] | None = None) -> tuple[
    dict[str, Any], bool]:
    expected_properties: dict[str, Any] = expected_parameters.get(EXPECTED_PARAMETERS_PROPERTIES,
                                                                  {}) if expected_parameters else {}

    arguments: dict[str, Any] = {}

    partial = False
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

            full_parameter = param_tail_norm.endswith(PARAMETER_END)
            if full_parameter:
                param_tail_norm = param_tail_norm[:-len(PARAMETER_END)]
            else:
                partial = True

            param_value_norm = param_tail_norm.strip()

            is_like_json = param_value_norm.startswith("[") or param_value_norm.startswith("{")
            if is_like_json:
                try:
                    result_parameter = json.loads(param_value_norm)
                except json.decoder.JSONDecodeError as e:
                    log.debug(
                        f"function parameter parsing error, parameter='{param_name}', value='{param_value_norm}', "
                        f"type '{type(param_value_norm)}': {e}")
                    try:
                        result_parameter = json_repair.loads(param_value_norm)
                        log.debug(f"repaired parameter value '{result_parameter}'")
                    except Exception as e:
                        log.debug(
                            f"fail to repaired parameter value, parameter='{param_name}', value='{param_value_norm}', "
                            f"type '{type(param_value_norm)}': {e}")
                        result_parameter = param_value_norm
                arguments[param_name_norm] = result_parameter
            else:
                arguments[param_name_norm] = param_value_norm

    return arguments, partial


class Qwen3MoeParser(QwenBaseParser):
    def new_state(self, prompt: str = "", init_chat_events=True) -> ParserState:
        if not prompt:
            state = super().new_state(prompt, init_chat_events)
            if init_chat_events:
                state.start_event(StateEvent.THINK)
        else:
            prompt = prompt.rstrip()
            tail_size = 200
            tail = prompt[-tail_size:] if len(prompt) > tail_size else prompt
            tail_lines = tail.rstrip().splitlines()
            state = self._new_state()
            if init_chat_events:
                is_think = None
                is_conversation = None
                role = ""

                for i, line in enumerate(reversed(tail_lines)):
                    line = line.strip()
                    if line.endswith(THINK_START):
                        is_think = i
                        log.debug(f"state init is_think: {is_think}")
                    elif line.startswith(IM_START):
                        is_conversation = i
                        log.debug(f"state init is_conversation: {is_conversation}")
                        prompt_role = line[len(IM_START):].strip()
                        is_assistant = self.is_assistant(prompt_role)
                        if is_assistant:
                            role = ROLE_ASSISTANT
                            log.debug(f"state init role: {role}")
                            break
                prefill_i = None
                if not is_conversation is None:
                    state.start_event(StateEvent.CONVERSATION)
                    prefill_i = (len(tail_lines) - 1 - is_conversation)
                if not is_think is None:
                    state.start_event(StateEvent.THINK)
                    prefill_i = (len(tail_lines) - 1 - is_think)
                state.role = role

                if not prefill_i is None:
                    prefill_i += 1
                    if prefill_i < len(tail_lines):
                        out_tokens = tail_lines[prefill_i:]
                        out_tokens.append("\n")
                        state.prefill_tokens = out_tokens

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
                function_block_rstrip = function_block.rstrip()
                if function_block_rstrip.endswith(FUNCTION_END):
                    function_block = function_block_rstrip[:-len(FUNCTION_END)]

                func_name, tail = parse_name(function_block)
                if func_name is None:
                    # log
                    continue

                parameters = state.get_function_parameters(func_name)
                arguments, partial_param = get_arguments(tail or "", parameters)
                if partial_param:
                    partial = True

                parsed_calls.append(ParsedFunctionCall(name=func_name, arguments=arguments))
        return parsed_calls, partial
