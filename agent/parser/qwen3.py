import json
import logging
import re
import time
from typing import Any

import json_repair

from common.openai_api import new_tool_call
from common.openai_model import ToolCall, FunctionCall, FunctionDefinition
from parser.base import Parser

EXPECTED_PROPERTY_TYPE = 'type'
EXPECTED_PARAMETERS_PROPERTIES = "properties"

CLOSE_TAG_PREF = "</"
OPEN_TAG_SUF = ">"

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"

FUNCTION_START_PREF = "<function"
FUNCTION_START = FUNCTION_START_PREF + OPEN_TAG_SUF
FUNCTION_END = CLOSE_TAG_PREF + "function>"

PARAMETER_START_PREF = "<parameter"
PARAMETER_END = CLOSE_TAG_PREF + "parameter>"

REASONING_START = "<think>"
REASONING_END = "</think>"

CONVERSATION_START = "<|im_start|>"
CONVERSATION_END = "<|im_end|>"

log = logging.getLogger(__name__)


class Qwen3Parser(Parser):
    def think_is_over(self,subword: str) -> bool:
        return subword.strip() == REASONING_END

    def think_is_started(self,subword: str) -> bool:
        return subword.strip() == REASONING_START

    def is_conversation_start(self,text: str) -> bool:
        return CONVERSATION_START == text.strip()

    def is_conversation_end(self,text: str) -> bool:
        return CONVERSATION_END == text.strip()

    def is_tool_call_start(self,text: str) -> bool:
        return TOOL_CALL_START == text.strip()

    def is_tool_call_end(self,text: str) -> bool:
        return TOOL_CALL_END == text.strip()

    def is_prompt_start_thinking(self,prompt: str) -> bool:
        return prompt.endswith(REASONING_START, 0, len(prompt) - 1) if prompt.endswith(
            '\n') else prompt.endswith(REASONING_START)

    def is_partial_tool_call(self,text: str) -> bool:
        strip = text.strip()
        if strip.endswith(TOOL_CALL_END):
            return False
        elif strip.endswith(FUNCTION_END):
            return False
        else:
            return True

    def parse_tool_calls(self,text: str, supported_functions: dict[str, FunctionDefinition] | None = None) -> tuple[
        list[ToolCall], bool]:
        text = text.lstrip()
        tool_call_blocks = text.split(TOOL_CALL_START)
        ts = int(time.time())

        parsed_calls: list[ToolCall] = []
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

                func_name, tail = self.parse_name(function_block)
                if func_name is None:
                    # log
                    continue

                function = supported_functions.get(func_name) if supported_functions else None

                parameters = function.parameters if function is not None else []
                arguments, partial_param = self.get_arguments(tail, parameters)
                if partial_param:
                    partial = True

                parsed_calls.append(new_tool_call(FunctionCall(name=func_name, arguments=arguments)))
        return parsed_calls, partial

    def get_arguments(self,function_block: str, expected_parameters: dict[str, Any] | None = None) -> tuple[str, bool]:
        expected_properties: dict[str, Any] = expected_parameters.get(EXPECTED_PARAMETERS_PROPERTIES,
                                                                      {}) if expected_parameters else {}

        arguments: dict[str, str | dict[str, Any] | list[Any]] = {}

        partial = False
        parameter_blocks = function_block.split(PARAMETER_START_PREF)
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

                expected_property: dict[str, Any] = expected_properties.get(param_name_norm, {})
                expected_type = expected_property.get(EXPECTED_PROPERTY_TYPE,
                                                      'string') if expected_property else 'string'
                if expected_type == 'array' or expected_type == 'object':
                    try:
                        structured_parameter: dict[str, Any] | list[Any] = json.loads(param_value_norm)
                    except json.decoder.JSONDecodeError as e:
                        log.debug(
                            f"function parameter parsing error, parameter '{param_name}', value '{param_value_norm}', "
                            f"expected_type '{expected_type}': {e}")
                        structured_parameter: dict[str, Any] | list[Any] = json_repair.loads(param_value_norm)
                        log.debug(f"repaired parameter value '{structured_parameter}'")

                    arguments[param_name_norm] = structured_parameter
                else:
                    arguments[param_name_norm] = param_value_norm

        if isinstance(arguments, dict):
            arguments_str = json.dumps(arguments, ensure_ascii=False)
        else:
            arguments_str = str(arguments)
        return arguments_str, partial

    def parse_name(self,parameters_block) -> tuple[str | None, str | None]:
        match = re.search(f"=(.*?){OPEN_TAG_SUF}(.*)", parameters_block, re.DOTALL)
        if match:
            name = match.group(1).strip()
            tail = match.group(2).strip()
            return name, tail
        else:
            return None, None
