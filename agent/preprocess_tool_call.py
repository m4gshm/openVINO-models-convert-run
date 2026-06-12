from agent.common.roles import ROLE_TOOL
from agent.common_openapi_model import ChatCompletionMessageParam


class FunctionCallResult:
    def __init__(self, name: str, result: str):
        self.name = name
        self.result = result


class PreprocessToolCall:
    max_repeated_tool_calls_with_the_same_result = 5
    max_messages_to_check = 5 * 4

    def check_loop_calls(self, messages: list[ChatCompletionMessageParam]) -> FunctionCallResult | None:
        last_message = messages[-1] if messages else False
        is_last_tool_call = last_message.role == ROLE_TOOL if last_message else False
        if not is_last_tool_call:
            return None


        repeated = 0
        # reverse loop from prelast message
        for i in range(len(messages)):
            if i >= self.max_messages_to_check:
                break
            message = messages[len(messages) - 2 - i]
            if message.role != ROLE_TOOL:
                continue
            name = message.name
            content = message.content
            if not name or not content:
                continue

            is_equal = name == last_message.name and content == last_message.content
            if is_equal:
                repeated += 1
            else:
                break

            if repeated >= self.max_repeated_tool_calls_with_the_same_result:
                return FunctionCallResult(name=name, result=content)

        return None
