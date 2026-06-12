from common_openapi_model import ChatCompletionMessageParam


class PreprocessToolCall:
    max_repeated_tool_calls_with_the_same_result = 5

    class FunctionCallResult:
        def __init__(self, name: str, result: str):
            self.name = name
            self.result = result

    class FunctionCall:
        class ContentStat:
            count: int = 0
            indexes: list[int] = []

        contents: dict[str, ContentStat] = {}

    functions: dict[str, FunctionCall] = {}

    def check_loop_call(self, index: int, message: ChatCompletionMessageParam) -> FunctionCallResult | None:
        name = message.name
        content = message.content
        if name and isinstance(content, str):
            function_call = self.functions.setdefault(name, PreprocessToolCall.FunctionCall())

            contents = function_call.contents

            content_stat = contents.setdefault(content, PreprocessToolCall.FunctionCall.ContentStat())
            content_stat.count += 1
            content_stat.indexes.append(index)

            loop_tool_call = content_stat.count >= self.max_repeated_tool_calls_with_the_same_result
            if loop_tool_call:
                return PreprocessToolCall.FunctionCallResult(name=name, result=content)

        return None
