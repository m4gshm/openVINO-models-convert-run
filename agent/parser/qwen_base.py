from agent.common.roles import ROLE_ASSISTANT
from agent.parser import ParserState, StateEvent, Parser, _is_conversation_start

ROLE = ROLE_ASSISTANT


CLOSE_TAG_PREF = "</"
OPEN_TAG_SUF = ">"

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"


THINK_START = "<think>"
THINK_END = "</think>"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

FIM_MIDDLE = "<|fim_middle|>"

END_OF_TEXT = "<|endoftext|>"


class QwenBaseParser(Parser):

    def is_erase(self, state: ParserState, token: str) -> bool:
        return super().is_erase(state, token) or self.is_fim_middle(state, token)

    def is_end(self, state: ParserState, token: str) -> bool:
        return token.strip() == END_OF_TEXT

    def is_fim_middle(self, state: ParserState, token: str) -> bool:
        return token.strip() == FIM_MIDDLE

    def is_think_end(self, state: ParserState, token: str) -> bool:
        return token.strip() == THINK_END

    def is_think_start(self, state: ParserState, token: str) -> bool:
        return token.strip() == THINK_START

    def is_conversation_start(self, state: ParserState, token: str) -> tuple[bool, str]:
        return _is_conversation_start(IM_START, token)

    def is_conversation_end(self, state: ParserState, token: str) -> bool:
        return IM_END == token.strip()

    def is_tool_call_start(self, state: ParserState, token: str) -> bool:
        return TOOL_CALL_START == token.strip()

    def is_tool_call_end(self, state: ParserState, token: str) -> bool:
        return TOOL_CALL_END == token.strip()

    def is_prompt_start_thinking(self, prompt: str) -> bool:
        return prompt.endswith(THINK_START, 0, len(prompt) - 1) if prompt.endswith(
            '\n') else prompt.endswith(THINK_START)

    def get_assistant_role_name(self) -> str:
        return ROLE
