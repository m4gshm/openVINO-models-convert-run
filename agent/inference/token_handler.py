import collections
import logging
import time
from datetime import timedelta
from enum import Enum
from typing import Sequence, SupportsInt, Callable, List, Literal

from openvino_genai.py_openvino_genai import Tokenizer, GenerationFinishReason
from pydantic import TypeAdapter, BaseModel

from agent.client.veai.tool_call_fixer import veai_fix_incorrect_arguments
from agent.common.roles import ROLE_ASSISTANT
from agent.common.time import format_time
from agent.openai.chat_api import new_chunk_response
from agent.openai.chat_completions_api import FunctionDefinition, CompletionResponse, ToolCall
from agent.parser import Parser, StateEvent, ParserState

log = logging.getLogger(__name__)


class TokenHandlerConfig(BaseModel):
    tool_call_parting_duration_warning: timedelta = timedelta(minutes=3)
    tool_call_parting_duration_limit: timedelta = timedelta(minutes=10)
    prevent_no_assistant_inference_output: bool = True
    no_conversation_counter_erased_max: int = 40
    no_conversation_counter_max: int = 20


class StopSignal(Enum):
    STOP = "stop", GenerationFinishReason.STOP
    TOOL_CALL = "tool_call", GenerationFinishReason.TOOL_CALL
    CANCEL = "cancel", GenerationFinishReason.STOP

    def __new__(cls, *args, **kwds):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _: str, finish_reason: GenerationFinishReason):
        self._finish_reason_ = finish_reason

    def __str__(self):
        return self.value

    @property
    def value(self) -> str:
        return self._value_

    @property
    def finish_reason(self):
        return self._finish_reason_


def get_finish_str(stop_signal: StopSignal) -> Literal["stop", "length", "tool_calls"]:
    return "tool_calls" if stop_signal == StopSignal.TOOL_CALL else "stop"


def get_stop_signal_by_finish_reason(finish_reason: GenerationFinishReason) -> StopSignal | None:
    for e in StopSignal:
        if e.finish_reason == finish_reason:
            return e
    return None


def line_encoded(last_line: str, tokenizer: Tokenizer) -> Sequence[SupportsInt]:
    encode = tokenizer.encode(last_line, True)
    data = encode.input_ids.data
    data_ = data[0]
    return data_.tolist()


def decode(tokens: Sequence[SupportsInt], tokenizer: Tokenizer) -> list[str]:
    return [tokenizer.decode(tokens=[token], skip_special_tokens=False) for token in tokens]


class TokenHandler:
    def __clean_phrase(self):
        self.phrase = ""
        self.phrase_tick = None

    def __init__(self,
                 tokenizer: Tokenizer,
                 parser: Parser,
                 init_chat_events: bool,
                 is_stop: Callable[[], bool] | None,
                 config: TokenHandlerConfig,
                 is_veai: bool,
                 supported_functions: dict[str, FunctionDefinition] | None = None):
        super().__init__()
        self.is_veai = is_veai

        self.tokenizer = tokenizer
        self.parser = parser
        state = parser.new_state(init_chat_events)
        if state:
            state.supported_functions = supported_functions if supported_functions else {}
        self.state = state
        self.is_chat_mode = init_chat_events
        self.is_stop = is_stop
        self.config = config
        self.prev_role = None
        self.token_conversation_start_number: int = -1
        self.expect_role = False
        self.phrase_tick: float | None = None
        self.phrase = ""
        self.generated = list[str]()
        self.tool_call_count = 0
        self.tool_call_phrase = ""
        self.tool_call_parsing_tick: float | None = None
        self.tool_call_parsing_start_time: float | None = None
        self.tool_call_parsing_long_time_warned: bool = False
        self.tool_call_parsing_max_time_warned: bool = False
        self.empty_conversation_counter = 0
        self.no_conversation_counter = 0
        self.no_conversation_counter_erased = 0
        self.stop_inference = False
        self.token_counter = 0

    def handle_token(self, tokens: collections.abc.Sequence[SupportsInt], stop_no_conversations=True) -> tuple[
        list[CompletionResponse], StopSignal | None]:

        is_stop = self.is_stop
        if is_stop and is_stop():
            log.info("stream finished by user disconnected")
            return [], StopSignal.STOP

        decoded_tokens = decode(tokens, self.tokenizer)
        return self.process_tokens(decoded_tokens, self.state, self.parser, stop_no_conversations)

    def process_tokens(self, decoded_tokens: list[str], state: ParserState, parser: Parser,
                       stop_no_conversations: bool = True, ) -> tuple[
        list[CompletionResponse], StopSignal | None]:
        result: list[CompletionResponse] = []
        try:
            for token in decoded_tokens:
                self.token_counter += 1
                log.debug(f"token '{token}', num {self.token_counter}")
                token_result, stop_signal = self.process_token(token, self.token_counter, state, parser,
                                                               stop_no_conversations)
                result += token_result
                if stop_signal:
                    # log
                    return result, stop_signal
        except Exception as e:
            log.error(f"streamer error: {e}", exc_info=e)
            return result, StopSignal.CANCEL

        return result, None

    def process_token(self, token: str, token_number: int, state: ParserState, parser: Parser,
                      stop_no_conversations=True) -> tuple[
        list[CompletionResponse], StopSignal | None]:
        now_time = time.perf_counter()
        result: list[CompletionResponse] = []
        stop_signal = None

        conversation_start, tail = parser.is_conversation_start(state, token)
        if conversation_start:
            self.conversation_start(state, tail, token_number)
        elif parser.is_conversation_end(state, token):
            result, stop_signal = self.conversation_end(state, token_number)
        elif self.expect_role and state.get_current_event() == StateEvent.CONVERSATION and token_number - self.token_conversation_start_number == 1:
            if len(token.rstrip()) > 0:  # conversation role
                self.set_role(token, state)
            else:
                log.debug("empty role for conversation start")
        elif parser.is_think_start(state, token):
            self.thinking_start(state)
        elif parser.is_think_end(state, token):
            self.thinking_end(state)
        elif parser.is_tool_call_start(state, token):
            self.tool_call_start(state, token)
        elif parser.is_tool_call_end(state, token):
            result, stop_signal = self.tool_call_end(state, token)
            state.expect_tool_response = True
        elif parser.is_tool_response_start(state, token):
            state.start_event(StateEvent.TOOL_RESPONSE)
            log.debug(f"tool response start: {token}")
        elif parser.is_tool_response_end(state, token):
            state.finish_current_event(StateEvent.TOOL_RESPONSE, log)
            log.debug(f"tool response end: {token}")
        elif parser.is_fim_middle(state, token):
            state.start_event(StateEvent.FIM_MIDDLE)
        elif parser.is_end(state, token):
            # log
            stop_signal = StopSignal.STOP
        else:
            last_state = state.get_current_event()
            if last_state == StateEvent.TOOL_CALL:
                self.tool_call_phrase += token
                parsing_time = timedelta(seconds=(now_time - self.tool_call_parsing_start_time))
                if not self.tool_call_parsing_long_time_warned and parsing_time >= self.config.tool_call_parting_duration_warning:
                    chunk = new_chunk_response(role=ROLE_ASSISTANT, content=f"Long parsing of tool call "
                                                                            f"({format_time(self.config.tool_call_parting_duration_warning)})")
                    result.append(chunk)
                    self.tool_call_parsing_long_time_warned = True
                elif self.tool_call_parsing_max_time_warned and parsing_time >= self.config.tool_call_parting_duration_limit:
                    chunk = new_chunk_response(role=ROLE_ASSISTANT, content=f"Tool call parsing exceeded time limit"
                                                                            f" {format_time(self.config.tool_call_parting_duration_limit)}.\n"
                                                                            f"```\n{self.tool_call_phrase}\n```")
                    result.append(chunk)
                    self.tool_call_parsing_max_time_warned = True
                tool_call_snapshot_time = now_time - self.tool_call_parsing_tick
                if tool_call_snapshot_time >= 10:
                    self.tool_call_parsing_tick = now_time
                    log.info(f"tool call part: {self.tool_call_phrase}")
            else:
                self.phrase = self.phrase + token
                generated = self.generated
                is_loop = False
                if generated:
                    prev_token = generated[-1]
                    if prev_token == token:
                        i = 1
                        for t in reversed(generated[:-1]):
                            if t != token:
                                break
                            if i >= 100:
                                log.error("looks like model is looping")
                                is_loop = True
                                break
                            i += 1

                if is_loop:
                    chunk = new_chunk_response(role=state.role, content="looks like model is looping")
                    result.append(chunk)
                    stop_signal = StopSignal.STOP
                else:
                    self.generated.append(token)

                    if self.phrase_tick is None:
                        self.phrase_tick = now_time

                    phrase_time = now_time - self.phrase_tick
                    if phrase_time >= 10:
                        self.phrase_tick = now_time
                        log.debug(f"phrase part: '{self.phrase.rstrip()}'")

                    phrase_end = self.phrase.endswith("\n")
                    if phrase_end:
                        phrase = self.phrase.rstrip()
                        if len(phrase) > 0:
                            self.__clean_phrase()
                            log.info(
                                f"{state.role} phrase: '{phrase}', last token num: {token_number}")

                    if not self.is_chat_mode:
                        result.append(new_chunk_response(role=state.role, content=token))
                    if not last_state and stop_no_conversations:
                        if parser.is_erase(state, token):
                            self.no_conversation_counter_erased += 1
                        elif not token.rstrip():
                            # empty or new line
                            self.no_conversation_counter_erased += 1
                        else:
                            log.debug("no more conversations")
                            self.no_conversation_counter += 1
                        if self.no_conversation_counter_erased > self.config.no_conversation_counter_erased_max:
                            log.debug(
                                f"empty conversations (erased) limits exceed ({self.no_conversation_counter_erased}), abort inference")
                            stop_signal = StopSignal.STOP
                        elif self.no_conversation_counter > self.config.no_conversation_counter_max:
                            log.debug(
                                f"empty conversations limits exceed ({self.no_conversation_counter}), abort inference")
                            stop_signal = StopSignal.STOP
                    else:
                        event = state.get_current_event()
                        if event == StateEvent.TOOL_CALL:
                            phrase_rstrip = self.phrase.rstrip()
                            log_msg = f"trying to out tool call as generated text: '{phrase_rstrip}'"
                            if len(phrase_rstrip) > 0:
                                log.warning(log_msg)
                            else:
                                log.debug(log_msg)
                        else:
                            erase = event == StateEvent.TOOL_RESPONSE or parser.is_erase(state, token)
                            is_assistant = ROLE_ASSISTANT == self.state.role
                            if not is_assistant:
                                log.warning(f"unexpected role {state.role}")
                            if is_assistant or not self.config.prevent_no_assistant_inference_output:
                                if not erase:
                                    chunk = new_chunk_response(role=state.role, content=token,
                                                               thinking=state.has_event(StateEvent.THINK))
                                    result.append(chunk)
                                else:
                                    log.debug(f"erase token: {token}")
                                    pass
                            else:
                                log.warning(
                                    f"prevent generating by unexpected role {state.role}, token '{token}'")
        state.finalize(token)
        return result, stop_signal

    def set_role(self, token: str, state: ParserState):
        self.expect_role = False
        self.prev_role = state.role

        new_role = ROLE_ASSISTANT if token == self.parser.get_assistant_role_name() else token
        state.role = new_role
        log.debug(f"set conversation role {state.role}, prev {self.prev_role}")

    def conversation_end(self, state: ParserState, token_number: int) -> tuple[
        list[CompletionResponse], Literal[StopSignal.CANCEL] | None]:
        result: list[CompletionResponse] = []
        stop_signal: Literal[StopSignal.CANCEL] | None = None
        if state.get_current_event() == StateEvent.TOOL_CALL:
            # sometimes Qwen3.5 ends tool call by end conversation token
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f"tool call ended by end conversation token '{self.tool_call_phrase}'")
            else:
                log.info("tool call ended by end conversation token")
            state.finish_current_event(StateEvent.TOOL_CALL)
            result = [self.handle_tool_call(state)]
        if state.get_current_event() == StateEvent.THINK:
            # The generated text is returned as a normal response because it was already sent as a thought,
            # but the model did not end it with a thought end marker.
            result = [new_chunk_response(role=self.state.role, content="".join(self.generated))]
            self.generated = ""
            state.finish_current_event(StateEvent.THINK)
        else:
            self.token_conversation_start_number = -1
            self.expect_role = False

            phrase = self.phrase.rstrip()
            if len(phrase) == 0:
                log.debug(f"empty conversation end, role {state.role}")
                self.empty_conversation_counter += 1
            else:
                self.empty_conversation_counter = 0
                log.info(
                    f"{state.role} conversation end by: {phrase}, last token num: {token_number}")
                self.__clean_phrase()

            if self.empty_conversation_counter > 20:
                log.warning(
                    f"many empty conversations ({self.empty_conversation_counter}), interrupt inference")
                result = []
                stop_signal = StopSignal.CANCEL
        state.finish_current_event(StateEvent.CONVERSATION)
        return result, stop_signal

    def conversation_start(self, state: ParserState, tail: str, token_number: int):
        state.start_event(StateEvent.CONVERSATION)
        self.no_conversation_counter = 0
        self.token_conversation_start_number = token_number

        phrase = self.phrase.rstrip()
        if len(phrase) > 0:
            log.info(
                f"phrase before conversation: '{phrase}', last token num: {token_number}")

        self.__clean_phrase()

        # check role in place
        if tail:
            self.set_role(tail, state)
        else:
            self.expect_role = True

    def handle_tool_call(self, state: ParserState) -> CompletionResponse:
        tool_call_expression = self.tool_call_phrase
        parsed_tool_calls, partial = self.parser.parse_tool_calls(state, tool_call_expression)
        if not parsed_tool_calls:
            log.info(
                f"phrase like tool calls: {tool_call_expression}")
            chunk = new_chunk_response(role=state.role, content=tool_call_expression)
        else:
            if partial:
                pass
            fixed_tool_calls = list(
                map(veai_fix_incorrect_arguments, parsed_tool_calls)) if self.is_veai else parsed_tool_calls
            if log.isEnabledFor(logging.INFO):
                adapter = TypeAdapter(List[ToolCall])
                log.info(
                    f"tool call: {adapter.dump_json(fixed_tool_calls).decode("utf-8")}")

            # self.tool_call_count += 1
            chunk = new_chunk_response(role=state.role, tool_calls=fixed_tool_calls)

        # self.tool_call_parsing_in_progress = False
        self.tool_call_phrase = ""
        self.tool_call_parsing_tick = None
        return chunk

    def tool_call_end(self, state: ParserState, token: str) -> tuple[
        list[CompletionResponse], Literal[StopSignal.TOOL_CALL]]:
        self.tool_call_parsing_start_time = None
        state.finish_current_event(expected_state=StateEvent.TOOL_CALL)
        log.debug(f"tool call end: {token}")
        self.tool_call_phrase += token

        result = [self.handle_tool_call(state)]
        stop_signal = StopSignal.TOOL_CALL
        return result, stop_signal

    def tool_call_start(self, state: ParserState, token: str):
        now_time = time.perf_counter()
        state.start_event(StateEvent.TOOL_CALL)
        log.debug(f"tool call start: {token}")

        phrase = self.phrase.rstrip()
        if len(phrase) > 0:
            log.info(f"{state.role} phrase before tool call: '{phrase}'")

        self.__clean_phrase()

        self.tool_call_parsing_tick = now_time
        self.tool_call_parsing_start_time = now_time
        self.tool_call_phrase += token

    def thinking_end(self, state: ParserState):
        now_time = time.perf_counter()
        state.finish_current_event(expected_state=StateEvent.THINK)
        if state.has_event(StateEvent.TOOL_CALL):
            self.tool_call_parsing_start_time = now_time
            log.warning(
                f"stop think token inside tool_call: '{self.tool_call_phrase}', phrase: '{self.phrase}'")
        log.debug("thinking is over")

    def thinking_start(self, state: ParserState):
        state.start_event(StateEvent.THINK)
        if state.has_event(StateEvent.TOOL_CALL):
            self.tool_call_parsing_start_time = None
            log.warning(f"start think token inside tool_call {self.tool_call_phrase}")
        log.debug("thinking is starting")
