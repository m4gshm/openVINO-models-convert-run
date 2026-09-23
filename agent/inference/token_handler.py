import collections
import logging
import time
from datetime import timedelta, datetime, timezone
from enum import Enum
from typing import Sequence, SupportsInt, Callable, List, Literal

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openvino_genai.py_openvino_genai import Tokenizer, GenerationFinishReason
from pydantic import TypeAdapter, BaseModel

from agent import inference
from agent.client.user_context import UserContext
from agent.common.time import format_time
from agent.inference.loop_error import LoopError
from agent.inference.phrase import Phrase
from agent.openai.chat_api import new_chat_completion_chunk, new_tool_call, new_stop_response, ROLE_ASSISTANT, Role
from agent.openai.chat_completions_api import FunctionDefinitionParameters
from agent.parser import Parser, StateEvent, ParserState, ParsedFunctionCall

log = logging.getLogger(__name__)
log_inference_generated = logging.getLogger(inference.log.name + ".generated")


class TokenHandlerConfig(BaseModel):
    tool_call_parting_duration_warning: timedelta = timedelta(minutes=3)
    tool_call_parting_duration_limit: timedelta = timedelta(minutes=10)
    prevent_no_assistant_inference_output: bool = True
    no_conversation_counter_erased_max: int = 40
    no_conversation_counter_max: int = 20
    empty_conversation_counter_max: int = 20
    is_detect_looped_inference: bool = True


class StopSignal(Enum):
    STOP = "stop", GenerationFinishReason.STOP
    TOOL_CALL = "tool_call", GenerationFinishReason.TOOL_CALL
    CANCEL = "cancel", GenerationFinishReason.STOP
    LENGTH = "length", GenerationFinishReason.LENGTH

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


def get_finish_str(stop_signal: StopSignal) -> Literal["stop", "length", "tool_calls", "content_filter"]:
    if stop_signal == StopSignal.TOOL_CALL:
        return "tool_calls"
    elif stop_signal == StopSignal.CANCEL:
        return "content_filter"
    elif stop_signal == StopSignal.LENGTH:
        return "length"
    else:
        return "stop"


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


def to_openai_tool_call(function: ParsedFunctionCall) -> ChoiceDeltaToolCall:
    return new_tool_call(function.to_openai_function_call())


def markdown_bold(value) -> str:
    return f"**{value}**"


def markdown_back_tick(value) -> str:
    return f"`{value}`"


def markdown_file_content(value, mime: str = "") -> str:
    return f"```{mime}\n{value}\n```"


def markdown_json(value: str) -> str:
    return markdown_file_content(value, "json")


def markdown_tool_call_loop_error(loop_cause: str | None, loop_error: str) -> str | None:
    return ((markdown_bold("WARNING: " + loop_cause) + "\n") if loop_cause else "") + markdown_file_content(
        loop_error)


def now() -> float:
    return time.perf_counter()


def print_log(phrase: Phrase):
    full = phrase.full
    if full:
        log_inference_generated.debug(full)


type ToolFixer = Callable[[ParsedFunctionCall, UserContext | None], list[ParsedFunctionCall] | ParsedFunctionCall]


class TokenHandler:

    def __init__(self,
                 tokenizer: Tokenizer,
                 prompt: str,
                 parser: Parser,
                 init_chat_events: bool,
                 is_stop: Callable[[], bool] | None,
                 config: TokenHandlerConfig,
                 tool_fixer: ToolFixer | None = None,
                 user_context: UserContext | None = None,
                 supported_functions: dict[str, FunctionDefinitionParameters] | None = None):
        super().__init__()
        self.processor = TokenProcessor(prompt=prompt, parser=parser, init_chat_events=init_chat_events,
                                        config=config, tool_fixer=tool_fixer, user_context=user_context,
                                        supported_functions=supported_functions)
        self.start_time: datetime | None = None
        self.tokenizer = tokenizer
        self.is_prefill_out = False
        self.is_stop = is_stop

    def get_stat_info(self):
        return self.processor.get_stat_info()

    def handle_tokens(self, tokens: collections.abc.Sequence[SupportsInt]) -> tuple[
        list[ChatCompletionChunk], StopSignal | None]:
        is_stop = self.is_stop
        if is_stop and is_stop():
            log.info("handle tokens is stopped")
            return [], StopSignal.CANCEL
        decoded_tokens = decode(tokens, self.tokenizer)
        prefill_tokens = self.processor.state.prefill_tokens
        if not self.is_prefill_out and prefill_tokens:
            all_tokens = []
            all_tokens.extend(prefill_tokens)
            all_tokens.extend(decoded_tokens)
            decoded_tokens = all_tokens
            self.is_prefill_out = True
        return self.processor.process_tokens(decoded_tokens)


def end(amount: int) -> str:
    return "s" if amount != 1 else ""


def fix_tool_calls(tool_fixer: ToolFixer, user_context: UserContext | None,
                   parsed_function_calls: list[ParsedFunctionCall]) -> list[ParsedFunctionCall]:
    if not tool_fixer:
        fixed_tool_calls = parsed_function_calls
    else:
        fixed_tool_calls: list[ParsedFunctionCall] = []
        for tc in parsed_function_calls:
            arguments = tool_fixer(tc, user_context)
            if isinstance(arguments, list):
                for arg in arguments:
                    fixed_tool_calls.append(arg)
            else:
                fixed_tool_calls.append(arguments)
        if log.isEnabledFor(logging.DEBUG):
            adapter = TypeAdapter(List[ParsedFunctionCall])
            parsed_str = adapter.dump_json(parsed_function_calls).decode("utf-8")
            fixed_str = adapter.dump_json(fixed_tool_calls).decode("utf-8")
            log.debug(f"tool calls: parsed={parsed_str}, fixed={fixed_str}")
    return fixed_tool_calls


def fix_tool_calls_and_convert_to_choice_delta_tool_calls(tool_fixer: ToolFixer, user_context: UserContext | None,
                                                          parsed_function_calls: list[ParsedFunctionCall]
                                                          ) -> list[ChoiceDeltaToolCall]:
    return list(map(to_openai_tool_call, fix_tool_calls(tool_fixer, user_context, parsed_function_calls)))


def fix_and_chat_complete_parsed_tool_calls(tool_fixer: ToolFixer,
                                            user_context: UserContext | None,
                                            parsed_function_calls: list[ParsedFunctionCall],
                                            tool_call_expression: str, role: Role | None
                                            ) -> tuple[ChatCompletionChunk, StopSignal]:
    tool_calls = fix_tool_calls_and_convert_to_choice_delta_tool_calls(tool_fixer, user_context, parsed_function_calls)
    if len(tool_calls) == 0:
        log.info(f"unparsed tool calls: {tool_call_expression}")
        return new_chat_completion_chunk(role=role, content=tool_call_expression), StopSignal.CANCEL
    else:
        return new_chat_completion_chunk(role=role, tool_calls=tool_calls), StopSignal.TOOL_CALL


class TokenProcessor:
    def __clean_phrase(self):
        print_log(self.phrase)
        new_phrase = Phrase(is_detect_looped_inference=self.config.is_detect_looped_inference)
        self.phrase = new_phrase
        self.empty_conversation_counter = 0
        self.phrase_tick = None
        return new_phrase

    def __clean_tool_call_phrase(self):
        call_phrase = self.tool_call_phrase
        print_log(call_phrase)
        phrase = Phrase(is_detect_looped_inference=self.config.is_detect_looped_inference)
        self.tool_call_phrase = phrase
        self.tool_call_parsing_tick = None
        self.tool_call_parsing_start_time = None

    def __init__(self,
                 prompt: str,
                 parser: Parser,
                 init_chat_events: bool,
                 config: TokenHandlerConfig,
                 tool_fixer: ToolFixer | None,
                 user_context: UserContext | None = None,
                 supported_functions: dict[str, FunctionDefinitionParameters] | None = None):
        super().__init__()
        self.no_conversation_counter = None
        self.create_time = datetime.now(timezone.utc)
        self.start_time: datetime | None = None
        self.user_context = user_context
        self.tool_fixer: ToolFixer | None = tool_fixer

        self.parser = parser
        state = parser.new_state(prompt, supported_functions, init_chat_events)
        self.state = state
        self.is_chat_mode = init_chat_events
        self.config = config
        self.prev_role = None
        self.token_conversation_start_number: int = -1
        self.expect_role = False
        self.phrase_tick: float | None = None
        self.phrase = Phrase(is_detect_looped_inference=self.config.is_detect_looped_inference)
        self.tool_call_phrase = Phrase(is_detect_looped_inference=self.config.is_detect_looped_inference)
        self.tool_call_parsing_tick: float | None = None
        self.tool_call_parsing_start_time: float | None = None
        self.tool_call_parsing_long_time_warned: bool = False
        self.tool_call_parsing_max_time_warned: bool = False
        self.empty_conversation_counter = 0
        self.stop_inference = False
        self.token_counter = 0
        self.role_initialized = False

    def get_stat_info(self) -> str | None:
        stat_info = None
        start_time = self.start_time
        if start_time:
            now = datetime.now(timezone.utc)
            time_delta = now - start_time
            amount = self.token_counter
            total_seconds = time_delta.total_seconds()
            ttft = (start_time - self.create_time).total_seconds()
            stat_info = (
                f"generated {amount} token{end(amount)} in {time_delta} sec, {amount / total_seconds} t/sec), ttft {ttft} sec"
            )
        return stat_info

    def process_tokens(self, decoded_tokens: list[str]) -> tuple[
        list[ChatCompletionChunk], StopSignal | None]:
        if self.start_time is None:
            self.start_time = datetime.now(timezone.utc)
        state = self.state
        result: list[ChatCompletionChunk] = []
        try:
            for token in decoded_tokens:
                self.token_counter += 1
                log.debug(f"token '{token}', num {self.token_counter}")
                token_result, stop_signal = self.process_token(token, self.token_counter)
                if not self.role_initialized and token_result:
                    choices = token_result[0].choices
                    if choices:
                        delta = choices[0].delta
                        if delta:
                            delta.role = self.state.role
                            self.role_initialized = True

                result += token_result
                if stop_signal:
                    # log
                    return result, stop_signal
        except Exception as e:
            log.error(f"streamer error: {e}", exc_info=e)
            result.append(new_stop_response(role=state.role, content=markdown_bold(f"StreamerError:{e}")))
            return result, StopSignal.CANCEL

        return result, None

    def process_token(self, token: str, token_number: int) -> tuple[
        list[ChatCompletionChunk], StopSignal | None]:
        now_time = now()
        result: list[ChatCompletionChunk] = []
        stop_signal = None

        state = self.state
        parser = self.parser

        conversation_start, tail = parser.is_conversation_start(state, token)
        current_event = state.get_current_event()
        if conversation_start:
            self.conversation_start(state, tail, token_number)
        elif parser.is_text_end(state, token):
            # ignore stop signal
            result, _ = self.conversation_end(state, token_number)
        elif parser.is_sequence_end(state, token):
            conversation_end_result, stop_signal = self.conversation_end(state, token_number)
            result.extend(conversation_end_result)
        elif self.expect_role and current_event == StateEvent.CONVERSATION and token_number - self.token_conversation_start_number == 1:
            if len(token.rstrip()) > 0:  # conversation role
                self.set_role(token, state)
            else:
                log.debug("empty role for conversation start")
        elif parser.is_think_start(state, token):
            if current_event == StateEvent.TOOL_CALL:
                log.debug(f"tool call is finished by generated thinking tag: '{token}'")
                tool_call, stop_signal = self.tool_call_end(state, token)
                result.append(tool_call)
            else:
                self.thinking_start(state)
        elif parser.is_think_end(state, token):
            self.thinking_end(state)
        elif parser.is_tool_call_start(state, token):
            if current_event == StateEvent.TOOL_CALL:
                log.debug(f"tool call is finished by starting new tool call: '{token}'")
                tool_call, stop_signal = self.tool_call_end(state, token)
                if stop_signal == StopSignal.TOOL_CALL:
                    stop_signal = None
                result.append(tool_call)
            else:
                self.tool_call_start(state, token)
        elif parser.is_tool_call_end(state, token):
            tool_call, stop_signal = self.tool_call_end(state, token)
            result.append(tool_call)
            if parser.is_support_multiple_tool_calls(state, token) and stop_signal == StopSignal.TOOL_CALL:
                stop_signal = None

        elif parser.is_tool_response_start(state, token):
            state.start_event(StateEvent.TOOL_RESPONSE)
            log.debug(f"tool response start: {token}")
        elif parser.is_tool_response_end(state, token):
            state.finish_current_event(StateEvent.TOOL_RESPONSE, log)
            log.debug(f"tool response end: {token}")
        elif parser.is_fim_middle(state, token):
            state.start_event(StateEvent.FIM_MIDDLE)
        elif parser.is_end(state, token):
            log.debug(f"parsed end: {token}")
            stop_signal = StopSignal.STOP
        elif current_event == StateEvent.TOOL_CALL:
            loop_error: str | None = None
            tool_call_phrase = self.tool_call_phrase
            try:
                tool_call_phrase.add_token(token)
            except LoopError as e:
                log.error(f"tool call error (loop): {e}")
                loop_error = markdown_tool_call_loop_error(e.message + " (tool call)", e.payload)

            if loop_error:
                tool_call, stop_signal = self.tool_call_end(state, token)
                result.append(new_chat_completion_chunk(role=state.role, content=loop_error))
                result.append(tool_call)
            else:
                parsing_time = timedelta(seconds=(now_time - self.tool_call_parsing_start_time))
                if not self.tool_call_parsing_long_time_warned and parsing_time >= self.config.tool_call_parting_duration_warning:
                    time = format_time(self.config.tool_call_parting_duration_warning)
                    warning_msg = markdown_bold(f"WARNING: Long tool call calling({time})") + "\n"
                    result.append(new_chat_completion_chunk(role=state.role, content=warning_msg))
                    self.tool_call_parsing_long_time_warned = True
                elif self.tool_call_parsing_max_time_warned and parsing_time >= self.config.tool_call_parting_duration_limit:
                    time = format_time(self.config.tool_call_parting_duration_limit)
                    warning_msg = markdown_bold(f"WARNING: Tool call parsing exceeded time limit {time}.\n"
                                                ) + markdown_file_content(tool_call_phrase.full)
                    result.append(new_chat_completion_chunk(role=state.role, content=warning_msg))
                    self.tool_call_parsing_max_time_warned = True
                tool_call_snapshot_time = now_time - self.tool_call_parsing_tick
                if tool_call_snapshot_time >= 10:
                    self.tool_call_parsing_tick = now_time
                    log.debug(f"tool call part: {tool_call_phrase.full}")
        else:
            phrase: Phrase = self.phrase
            handle_delay_end = False
            if state.is_streaming_delayed:
                log.debug(f"delayed phrase: {phrase.full}")
                delay_end = parser.is_delay_streaming_end(state, token)
                if delay_end:
                    log.info(f"delay streaming finish: token '{token}', token num '{token_number}'")
                    handle_delay_end = True
                    state.is_streaming_delayed = False
            else:
                delay_start = parser.is_delay_streaming_start(state, token)
                if delay_start:
                    state.is_streaming_delayed = delay_start
                    log.info(f"delay streaming start: token '{token}', token num '{token_number}'")
                    # restart phrase
                    phrase = self.__clean_phrase()

            loop_error: str | None = None
            try:
                new_lines = phrase.add_token(token)
            except LoopError as e:
                new_lines = None
                log.error(f"loop error: {e}")
                loop_error = markdown_tool_call_loop_error(e.message, e.payload)

            thinking = state.has_event(StateEvent.THINK)
            is_handle_delay_end = state.is_streaming_delayed or handle_delay_end
            if loop_error:
                if is_handle_delay_end:
                    result.append(new_chat_completion_chunk(role=state.role, content=phrase.full,
                                                            thinking=thinking))
                result.append(new_chat_completion_chunk(role=state.role, content=loop_error))
                stop_signal = StopSignal.CANCEL
            elif is_handle_delay_end:
                pass
            else:
                if self.phrase_tick is None:
                    self.phrase_tick = now_time

                if new_lines and len(new_lines) > 0:
                    log.info(f"{state.role} phrase: '{"".join(new_lines)}', last token num: {token_number}")

                if handle_delay_end:
                    delayed_result, stop_signal = self.handle_delayed(phrase)
                    result.extend(delayed_result)
                else:
                    if not self.is_chat_mode:
                        result.append(new_chat_completion_chunk(role=state.role, content=token))

                    erase = current_event == StateEvent.TOOL_RESPONSE or parser.is_erase(state, token)
                    is_assistant = ROLE_ASSISTANT == state.role
                    if not is_assistant:
                        log.warning(f"unexpected role {state.role}")
                    if is_assistant or not self.config.prevent_no_assistant_inference_output:
                        if not erase:
                            result.append(new_chat_completion_chunk(role=state.role, content=token,
                                                                    thinking=thinking))
                        else:
                            log.debug(f"erase token: {token}")
                    else:
                        log.warning(f"prevent generating by unexpected role {state.role}, token '{token}'")
        state.finalize(token)
        return result, stop_signal

    def handle_delayed(self, phrase: Phrase) -> tuple[list[ChatCompletionChunk], StopSignal | None]:
        result = list[ChatCompletionChunk]()
        parser = self.parser
        state = self.state
        delayed = parser.handle_delayed_phrase(phrase)
        thinking = state.has_event(StateEvent.THINK)
        if delayed is None:
            stop_signal = None
            result.append(new_chat_completion_chunk(role=state.role, content=phrase.full, thinking=thinking))
        else:
            tool_calls = fix_tool_calls_and_convert_to_choice_delta_tool_calls(self.tool_fixer,
                                                                               self.user_context,
                                                                               delayed.tool_calls)
            stop_signal = StopSignal.TOOL_CALL if len(tool_calls) > 0 else None
            chunk = new_chat_completion_chunk(role=state.role, content=delayed.content,
                                              tool_calls=tool_calls, thinking=thinking)
            result.append(chunk)
        return result, stop_signal

    def set_role(self, token: str, state: ParserState):
        self.expect_role = False
        self.prev_role = state.role

        new_role = ROLE_ASSISTANT if token == self.parser.get_assistant_role_name() else token
        state.role = new_role
        log.debug(f"set conversation role {state.role}, prev {self.prev_role}")

    def conversation_end(self, state: ParserState, token_number: int) -> tuple[
        list[ChatCompletionChunk], StopSignal | None]:
        result: list[ChatCompletionChunk] = []
        stop_signal: Literal[StopSignal.STOP, StopSignal.TOOL_CALL, StopSignal.CANCEL]
        if state.is_streaming_delayed:
            delayed_result, stop_signal = self.handle_delayed(self.phrase)
            result.extend(delayed_result)
            if not stop_signal:
                stop_signal = StopSignal.STOP
            state.is_streaming_delayed = False
        elif state.get_current_event() == StateEvent.TOOL_CALL:
            # sometimes Qwen3.5 ends tool call by end conversation token
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f"tool call ended by end conversation token '{self.tool_call_phrase.full}'")
            else:
                log.info("tool call ended by end conversation token")
            state.finish_current_event(StateEvent.TOOL_CALL)
            resp, stop_signal = self.handle_tool_call(state)
            result.append(resp)
        else:
            stop_signal = StopSignal.STOP

        phrase = self.phrase
        if state.get_current_event() == StateEvent.THINK:
            # The generated text is returned as a normal response because it was already sent as a thought,
            # but the model did not end it with a thought end marker.
            result = [new_chat_completion_chunk(role=state.role, content="".join(phrase.full))]
            state.finish_current_event(StateEvent.THINK)
        else:
            self.token_conversation_start_number = -1
            self.expect_role = False

            phrase = phrase.full.rstrip()
            if len(phrase) == 0:
                log.debug(f"empty conversation end, role {state.role}")
                self.empty_conversation_counter += 1
            else:
                self.empty_conversation_counter = 0
                log.info(
                    f"{state.role} conversation end by: {phrase}, last token num: {token_number}")
                self.__clean_phrase()

            if self.empty_conversation_counter > self.config.empty_conversation_counter_max:
                warning_msg = f"many empty conversations ({self.empty_conversation_counter}), interrupt inference"
                log.warning(warning_msg)
                result.append(new_chat_completion_chunk(role=state.role, content=markdown_bold(warning_msg)))
                stop_signal = StopSignal.CANCEL
        state.finish_current_event(StateEvent.CONVERSATION)
        return result, stop_signal

    def conversation_start(self, state: ParserState, tail: str, token_number: int):
        state.start_event(StateEvent.CONVERSATION)
        self.no_conversation_counter = 0
        self.token_conversation_start_number = token_number

        phrase = self.phrase.full.rstrip()
        if len(phrase) > 0:
            log.info(
                f"phrase before conversation: '{phrase}', last token num: {token_number}")

        # check role in place
        if tail:
            self.set_role(tail, state)
        else:
            self.expect_role = True

    def handle_tool_call(self, state: ParserState) -> tuple[ChatCompletionChunk, StopSignal]:
        parser = self.parser
        tool_call_phrase = self.tool_call_phrase
        tool_call_expression = tool_call_phrase.full
        parsed_function_calls, _ = parser.parse_tool_calls(state, tool_call_expression)
        tool_calls = fix_tool_calls_and_convert_to_choice_delta_tool_calls(self.tool_fixer, self.user_context,
                                                                           parsed_function_calls)
        if len(tool_calls) == 0:
            log.info(f"unparsed tool calls: {tool_call_expression}")
            chunk, stop_signal = new_chat_completion_chunk(role=state.role,
                                                           content=tool_call_expression), StopSignal.CANCEL
        else:
            chunk, stop_signal = new_chat_completion_chunk(role=state.role, tool_calls=tool_calls), StopSignal.TOOL_CALL
        self.__clean_tool_call_phrase()
        return chunk, stop_signal

    def tool_call_end(self, state: ParserState, token: str | None) -> tuple[ChatCompletionChunk, StopSignal]:
        self.tool_call_parsing_start_time = None
        tool_call_phrase = self.tool_call_phrase
        state.finish_current_event(expected_state=StateEvent.TOOL_CALL)
        if token:
            try:
                tool_call_phrase.add_token(token)
            except LoopError as e:
                log.warning(f"loop at the end of tool call {tool_call_phrase.full}")
        log.debug(f"tool call end: {tool_call_phrase.full}")

        return self.handle_tool_call(state)

    def tool_call_start(self, state: ParserState, token: str):
        state.start_event(StateEvent.TOOL_CALL)
        log.debug(f"tool call start: {token}")

        phrase = self.phrase.full.rstrip()
        if len(phrase) > 0:
            log.info(f"{state.role} phrase before tool call: '{phrase}'")
            self.__clean_phrase()

        now_time = now()
        self.tool_call_parsing_tick = now_time
        self.tool_call_parsing_start_time = now_time
        self.tool_call_phrase.add_token(token)

    def thinking_end(self, state: ParserState):
        now_time = now()
        state.finish_current_event(expected_state=StateEvent.THINK)
        if state.has_event(StateEvent.TOOL_CALL):
            self.tool_call_parsing_start_time = now_time
            log.warning(
                f"stop think token inside tool_call: '{self.tool_call_phrase.full}', phrase: '{self.phrase.full}'")
        log.debug("thinking is over")

    def thinking_start(self, state: ParserState):
        state.start_event(StateEvent.THINK)
        if state.has_event(StateEvent.TOOL_CALL):
            self.tool_call_parsing_start_time = None
            log.warning(f"start think token inside tool_call {self.tool_call_phrase.full}")
        log.debug("thinking is starting")
