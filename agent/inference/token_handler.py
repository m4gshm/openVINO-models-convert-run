"""Converts decoded model tokens into OpenAI chat completion chunks."""
import collections
import logging
import time
from datetime import timedelta, datetime, timezone
from enum import Enum
from typing import Sequence, SupportsInt, Callable, List, Literal, cast

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
# Native openvino_genai extension exposes these members dynamically.
from openvino_genai.py_openvino_genai import Tokenizer, GenerationFinishReason  # pylint: disable=no-name-in-module
from pydantic import TypeAdapter, BaseModel

from agent import inference
from agent.client.user_context import UserContext
from agent.common.markdown import markdown_bold, markdown_file_content, markdown_tool_call_loop_error
from agent.common.time import format_time
from agent.inference.loop_error import LoopError
from agent.inference.phrase import Phrase
from agent.openai.chat_api import new_chat_completion_chunk, new_tool_call, new_stop_response, ROLE_ASSISTANT, Role
from agent.openai.chat_completions_api import FunctionDefinitionParameters
from agent.parser import Parser, StateEvent, ParserState, ParsedFunctionCall

log = logging.getLogger(__name__)
log_inference_generated = logging.getLogger(inference.log.name + ".generated")


class TokenHandlerConfig(BaseModel):
    """Tuning knobs of the token stream handler."""
    tool_call_parting_duration_warning: timedelta = timedelta(minutes=3)
    tool_call_parting_duration_limit: timedelta = timedelta(minutes=10)
    prevent_no_assistant_inference_output: bool = True
    no_conversation_counter_erased_max: int = 40
    no_conversation_counter_max: int = 20
    empty_conversation_counter_max: int = 20
    is_detect_looped_inference: bool = True
    # stop phrases of the request: a generated token equal to one of them ends the generation;
    # matched on the raw token stream, single-token form (like the Qwen FIM special tags)
    stop_strings: tuple[str, ...] = ()


class StopSignal(Enum):
    """Stop reasons paired with the native GenerationFinishReason."""
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
    def value(self) -> str:  # pylint: disable=invalid-overridden-method  # intentional: expose the str value alongside the native finish reason
        return self._value_

    @property
    def finish_reason(self):
        """Return the native GenerationFinishReason paired with this stop signal."""
        return self._finish_reason_


def get_finish_str(stop_signal: StopSignal) -> Literal["stop", "length", "tool_calls", "content_filter"]:
    """Map a stop signal to the OpenAI finish reason string."""
    if stop_signal == StopSignal.TOOL_CALL:
        return "tool_calls"
    if stop_signal == StopSignal.CANCEL:
        return "content_filter"
    if stop_signal == StopSignal.LENGTH:
        return "length"
    return "stop"


def get_stop_signal_by_finish_reason(finish_reason: GenerationFinishReason) -> StopSignal | None:
    """Reverse lookup of the stop signal by the native finish reason."""
    for e in StopSignal:
        if e.finish_reason == finish_reason:
            return e
    return None


def line_encoded(last_line: str, tokenizer: Tokenizer) -> Sequence[SupportsInt]:
    """Encode a line into token ids with the model tokenizer."""
    encode = tokenizer.encode(last_line, True)
    data = encode.input_ids.data
    data_ = data[0]
    return data_.tolist()


def decode(tokens: Sequence[SupportsInt], tokenizer: Tokenizer) -> list[str]:
    """Decode token ids back to strings, keeping special tokens."""
    return [tokenizer.decode(tokens=[token], skip_special_tokens=False) for token in tokens]


def to_openai_tool_call(function: ParsedFunctionCall) -> ChoiceDeltaToolCall:
    """Convert a parsed function call into an OpenAI tool call chunk."""
    return new_tool_call(function.to_openai_function_call())


def now() -> float:
    """Return the monotonic clock used for parsing time measurements."""
    return time.perf_counter()


def print_log(phrase: Phrase):
    """Log the accumulated phrase into the generated logger."""
    full = phrase.full
    if full:
        log_inference_generated.debug(full)


ToolFixer = Callable[[ParsedFunctionCall, UserContext | None], list[ParsedFunctionCall] | ParsedFunctionCall]


class TokenHandler:
    """Converts decoded model tokens into OpenAI chat completion chunks."""

    def __init__(self,  # pylint: disable=too-many-arguments,too-many-positional-arguments  # constructor of the stream handler, args mirror engine wiring
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
        self.stop_check = is_stop

    def is_stop(self) -> bool:
        """Whether the generation must stop: the server signalled it or the client is gone.

        A request without a stop check never stops on its own.
        """
        stop_check = self.stop_check
        if stop_check is None:
            return False
        return stop_check()

    @property
    def state(self) -> ParserState:
        """Parsing state of the processed stream."""
        return self.processor.state

    def conversation_end(self, state: ParserState, token_number: int) -> tuple[
        list[ChatCompletionChunk], StopSignal | None]:
        """Finalize the conversation of the processed stream."""
        return self.processor.conversation_end(state, token_number)

    def get_stat_info(self):
        """Return generation statistics of the processor."""
        return self.processor.get_stat_info()

    def handle_tokens(self, tokens: collections.abc.Sequence[SupportsInt]) -> tuple[
        list[ChatCompletionChunk], StopSignal | None]:
        """Decode the incoming tokens and process them into chat completion chunks."""
        if self.is_stop():
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
    """Return the English plural suffix for the given amount."""
    return "s" if amount != 1 else ""


def fix_tool_calls(tool_fixer: ToolFixer | None, user_context: UserContext | None,
                   parsed_function_calls: list[ParsedFunctionCall]) -> list[ParsedFunctionCall]:
    """Apply the tool fixer to parsed calls and debug-log the before and after."""
    fixed_tool_calls: list[ParsedFunctionCall] = []
    if not tool_fixer:
        return parsed_function_calls
    for tc in parsed_function_calls:
        arguments = tool_fixer(tc, user_context)
        if isinstance(arguments, list):
            fixed_tool_calls.extend(arguments)
        else:
            fixed_tool_calls.append(arguments)
    if log.isEnabledFor(logging.DEBUG):
        adapter = TypeAdapter(List[ParsedFunctionCall])
        parsed_str = adapter.dump_json(parsed_function_calls).decode("utf-8")
        fixed_str = adapter.dump_json(fixed_tool_calls).decode("utf-8")
        log.debug("tool calls: parsed=%s, fixed=%s", parsed_str, fixed_str)
    return fixed_tool_calls


def fix_tool_calls_and_convert_to_choice_delta_tool_calls(tool_fixer: ToolFixer | None,
                                                          user_context: UserContext | None,
                                                          parsed_function_calls: list[ParsedFunctionCall]
                                                          ) -> list[ChoiceDeltaToolCall]:
    """Fix parsed tool calls and convert them into OpenAI tool call chunks."""
    return list(map(to_openai_tool_call, fix_tool_calls(tool_fixer, user_context, parsed_function_calls)))


def fix_and_chat_complete_parsed_tool_calls(tool_fixer: ToolFixer | None,
                                            user_context: UserContext | None,
                                            parsed_function_calls: list[ParsedFunctionCall],
                                            tool_call_expression: str, role: Role | None
                                            ) -> tuple[ChatCompletionChunk, StopSignal]:
    """Fix parsed tool calls and wrap them into a chat completion chunk."""
    tool_calls = fix_tool_calls_and_convert_to_choice_delta_tool_calls(tool_fixer, user_context,
                                                                       parsed_function_calls)
    if len(tool_calls) == 0:
        log.info("unparsed tool calls: %s", tool_call_expression)
        return new_chat_completion_chunk(role=role, content=tool_call_expression), StopSignal.CANCEL
    return new_chat_completion_chunk(role=role, tool_calls=tool_calls), StopSignal.TOOL_CALL


class TokenProcessor:  # pylint: disable=too-many-instance-attributes  # keeps parser, phrase and timing state of one stream
    """Stateful processor turning decoded tokens into chat completion chunks and stop signals."""

    # The state of the processed stream; declared here so that the helpers below
    # can use it before the constructor assigns the values.
    phrase: Phrase
    tool_call_phrase: Phrase
    phrase_tick: float | None
    tool_call_parsing_tick: float | None
    tool_call_parsing_start_time: float | None

    def __clean_phrase(self) -> Phrase:
        """Drop the accumulated phrase and start a new one."""
        print_log(self.phrase)
        new_phrase = Phrase(is_detect_looped_inference=self.config.is_detect_looped_inference)
        self.phrase = new_phrase
        self.empty_conversation_counter = 0
        self.phrase_tick = None
        return new_phrase

    def __clean_tool_call_phrase(self) -> None:
        """Drop the accumulated tool call phrase and stop its parsing timers."""
        call_phrase = self.tool_call_phrase
        print_log(call_phrase)
        phrase = Phrase(is_detect_looped_inference=self.config.is_detect_looped_inference)
        self.tool_call_phrase = phrase
        self.tool_call_parsing_tick = None
        self.tool_call_parsing_start_time = None

    def __init__(self,  # pylint: disable=too-many-arguments,too-many-positional-arguments  # constructor of the stream processor, args mirror engine wiring
                 prompt: str,
                 parser: Parser,
                 init_chat_events: bool,
                 config: TokenHandlerConfig,
                 tool_fixer: ToolFixer | None,
                 user_context: UserContext | None = None,
                 supported_functions: dict[str, FunctionDefinitionParameters] | None = None):
        super().__init__()
        self.no_conversation_counter = 0
        self.create_time = datetime.now(timezone.utc)
        self.start_time: datetime | None = None
        self.user_context = user_context
        self.tool_fixer: ToolFixer | None = tool_fixer

        self.parser = parser
        state = parser.new_state(prompt, supported_functions, init_chat_events)
        self.state = state
        self.is_chat_mode = init_chat_events
        self.config = config
        self.prev_role: str | None = None
        self.token_conversation_start_number: int = -1
        self.expect_role = False
        self.empty_conversation_counter = 0
        self.stop_inference = False
        self.token_counter = 0
        self.role_initialized = False
        self.phrase_tick = None
        self.phrase = Phrase(is_detect_looped_inference=self.config.is_detect_looped_inference)
        self.tool_call_phrase = Phrase(is_detect_looped_inference=self.config.is_detect_looped_inference)
        self.tool_call_parsing_tick = None
        self.tool_call_parsing_start_time = None
        self.tool_call_parsing_long_time_warned: bool = False
        self.tool_call_parsing_max_time_warned: bool = False

    def tool_call_parsing_time(self, now_time: float) -> timedelta:
        """Time the current tool call has been parsed for, zero when it did not start yet."""
        start_time = self.tool_call_parsing_start_time
        if start_time is None:
            return timedelta(seconds=0)
        return timedelta(seconds=now_time - start_time)

    def get_stat_info(self) -> str | None:
        """Return generation statistics, or None before the first processed token."""
        stat_info = None
        start_time = self.start_time
        if start_time:
            finished_at = datetime.now(timezone.utc)
            time_delta = finished_at - start_time
            amount = self.token_counter
            total_seconds = time_delta.total_seconds()
            ttft = (start_time - self.create_time).total_seconds()
            stat_info = (
                f"generated {amount} token{end(amount)} in {time_delta} sec, "
                f"{amount / total_seconds} t/sec), ttft {ttft} sec"
            )
        return stat_info

    def process_tokens(self, decoded_tokens: list[str]) -> tuple[
        list[ChatCompletionChunk], StopSignal | None]:
        """Process a decoded token batch and return chunks with an optional stop signal."""
        if self.start_time is None:
            self.start_time = datetime.now(timezone.utc)
        state = self.state
        result: list[ChatCompletionChunk] = []
        try:
            for token in decoded_tokens:
                self.token_counter += 1
                log.debug("token '%s', num %s", token, self.token_counter)
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
        except Exception as e:  # pylint: disable=broad-exception-caught  # streaming barrier: any parser error must degrade to an error chunk, not kill the stream
            log.error("streamer error: %s", e, exc_info=e)
            result.append(new_stop_response(role=state.role, content=markdown_bold(f"StreamerError:{e}")))
            return result, StopSignal.CANCEL

        return result, None

    # Single dispatch over parser events; splitting would scatter the state transitions.
    def process_token(self, token: str, token_number: int) -> tuple[  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        list[ChatCompletionChunk], StopSignal | None]:
        """Process one decoded token and dispatch it by the current parser event."""
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
        elif (self.expect_role and current_event == StateEvent.CONVERSATION
              and token_number - self.token_conversation_start_number == 1):
            if len(token.rstrip()) > 0:  # conversation role
                self.set_role(token, state)
            else:
                log.debug("empty role for conversation start")
        elif parser.is_think_start(state, token):
            if current_event == StateEvent.TOOL_CALL:
                log.debug("tool call is finished by generated thinking tag: '%s'", token)
                tool_call, stop_signal = self.tool_call_end(state, token)
                result.append(tool_call)
            else:
                self.thinking_start(state)
        elif parser.is_think_end(state, token):
            self.thinking_end(state)
        elif parser.is_tool_call_start(state, token):
            if current_event == StateEvent.TOOL_CALL:
                log.debug("tool call is finished by starting new tool call: '%s'", token)
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
            log.debug("tool response start: %s", token)
        elif parser.is_tool_response_end(state, token):
            state.finish_current_event(StateEvent.TOOL_RESPONSE, log)
            log.debug("tool response end: %s", token)
        elif parser.is_fim_middle(state, token):
            state.start_event(StateEvent.FIM_MIDDLE)
        elif parser.is_end(state, token):
            log.debug("parsed end: %s", token)
            stop_signal = StopSignal.STOP
        elif current_event == StateEvent.TOOL_CALL:
            tool_call_loop_error: str | None = None
            tool_call_phrase = self.tool_call_phrase
            try:
                tool_call_phrase.add_token(token)
            except LoopError as e:
                log.error("tool call error (loop): %s", e)
                tool_call_loop_error = markdown_tool_call_loop_error(e.message + " (tool call)", e.payload)

            if tool_call_loop_error:
                tool_call, stop_signal = self.tool_call_end(state, token)
                result.append(new_chat_completion_chunk(role=state.role, content=tool_call_loop_error))
                result.append(tool_call)
            else:
                parsing_time = self.tool_call_parsing_time(now_time)
                if (not self.tool_call_parsing_long_time_warned
                        and parsing_time >= self.config.tool_call_parting_duration_warning):
                    formatted = format_time(self.config.tool_call_parting_duration_warning)
                    warning_msg = markdown_bold(f"WARNING: Long tool call calling({formatted})") + "\n"
                    result.append(new_chat_completion_chunk(role=state.role, content=warning_msg))
                    self.tool_call_parsing_long_time_warned = True
                elif (self.tool_call_parsing_max_time_warned
                      and parsing_time >= self.config.tool_call_parting_duration_limit):
                    formatted = format_time(self.config.tool_call_parting_duration_limit)
                    warning_msg = markdown_bold(f"WARNING: Tool call parsing exceeded time limit {formatted}.\n"
                                                ) + markdown_file_content(tool_call_phrase.full)
                    result.append(new_chat_completion_chunk(role=state.role, content=warning_msg))
                    self.tool_call_parsing_max_time_warned = True
                tool_call_parsing_tick = self.tool_call_parsing_tick
                tool_call_snapshot_time = now_time - tool_call_parsing_tick if tool_call_parsing_tick else 0
                if tool_call_snapshot_time >= 10:
                    self.tool_call_parsing_tick = now_time
                    log.debug("tool call part: %s", tool_call_phrase.full)
        else:
            phrase: Phrase = self.phrase
            handle_delay_end = False
            if state.is_streaming_delayed:
                log.debug("delayed phrase: %s", phrase.full)
                delay_end = parser.is_delay_streaming_end(state, token)
                if delay_end:
                    log.info("delay streaming finish: token '%s', token num '%s'", token, token_number)
                    handle_delay_end = True
                    state.is_streaming_delayed = False
            else:
                delay_start = parser.is_delay_streaming_start(state, token)
                if delay_start:
                    state.is_streaming_delayed = delay_start
                    log.info("delay streaming start: token '%s', token num '%s'", token, token_number)
                    # restart phrase
                    phrase = self.__clean_phrase()

            loop_error: str | None = None
            try:
                new_lines = phrase.add_token(token)
            except LoopError as e:
                new_lines = None
                log.error("loop error: %s", e)
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
                    log.info("%s phrase: '%s', last token num: %s", state.role, "".join(new_lines), token_number)

                if handle_delay_end:
                    delayed_result, stop_signal = self.handle_delayed(phrase)
                    result.extend(delayed_result)
                else:
                    stop_hit = token in self.config.stop_strings
                    if stop_hit:
                        # stop words are matched on the raw token stream in both chat and
                        # completion modes: the matched token ends the generation, is not
                        # emitted and does not reach the phrase content
                        log.debug("stop string '%s' reached, cut the generation", token)
                        stop_signal = StopSignal.STOP
                    if not self.is_chat_mode:
                        if not stop_hit:
                            result.append(new_chat_completion_chunk(role=state.role, content=token))

                    erase = stop_hit or current_event == StateEvent.TOOL_RESPONSE or parser.is_erase(state, token)
                    is_assistant = ROLE_ASSISTANT == state.role
                    if not is_assistant:
                        log.warning("unexpected role %s", state.role)
                    if is_assistant or not self.config.prevent_no_assistant_inference_output:
                        if erase:
                            log.debug("erase token: %s", token)
                        else:
                            result.append(new_chat_completion_chunk(role=state.role, content=token,
                                                                    thinking=thinking))
                    else:
                        log.warning("prevent generating by unexpected role %s, token '%s'", state.role, token)
        state.finalize(token)
        return result, stop_signal

    def handle_delayed(self, phrase: Phrase) -> tuple[list[ChatCompletionChunk], StopSignal | None]:
        """Emit the delayed phrase as content or finish it as a tool call."""
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
        """Apply the conversation role parsed from the token."""
        self.expect_role = False
        self.prev_role = state.role

        new_role: Role = ROLE_ASSISTANT if token == self.parser.get_assistant_role_name() \
            else cast(Role, token)
        state.role = new_role
        log.debug("set conversation role %s, prev %s", state.role, self.prev_role)

    def conversation_end(self, state: ParserState, token_number: int) -> tuple[
        list[ChatCompletionChunk], StopSignal | None]:
        """Finalize the conversation and flush the accumulated phrase."""
        result: list[ChatCompletionChunk] = []
        stop_signal: StopSignal | None = None
        if state.is_streaming_delayed:
            delayed_result, stop_signal = self.handle_delayed(self.phrase)
            result.extend(delayed_result)
            if not stop_signal:
                stop_signal = StopSignal.STOP
            state.is_streaming_delayed = False
        elif state.get_current_event() == StateEvent.TOOL_CALL:
            # sometimes Qwen3.5 ends tool call by end conversation token
            if log.isEnabledFor(logging.DEBUG):
                log.debug("tool call ended by end conversation token '%s'", self.tool_call_phrase.full)
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
                log.debug("empty conversation end, role %s", state.role)
                self.empty_conversation_counter += 1
            else:
                self.empty_conversation_counter = 0
                log.info("%s conversation end by: %s, last token num: %s", state.role, phrase, token_number)
                self.__clean_phrase()

            if self.empty_conversation_counter > self.config.empty_conversation_counter_max:
                warning_msg = f"many empty conversations ({self.empty_conversation_counter}), interrupt inference"
                log.warning(warning_msg)
                result.append(new_chat_completion_chunk(role=state.role, content=markdown_bold(warning_msg)))
                stop_signal = StopSignal.CANCEL
        state.finish_current_event(StateEvent.CONVERSATION)
        return result, stop_signal

    def conversation_start(self, state: ParserState, tail: str, token_number: int):
        """Open a conversation event and detect the role token."""
        state.start_event(StateEvent.CONVERSATION)
        self.no_conversation_counter = 0
        self.token_conversation_start_number = token_number

        phrase = self.phrase.full.rstrip()
        if len(phrase) > 0:
            log.info("phrase before conversation: '%s', last token num: %s", phrase, token_number)

        # check role in place
        if tail:
            self.set_role(tail, state)
        else:
            self.expect_role = True

    def handle_tool_call(self, state: ParserState) -> tuple[ChatCompletionChunk, StopSignal]:
        """Parse the accumulated tool call phrase and turn it into a chunk."""
        parser = self.parser
        tool_call_phrase = self.tool_call_phrase
        tool_call_expression = tool_call_phrase.full
        parsed_function_calls, _ = parser.parse_tool_calls(state, tool_call_expression)
        tool_calls = fix_tool_calls_and_convert_to_choice_delta_tool_calls(self.tool_fixer, self.user_context,
                                                                           parsed_function_calls)
        if len(tool_calls) == 0:
            log.info("unparsed tool calls: %s", tool_call_expression)
            chunk, stop_signal = new_chat_completion_chunk(role=state.role,
                                                           content=tool_call_expression), StopSignal.CANCEL
        else:
            chunk, stop_signal = new_chat_completion_chunk(role=state.role, tool_calls=tool_calls), StopSignal.TOOL_CALL
        self.__clean_tool_call_phrase()
        return chunk, stop_signal

    def tool_call_end(self, state: ParserState, token: str | None) -> tuple[ChatCompletionChunk, StopSignal]:
        """Finish the tool call event and return the resulting chunk."""
        self.tool_call_parsing_start_time = None
        tool_call_phrase = self.tool_call_phrase
        state.finish_current_event(expected_state=StateEvent.TOOL_CALL)
        if token:
            try:
                tool_call_phrase.add_token(token)
            except LoopError:
                log.warning("loop at the end of tool call %s", tool_call_phrase.full)
        log.debug("tool call end: %s", tool_call_phrase.full)

        return self.handle_tool_call(state)

    def tool_call_start(self, state: ParserState, token: str):
        """Start the tool call event and arm its parsing timers."""
        state.start_event(StateEvent.TOOL_CALL)
        log.debug("tool call start: %s", token)

        phrase = self.phrase.full.rstrip()
        if len(phrase) > 0:
            log.info("%s phrase before tool call: '%s'", state.role, phrase)
            self.__clean_phrase()

        now_time = now()
        self.tool_call_parsing_tick = now_time
        self.tool_call_parsing_start_time = now_time
        self.tool_call_phrase.add_token(token)

    def thinking_end(self, state: ParserState):
        """Close the thinking event, possibly starting tool call parsing."""
        now_time = now()
        state.finish_current_event(expected_state=StateEvent.THINK)
        if state.has_event(StateEvent.TOOL_CALL):
            self.tool_call_parsing_start_time = now_time
            log.warning("stop think token inside tool_call: '%s', phrase: '%s'",
                        self.tool_call_phrase.full, self.phrase.full)
        log.debug("thinking is over")

    def thinking_start(self, state: ParserState):
        """Open the thinking event, aborting pending tool call parsing."""
        state.start_event(StateEvent.THINK)
        if state.has_event(StateEvent.TOOL_CALL):
            self.tool_call_parsing_start_time = None
            log.warning("start think token inside tool_call %s", self.tool_call_phrase.full)
        log.debug("thinking is starting")
