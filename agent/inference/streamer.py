import logging
import time
from datetime import timedelta
from enum import Enum
from queue import Queue
from typing import Sequence, SupportsInt, Callable, List, Any

import openvino_genai as ov_genai
from openvino_genai.py_openvino_genai import StreamingStatus, Tokenizer
from pydantic import TypeAdapter, BaseModel

from common.openai_api import new_chunk_response
from common.openai_model import OpenAICompletionResponse, ToolCall, FunctionDefinition
from common.roles import ROLE_ASSISTANT, ROLE_USER
from common.time import format_time
from parser.base import Parser
from veai.tool_call_fixer import fix_incorrect_arguments


class StreamerConfig(BaseModel):
    tool_call_parting_duration_warning: timedelta = timedelta(minutes=2)
    tool_call_parting_duration_limit: timedelta = timedelta(minutes=10)
    prevent_no_assistant_inference_output: bool = True


class State(Enum):
    CONVERSATION = 1
    THINK = 2
    TOOL_CALL = 3


class Streamer(ov_genai.StreamerBase):
    def __clean_phrase(self):
        self.phrase = ""
        self.phrase_tick = None

    def __get_last_state(self):
        return self.states[-1] if self.states else None

    def __remove_state(self, expected_state: State):
        s = self.__get_last_state()
        if s == expected_state:
            self.states.pop()
        else:
            self.log.error(f"unexpected state {s}, expected {expected_state}")

    def __init__(self, tokenizer: Tokenizer, parser: Parser, chunk_queue: Queue[OpenAICompletionResponse | None],
                 stop_stream_handling: Queue[bool], start_stream_handling: Queue[bool],
                 is_disconnected: Callable[[], bool], supported_functions: dict[str, FunctionDefinition],
                 config: StreamerConfig, start_thinking: bool = True):
        super().__init__()
        self.log = logging.getLogger("inference.stream")
        self.tokenizer = tokenizer
        self.parser = parser
        self.chunk_queue = chunk_queue
        self.is_disconnected = is_disconnected
        self.supported_functions = supported_functions
        self.config = config
        self.stop_stream_handling = stop_stream_handling
        self.start_stream_handling = start_stream_handling
        self.role = ROLE_ASSISTANT
        self.prev_role = None
        self.tool_call_expression = ""
        self.tool_call_parsing_in_progress = False
        self.tool_call_count = 0
        self.token_conversation_start_number: int = -1
        self.expect_role = False
        self.user_phrase_generated = False
        self.phrase_tick: float | None = None
        self.tool_call_parsing_tick: float | None = None
        self.tool_call_parsing_start_time: float | None = None
        self.tool_call_parsing_long_time_warned: bool = False
        self.tool_call_parsing_max_time_warned: bool = False
        self.phrase = ""
        self.full_generated = ""
        self.empty_conversation_counter = 0
        self.no_conversation_counter = 0
        self.stop_inference = False
        self.token_counter = 0
        self.started = False

        # by default conversation is opened by assistant in chat template
        self.states: list[State] = [State.CONVERSATION]
        self.in_conversation = True
        self.thinking_progress_counter = 0
        if start_thinking > 0:
            self.states.append(State.THINK)
            self.thinking_progress_counter += 1

    def end(self) -> None:
        self.full_generated = ""
        self.log.debug("stream end")

    def write(self, tokens: Sequence[SupportsInt]) -> StreamingStatus:
        log = self.log

        if not self.started:
            self.start_stream_handling.put(True)
            self.started = True

        decoded_tokens: list[str] = self.tokenizer.decode(tokens=[tokens], skip_special_tokens=False)

        if self.is_disconnected():
            log.info("stream finished by user disconnected")
            return StreamingStatus.STOP

        if not self.stop_stream_handling.empty() and self.stop_stream_handling.get_nowait():
            log.debug("stream finished by stop signal")
            return StreamingStatus.STOP

        try:
            for t in decoded_tokens:
                self.token_counter += 1
                log.debug(f"token '{t}', num {self.token_counter}")
                self.full_generated += t
                stream_status = self.process_token(t, self.token_counter)
                if not (stream_status == StreamingStatus.RUNNING or stream_status is None):
                    # log
                    return stream_status
        except Exception as e:
            log.error(f"streamer error: {e}", exc_info=e)
            return StreamingStatus.CANCEL

        return StreamingStatus.RUNNING

    def process_token(self, token: str, token_number: int) -> StreamingStatus | None:
        log = self.log

        def handle_tool_call() -> OpenAICompletionResponse:
            parsed_tool_calls, partial = self.parser.parse_tool_calls(self.tool_call_expression,
                                                                      self.supported_functions)
            if not parsed_tool_calls:
                log.info(
                    f"phrase like tool calls: {self.tool_call_expression}")
                chunk = new_chunk_response(role=self.role, content=self.tool_call_expression)
            else:
                if partial:
                    pass
                fixed_tool_calls = list(map(fix_incorrect_arguments, parsed_tool_calls))
                if log.isEnabledFor(logging.INFO):
                    adapter = TypeAdapter(List[ToolCall])
                    log.info(
                        f"tool call: {adapter.dump_json(fixed_tool_calls).decode("utf-8")}")

                self.tool_call_count += 1
                chunk = new_chunk_response(role=self.role, tool_calls=fixed_tool_calls)

            self.tool_call_parsing_in_progress = False
            self.tool_call_expression = ""
            self.tool_call_parsing_tick = None
            return chunk

        now_time = time.perf_counter()
        if self.parser.is_conversation_start(token):
            self.states.append(State.CONVERSATION)
            self.in_conversation = True
            self.no_conversation_counter = 0
            self.token_conversation_start_number = token_number
            self.expect_role = True

            phrase = self.phrase.rstrip()
            if len(phrase) > 0:
                log.info(
                    f"phrase before conversation: '{phrase}', last token num: {token_number}")

            self.__clean_phrase()

        elif self.parser.is_conversation_end(token):
            self.in_conversation = False
            if self.__get_last_state() == State.TOOL_CALL:
                # sometimes Qwen3.5 ends tool call by end conversation token
                if log.isEnabledFor(logging.DEBUG):
                    log.debug(
                        f"tool call ended by end conversation token '{self.tool_call_expression}'")
                else:
                    log.info("tool call ended by end conversation token")
                self.__remove_state(expected_state=State.TOOL_CALL)

                self.chunk_queue.put(handle_tool_call())
            else:
                self.__remove_state(expected_state=State.CONVERSATION)

            self.token_conversation_start_number = -1
            self.expect_role = False

            phrase = self.phrase.rstrip()
            if len(phrase) == 0:
                log.debug(f"empty conversation end, role {self.role}")
                self.empty_conversation_counter += 1
            else:
                self.empty_conversation_counter = 0
                log.info(
                    f"{self.role} conversation end by: {phrase}, last token num: {token_number}")
                self.__clean_phrase()

            if self.empty_conversation_counter > 20:
                log.warning(
                    f"many empty conversations ({self.empty_conversation_counter}), interrupt inference")
                return StreamingStatus.CANCEL

            if self.tool_call_count > 0:
                log.debug(
                    f"stop inference by ending conversation with tool calling (count {self.tool_call_count})")
                return StreamingStatus.TOOL_CALL_STOP
            return None
        elif self.expect_role and self.in_conversation and token_number - self.token_conversation_start_number == 1:
            if len(token.rstrip()) > 0:  # conversation role
                self.expect_role = False
                self.prev_role = self.role
                self.role = token
                if self.role == ROLE_USER:
                    self.user_phrase_generated = True
                log.debug(f"set conversation role {self.role}, prev {self.prev_role}")
            else:
                log.debug("empty role for conversation start")

        elif self.parser.think_is_started(token):
            self.states.append(State.THINK)
            if self.tool_call_parsing_in_progress:
                self.tool_call_parsing_start_time = None
                log.warning(f"start think token inside tool_call {self.tool_call_expression}")
            self.thinking_progress_counter += 1
            if self.thinking_progress_counter == 1:
                log.debug("thinking is starting")
            else:
                log.debug(
                    f"More thinking for God of thinking!!! {self.thinking_progress_counter}")
        elif self.parser.think_is_over(token):
            self.__remove_state(expected_state=State.THINK)
            if self.tool_call_parsing_in_progress:
                self.tool_call_parsing_start_time = now_time
                log.warning(
                    f"stop think token inside tool_call: '{self.tool_call_expression}', phrase: '{self.phrase}'")

            if self.thinking_progress_counter > 0:
                self.thinking_progress_counter -= 1
                if self.thinking_progress_counter == 0:
                    log.debug("thinking is over")
                else:
                    log.debug(
                        f"intensity of thinking decreased {self.thinking_progress_counter}")
        elif self.parser.is_tool_call_start(token):
            self.states.append(State.TOOL_CALL)
            log.debug(f"tool call start: {token}")

            phrase = self.phrase.rstrip()
            if len(phrase) > 0:
                log.info(f"{self.role} phrase before tool call: '{phrase}'")

            self.__clean_phrase()

            self.tool_call_parsing_tick = now_time
            self.tool_call_parsing_start_time = now_time
            self.tool_call_parsing_in_progress = True
            self.tool_call_expression += token
        elif self.parser.is_tool_call_end(token):
            self.tool_call_parsing_start_time = None
            self.__remove_state(expected_state=State.TOOL_CALL)
            log.debug(f"tool call end: {token}")
            self.tool_call_expression += token
            self.chunk_queue.put(handle_tool_call())
        else:
            last_state = self.__get_last_state()
            if self.tool_call_parsing_in_progress and last_state == State.TOOL_CALL:
                self.tool_call_expression += token
                parsing_time = timedelta(seconds=(now_time - self.tool_call_parsing_start_time))
                if not self.tool_call_parsing_long_time_warned and parsing_time >= self.config.tool_call_parting_duration_warning:
                    chunk = new_chunk_response(role=ROLE_ASSISTANT, content=f"Long parsing of tool call "
                                                                            f"({format_time(self.config.tool_call_parting_duration_warning)})")
                    self.chunk_queue.put(chunk)
                    self.tool_call_parsing_long_time_warned = True
                elif self.tool_call_parsing_max_time_warned and parsing_time >= self.config.tool_call_parting_duration_limit:
                    chunk = new_chunk_response(role=ROLE_ASSISTANT, content=f"Tool call parsing exceeded time limit"
                                                                            f" {format_time(self.config.tool_call_parting_duration_limit)}.\n"
                                                                            f"```\n{self.tool_call_expression}\n```")
                    self.chunk_queue.put(chunk)
                    self.tool_call_parsing_max_time_warned = True

                tool_call_snapshot_time = now_time - self.tool_call_parsing_tick
                if tool_call_snapshot_time >= 10:
                    self.tool_call_parsing_tick = now_time
                    log.info(f"tool call part: {self.tool_call_expression}")
                    words = self.tool_call_expression.split(" ")
                    word_dict: dict[str, Any] = {}
                    for i, word in enumerate(words):
                        word_stat: dict[str, Any] = word_dict.get(word, {})
                        count = word_stat.get("count", 0) + 1
                        word_stat["count"] = count
                        position = word_stat.get("position", [])
                        position.append(i)
                        word_stat["position"] = position
            else:
                is_assistant = ROLE_ASSISTANT == self.role
                if not is_assistant:
                    log.warning(f"unexpected role {self.role}")

                self.phrase = self.phrase + token

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
                        log.info(
                            f"{self.role} phrase: '{phrase}', last token num: {token_number}")
                        self.__clean_phrase()

                if not last_state:
                    log.debug("no more conversations")
                    self.no_conversation_counter += 1
                    if self.no_conversation_counter > 5:
                        log.debug(
                            f"empty conversations limits exceed ({self.no_conversation_counter}), abort inference")
                        return StreamingStatus.STOP
                else:
                    if self.in_conversation:
                        if is_assistant or not self.config.prevent_no_assistant_inference_output:
                            chunk = new_chunk_response(role=self.role, content=token,
                                                       thinking=self.thinking_progress_counter > 0)
                            self.chunk_queue.put(chunk)
                        else:
                            log.warning(
                                f"prevent generating by unexpected role {self.role}, token '{token}'")
                    else:
                        phrase_rstrip = self.phrase.rstrip()
                        log_msg = f"generated out of conversation: '{phrase_rstrip}'"
                        if len(phrase_rstrip) > 0:
                            log.warning(log_msg)
                        else:
                            log.debug(log_msg)
        return None
