import asyncio
import logging.config
import os
import queue
import sys
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Generator, List
from typing import SupportsInt, Optional

import openvino_genai as ov_genai
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from openvino_genai import Tokenizer
from openvino_genai.py_openvino_genai import ChatHistory, GenerationFinishReason
from openvino_genai.py_openvino_genai import StreamingStatus
from pydantic import TypeAdapter
from pydantic.json import pydantic_encoder

from agent.common.log import LoggingRoute, log_format_prefix, log_format_simple
from agent.common.metric_mem import get_current_memory
from agent.common.openai_model import OpenAIChatCompletionRequest, ChatCompletionMessageParam, \
    ToolCall, OpenAICompletionResponse, \
    ChatCompletionChoice, ChatCompletionMessage, ResponseFormat, CHAT_COMPLETION, CHAT_COMPLETION_CHUNK, \
    FunctionDefinition
from agent.common.roles import ROLE_ASSISTANT, ROLE_USER
from agent.parser.qwen3 import parse_tool_calls, is_conversation_start, is_conversation_end, think_is_started, \
    think_is_over, \
    is_possible_tool_call_start, is_possible_tool_call_end, is_prompt_start_thinking, is_partial_tool_call
from agent.preprocess.tool_call import PreprocessToolCall
from client.veai.tool_call_fixer import fix_incorrect_arguments

device_name = "GPU"

reasoning_supported = True
model_name = "OmniCoder-9B-int4-sym-g128"
model_path = f"../models/{model_name}/1"
model_cache_dir = f"../models_cache/{model_name}"

prevent_no_assistant_inference_output = True

default_max_new_tokens = 4096
default_max_tokens = 65536
default_temperature = 0.4
default_top_p = 0.95
default_top_k = 40
default_min_p = 0.05
default_repetition_penalty = 1.1

scheduler_config = ov_genai.SchedulerConfig()
scheduler_config.enable_prefix_caching = True
scheduler_config.max_num_batched_tokens = 256
scheduler_config.max_num_seqs = 1
scheduler_config.cache_interval_multiplier = None  # 2
scheduler_config.dynamic_split_fuse = True
scheduler_config.use_sparse_attention = False

scheduler_config.cache_size = 8
scheduler_config.use_cache_eviction = True

config = {
    "CACHE_DIR": model_cache_dir,
    # "GPU_ENABLE_LARGE_ALLOCATIONS": "YES",
    # "KV_CACHE_PRECISION": "u4",
    "PERFORMANCE_HINT": "LATENCY",  # THROUGHPUT crashes process
    # "scheduler_config": scheduler_config,
    "ATTENTION_BACKEND": "PA",
    # "ATTENTION_BACKEND": "SDPA",
}

# os.environ["LOG_LEVEL"] = "4"
os.environ["OPENVINO_LOG_LEVEL"] = "4"
os.environ["ONEDNN_VERBOSE"] = "ON"
os.environ["ONEDNN_VERBOSE_TIMESTAMP"] = "1"

log = logging.getLogger(__name__)
log_inference = logging.getLogger("inference")
log_inference_prompt = logging.getLogger("inference.prompt")

log.info(f"model loading {model_path}, device: {device_name}, scheduler_config: {scheduler_config.to_string()}")

start_mem = get_current_memory()
log.debug(f"consumed memory: {start_mem:.2f} MB")

try:
    pipe = ov_genai.VLMPipeline(models_path=model_path, device=device_name, **config)
    # pipe = ov_genai.ContinuousBatchingPipeline(models_path=model_path, device=device_name,
    #                                            scheduler_config=scheduler_config,
    #                                            **config)

    log.info(f"model loaded successfully")

    loaded_pipe_mem = get_current_memory()
    pipe_cost = loaded_pipe_mem - start_mem

    log.debug(f"consumed memory: {loaded_pipe_mem:.2f} MB, pipe loading delta: {pipe_cost:.2f} MB")
except Exception as e:
    log.error(f"instantiate pipeline error: {e}", exc_info=e)
    sys.exit(1)

app = FastAPI()
app.router.route_class = LoggingRoute
executor = ThreadPoolExecutor()


@app.post("/v1/chat/completions")
async def chat(body: OpenAIChatCompletionRequest, request: Request):
    loop = asyncio.get_event_loop()

    def is_disconnected() -> bool:
        disconnected = False
        try:
            disconnected = asyncio.run_coroutine_threadsafe(request.is_disconnected(), loop).result(0.5)
            if disconnected:
                log.debug(f"disconnected http request")
        except asyncio.TimeoutError:
            pass
            # log.debug(f"disconnected http request check timeout")
        return disconnected

    is_reasoning_enabled: bool = reasoning_supported and (body.model_config.get("reasoning") or True)

    body.response_format = ResponseFormat(type="json_object")
    messages = body.messages
    log.info(f"inbound history messages {len(messages)}")

    preprocess_tool_call = PreprocessToolCall()
    looped_function = preprocess_tool_call.check_loop_calls(messages)
    if looped_function:
        # log
        result = ChatCompletionMessage(role=ROLE_ASSISTANT,
                                       content=(f"Multiple calls of the '{looped_function.name}' tool "
                                                f"result in the same response '{looped_function.result}'. "
                                                f"Generation is interrupted."))
        object = CHAT_COMPLETION_CHUNK if body.stream else CHAT_COMPLETION
        response = OpenAICompletionResponse(object=object, id=str(uuid.uuid4()), created=int(time.time()),
                                            model=model_name, choices=[
                ChatCompletionChoice(delta=(result if body.stream else None),
                                     message=(result if not body.stream else None))])
        return response

    tools = body.tools
    function_by_name: dict[str, FunctionDefinition] = {}
    tools_raw: list[dict[str, Any]] = []
    for tool in (tools or []):
        tools_raw.append(tool.model_dump())
        function = tool.function
        function_by_name[function.name] = function

    chat_history = ChatHistory()
    chat_history_messages = list(map(ChatCompletionMessageParam.model_dump, messages)) if messages else []
    chat_history.set_messages(chat_history_messages)
    chat_history.set_tools(tools_raw)

    tokenizer: Tokenizer = pipe.get_tokenizer()
    extra_context = {}
    if reasoning_supported:
        extra_context["enable_thinking"] = is_reasoning_enabled

    full_prompt = tokenizer.apply_chat_template(history=chat_history,
                                                add_generation_prompt=True,
                                                tools=tools_raw,
                                                extra_context=extra_context)

    log_inference_prompt.debug(full_prompt)

    generation_config = ov_genai.GenerationConfig()
    generation_config.max_new_tokens = body.max_completion_tokens or default_max_new_tokens
    generation_config.max_length = body.max_tokens or default_max_tokens
    generation_config.apply_chat_template = False if full_prompt else True

    temp = body.temperature or default_temperature
    if temp < 0.05:
        # Greedy Search
        generation_config.do_sample = False
    else:
        generation_config.do_sample = True
        generation_config.temperature = temp
        generation_config.top_p = body.top_p or default_top_p
        generation_config.top_k = default_top_k
        generation_config.min_p = default_min_p

        if body.frequency_penalty:
            generation_config.frequency_penalty = body.frequency_penalty

        if body.logprobs:
            generation_config.logprobs = 1

    generation_config.repetition_penalty = default_repetition_penalty

    def stream_generator() -> Generator[str, None, None]:
        for chunk in chunk_generator():
            yield f"data: {chunk.model_dump_json()}\n\n"

    def chunk_generator() -> Generator[OpenAICompletionResponse, None, None]:
        chunk_queue: queue.Queue[OpenAICompletionResponse | None] = queue.Queue()
        stop_stream_handling: queue.Queue[bool] = queue.Queue()
        start_stream_handling: queue.Queue[bool] = queue.Queue()

        def run_inference():
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

                def __init__(self, start_thinking: bool = True):
                    super().__init__()
                    self.log = logging.getLogger("inference.stream")
                    self.role = ROLE_ASSISTANT
                    self.prev_role = None
                    self.possible_tool_call_expression = ""
                    self.possible_call_in_progress = False
                    self.tool_call_count = 0
                    self.token_conversation_start_number: int = -1
                    self.expect_role = False
                    self.user_phrase_generated = False
                    self.phrase_tick: float | None = None
                    self.possible_tool_call_expression_tick: float | None = None
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

                    if start_thinking > 0:
                        self.thinking_progress_counter = 1
                        self.states.append(State.THINK)

                def end(self) -> None:
                    self.full_generated = ""
                    self.log.debug("stream end")

                def write(self, tokens: Sequence[SupportsInt]) -> StreamingStatus:
                    log = self.log

                    if not self.started:
                        start_stream_handling.put(True)
                        self.started = True

                    decoded_tokens: list[str] = tokenizer.decode(tokens=[tokens], skip_special_tokens=False)

                    if is_disconnected():
                        log.info("stream finished by user disconnected")
                        return StreamingStatus.STOP

                    if not stop_stream_handling.empty() and stop_stream_handling.get_nowait():
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

                    def handle_possible_tool_call() -> OpenAICompletionResponse:
                        parsed_tool_calls, partial = parse_tool_calls(self.possible_tool_call_expression,
                                                                      function_by_name)
                        if not parsed_tool_calls:
                            log.info(
                                f"phrase like tool calls: {self.possible_tool_call_expression}")
                            chunk = new_chunk_response(role=self.role, content=self.possible_tool_call_expression)
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

                        self.possible_call_in_progress = False
                        self.possible_tool_call_expression = ""
                        self.possible_tool_call_expression_tick = None
                        return chunk

                    if is_conversation_start(token):
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

                    elif is_conversation_end(token):
                        self.in_conversation = False
                        if self.__get_last_state() == State.TOOL_CALL:
                            # sometimes Qwen3.5 ends tool call by end conversation token
                            if log.isEnabledFor(logging.DEBUG):
                                log.debug(
                                    f"tool call ended by end conversation token '{self.possible_tool_call_expression}'")
                            else:
                                log.info("tool call ended by end conversation token")
                            self.__remove_state(expected_state=State.TOOL_CALL)
                            if is_partial_tool_call(self.possible_tool_call_expression):
                                log.info(
                                    f"partial tool call ended by end conversation token '{self.possible_tool_call_expression}'")
                            else:
                                chunk = handle_possible_tool_call()
                                if chunk:
                                    chunk_queue.put(chunk)
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
                                f"many empty conversations ({self.empty_conversation_counter}), abort inference")
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

                    elif think_is_started(token):
                        self.states.append(State.THINK)
                        if len(self.possible_tool_call_expression) > 0:
                            log.warning(f"start think token inside tool_call {self.possible_tool_call_expression}")
                        self.thinking_progress_counter += 1
                        if self.thinking_progress_counter == 1:
                            log.debug("thinking is starting")
                        else:
                            log.debug(
                                f"More thinking for God of thinking!!! {self.thinking_progress_counter}")
                    elif think_is_over(token):
                        self.__remove_state(expected_state=State.THINK)
                        if len(self.possible_tool_call_expression) > 0:
                            log.warning(
                                f"stop think token inside tool_call: '{self.possible_tool_call_expression}', phrase: '{self.phrase}'")

                        if self.thinking_progress_counter > 0:
                            self.thinking_progress_counter -= 1
                            if self.thinking_progress_counter == 0:
                                log.debug("thinking is over")
                            else:
                                log.debug(
                                    f"intensity of thinking decreased {self.thinking_progress_counter}")
                    elif is_possible_tool_call_start(token):
                        self.states.append(State.TOOL_CALL)
                        log.debug(f"possible tool call start: {token}")

                        phrase = self.phrase.rstrip()
                        if len(phrase) > 0:
                            log.info(f"{self.role} phrase before tool call: '{phrase}'")

                        self.__clean_phrase()

                        self.possible_tool_call_expression_tick = time.time()
                        self.possible_call_in_progress = True
                        self.possible_tool_call_expression += token
                    elif is_possible_tool_call_end(token):
                        self.__remove_state(expected_state=State.TOOL_CALL)
                        log.debug(f"possible tool call end: {token}")
                        self.possible_tool_call_expression += token
                        chunk = handle_possible_tool_call()
                        chunk_queue.put(chunk)
                    else:
                        last_state = self.__get_last_state()
                        if self.possible_call_in_progress and last_state == State.TOOL_CALL:
                            self.possible_tool_call_expression += token
                            now_time = time.time()
                            possible_tool_call_expression_time = now_time - self.possible_tool_call_expression_tick
                            if possible_tool_call_expression_time >= 10:
                                self.possible_tool_call_expression_tick = now_time
                                log.info(
                                    f"possible tool call part: {self.possible_tool_call_expression}")
                                words = self.possible_tool_call_expression.split(" ")
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

                            now_time = time.time()
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
                                    if is_assistant or not prevent_no_assistant_inference_output:
                                        chunk = new_chunk_response(role=self.role, content=token,
                                                                   thinking=self.thinking_progress_counter > 0)
                                        chunk_queue.put(chunk)
                                    else:
                                        log.warning(
                                            f"prevent generating by unexpected role {self.role}, token '{token}'")
                                else:
                                    log.warning(f"generated out of conversation: '{self.phrase.rstrip()}'")
                    return None

            try:
                pipe.start_chat()
                if log_inference.isEnabledFor(logging.DEBUG):
                    log_inference.debug(
                        f"inference starting with parameters max_length={generation_config.max_length}, "
                        f"max_new_tokens={generation_config.max_new_tokens}, "
                        f"do_sample={generation_config.do_sample}, temperature={generation_config.temperature:.2f}, "
                        f"top_p={generation_config.top_p:.2f}, top_k={generation_config.top_k}, "
                        f"min_p={generation_config.min_p:.2f}, repetition_penalty={generation_config.repetition_penalty:.2f}, "
                        f"presence_penalty={generation_config.presence_penalty:.2f}, "
                        f"frequency_penalty={generation_config.frequency_penalty:.2f}")
                else:
                    log_inference.info(f"inference starting")

                before_generate_mem = get_current_memory()
                streamer = Streamer(start_thinking=is_prompt_start_thinking(full_prompt))
                generate_result = pipe.generate(prompt=full_prompt, generation_config=generation_config,
                                                streamer=streamer)
                chunk_queue.put_nowait(None)
                after_generate_mem = get_current_memory()
                generate_cost = after_generate_mem - before_generate_mem
                log.debug(f"consumed memory: {after_generate_mem:.2f} MB, generate delta: {generate_cost:.2f} MB")

                inference_finish_reasons = generate_result.finish_reasons
                if log_inference.isEnabledFor(logging.DEBUG):
                    log_inference.debug(f"inference finished, "
                                        f"reason={inference_finish_reasons}, "
                                        f"result={generate_result.texts}")
                else:
                    log_inference.info(f"inference finished")
                inference_finish_reason = inference_finish_reasons[0] if inference_finish_reasons else None
                if inference_finish_reason is None or inference_finish_reason == GenerationFinishReason.NONE:
                    log.warning(f"inference finished by unexpected status {inference_finish_reason}")

            except Exception as e:
                log_inference.error(f"inference error: {e}", exc_info=e)
                raise
            finally:
                pipe.finish_chat()

        unique_id = str(uuid.uuid4())
        inference_task = executor.submit(run_inference)
        try:
            stream_started = start_stream_handling.get()
            stop_inference = False
            while not stop_inference:
                if is_disconnected():
                    break
                try:
                    chunk = chunk_queue.get(timeout=20)
                    if chunk:
                        chunk.id = unique_id
                        yield chunk
                    else:
                        stop_inference = True
                except queue.Empty:
                    pass
                except TimeoutError:
                    pass

        except Exception as e:
            log.error(f"chunk processing error: {e}", exc_info=e)
        finally:
            stop_stream_handling.put_nowait(True)
            if not inference_task.done():
                log.info("waiting for inference to complete")
                try:
                    r = inference_task.result(timeout=20)
                except Exception as e:
                    log.error(f"waiting inference completion error: {e}", exc_info=e)
            log.info("inference handling is done")

    if body.stream:
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        full_content = ""
        full_reasoning_content = ""
        full_tool_calls: list[ToolCall] = []
        finish_reason = "stop"

        for chunk_data in chunk_generator():
            choices = chunk_data.choices
            if choices:
                finish_reason = choices[-1].finish_reason or finish_reason
                for choice in choices:
                    delta = choice.delta
                    delta_content = delta.content
                    if delta_content:
                        full_content += delta_content
                    delta_reasoning_content = delta.reasoning_content
                    if delta_reasoning_content:
                        full_reasoning_content += delta_reasoning_content
                    delta_tool_calls = delta.tool_calls
                    if delta_tool_calls:
                        full_tool_calls += delta_tool_calls

        message = ChatCompletionMessage()
        message.role = ROLE_ASSISTANT
        message.content = full_content
        message.reasoning_content = full_reasoning_content
        message.tool_calls = full_tool_calls
        return OpenAICompletionResponse(object=CHAT_COMPLETION, id=str(uuid.uuid4()), created=int(time.time()),
                                        model=model_name, choices=[
                ChatCompletionChoice(index=0, finish_reason=finish_reason, message=message)])


def new_chunk_response(role: str, content: str | None = None, thinking: bool = False,
                       tool_calls: Optional[List[ToolCall]] = None,
                       finish_reason: str | None = None) -> OpenAICompletionResponse:
    delta = ChatCompletionMessage()
    delta.role = role
    if thinking:
        delta.reasoning_content = content
    else:
        delta.content = content

    if tool_calls:
        delta.tool_calls = tool_calls

    choices = [new_choice(delta=delta, finish_reason=finish_reason)]
    return OpenAICompletionResponse(model=model_name, created=int(time.time()), choices=choices)


def new_choice(delta: ChatCompletionMessage, finish_reason: str | None = None) -> ChatCompletionChoice:
    return ChatCompletionChoice(index=0, finish_reason=finish_reason, delta=delta)


if __name__ == "__main__":
    log.info("server starting")
    log_config = uvicorn.config.LOGGING_CONFIG

    log_config["formatters"]["default"]["format"] = log_format_simple
    log_config["formatters"]["access"]["format"] = (
            log_format_prefix + " - %(client_addr)s - '%(request_line)s' %(status_code)s"
    )

    uvicorn.run(app, host="127.0.0.1", port=8888)
