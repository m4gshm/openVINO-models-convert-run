# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi>=0.136.3",
#     "openvino-genai>=2026.2.0.0",
#     "pydantic>=2.13.4",
#     "typing>=3.10.0.0",
# ]
# ///
import time
import typing
import uuid
from collections.abc import Sequence
from enum import Enum

from openvino_genai import Tokenizer
from openvino_genai.py_openvino_genai import StreamingStatus
from pydantic import TypeAdapter

from commom_metric_mem import get_current_memory
from common_openapi_model import OpenAIChatCompletionRequest, ToolDefinition, ChatCompletionMessageParam, \
    OpenAICompletionChunkResponse, \
    ChatCompletionChunkChoice, ChatCompletionChunkDelta, ToolCall, FunctionCall, OpenAICompletionResponse, \
    ChatCompletionChoice, ChatCompletionMessage, ResponseFormat

model_name = "OmniCoder-9B-int4-sym-g128"
model_path = f"./models/{model_name}/1"

reasoning_supported = True

ROLE_ASSISTANT = "assistant"
ROLE_USER = "user"
ROLES: set[str] = {ROLE_ASSISTANT, ROLE_USER}

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
REASONING_START = "<think>"
REASONING_END = "</think>"

CONVERSATION_START = "<|im_start|>"
CONVERSATION_END = "<|im_end|>"

ERASED_TOKENS: set[str] = {TOOL_CALL_START, TOOL_CALL_END, REASONING_START, REASONING_END, CONVERSATION_START,
                           CONVERSATION_END} | ROLES

device_name = "GPU"
model_cache_dir = f"./models_cache/{model_name}"

import openvino_genai as ov_genai

default_temperature = 0.4
default_top_p = 0.95
default_top_k = 40
default_min_p = 0.05
default_repetition_penalty = 1.0
default_max_new_tokens = 65536

scheduler_config = ov_genai.SchedulerConfig()
scheduler_config.enable_prefix_caching = True
scheduler_config.max_num_batched_tokens = 256
# scheduler_config.num_kv_blocks = 4096
scheduler_config.max_num_seqs = 1
scheduler_config.cache_interval_multiplier = None  # 2
scheduler_config.dynamic_split_fuse = True
scheduler_config.use_sparse_attention = False

# scheduler_config.cache_size = 8
# scheduler_config.use_cache_eviction = True

config = {
    "CACHE_DIR": model_cache_dir,
    # "GPU_ENABLE_LARGE_ALLOCATIONS": "YES",
    # "KV_CACHE_PRECISION": "u4",
    "PERFORMANCE_HINT": "LATENCY",  # THROUGHPUT crashes process
    "scheduler_config": scheduler_config,
    "ATTENTION_BACKEND": "PA",
    # "ATTENTION_BACKEND": "SDPA",
}

import asyncio
import json
import logging.config
import os
import queue
import re
import sys
import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any, Generator, List

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from openvino_genai.py_openvino_genai import ChatHistory, VLMDecodedResults
from pydantic.json import pydantic_encoder

from common_log import log_format_simple, log_format_prefix, LoggingRoute

executor = ThreadPoolExecutor()
tool_call_counter = itertools.count(start=0)

# os.environ["LOG_LEVEL"] = "4"
os.environ["OPENVINO_LOG_LEVEL"] = "4"
os.environ["ONEDNN_VERBOSE"] = "ON"
os.environ["ONEDNN_VERBOSE_TIMESTAMP"] = "1"

log = logging.getLogger(__name__)
log_inference_prompt = logging.getLogger("inference.prompt")
log_stream = logging.getLogger("inference.stream")

app = FastAPI()
app.router.route_class = LoggingRoute

log.info(f"model loading {model_path}, device: {device_name}, scheduler_config: {scheduler_config.to_string()}")

start_mem = get_current_memory()
log.debug(f"consumed memory: {start_mem:.2f} MB")

try:
    pipe = ov_genai.VLMPipeline(models_path=model_path, device=device_name, **config)
    # pipe = ov_genai.ContinuousBatchingPipeline(models_path=model_path, device=device_name, **config)
    default_generation_config = pipe.get_generation_config()
    log.info(f"model loaded successfully")

    loaded_pipe_mem = get_current_memory()
    pipe_cost = loaded_pipe_mem - start_mem

    log.debug(f"consumed memory: {loaded_pipe_mem:.2f} MB, pipe loading cost: {pipe_cost:.2f} MB")
except Exception as e:
    log.error(f"instantiate pipeline error: {e}", exc_info=e)
    sys.exit(1)


def is_erase(token: str) -> bool:
    service_token: bool = token.strip() in ERASED_TOKENS
    return service_token


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
            log.debug(f"disconnected http request check timeout")
        return disconnected

    is_reasoning_enabled: bool = reasoning_supported and (body.model_config.get("reasoning") or True)

    body.response_format = ResponseFormat(type="json_object")
    messages = body.messages

    log.info(f"inbound history messages {len(messages)}")

    # for message in messages:
    #     content = message.get("content", {})
    #     role = content.get("role", None)
    #     tool_calls = content.get("tool_calls", None)
    #     if cast(list[dict[str, Any]], tool_calls):
    #         for tool_call in tool_calls:
    #             tool_choice = tool_call.get("type", None)

    messages: list[dict[str, Any]] = list(
        map(ChatCompletionMessageParam.model_dump, body.messages)) if body.tools else []

    chat_history = ChatHistory()
    chat_history_messages = messages
    chat_history.set_messages(chat_history_messages)

    tools: list[dict[str, Any]] = list(map(ToolDefinition.model_dump, body.tools)) if body.tools else []
    if tools:
        chat_history.set_tools(tools)

    tokenizer: Tokenizer = pipe.get_tokenizer()
    extra_context = {}
    if reasoning_supported:
        extra_context["enable_thinking"] = is_reasoning_enabled

    full_prompt = tokenizer.apply_chat_template(history=chat_history,
                                                add_generation_prompt=True,
                                                tools=tools,
                                                extra_context=extra_context)

    log_inference_prompt.debug(full_prompt)

    generation_config = ov_genai.GenerationConfig()
    generation_config.max_new_tokens = body.max_tokens or default_max_new_tokens
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

    generation_config.repetition_penalty = default_repetition_penalty

    # generation_config.parsers = {ReasoningParser()}
    # generation_config.include_stop_str_in_output = True
    # generation_config.adapters = AdapterConfig()

    # generation_config.include_stop_str_in_output = True
    # generation_config.stop_strings = { "<|im_end|>", "<|endoftext|>"}

    def stream_generator() -> Generator[str, None, None]:
        for chunk in chunk_generator():
            yield f"data: {chunk.model_dump_json()}\n\n"

    def chunk_generator() -> Generator[OpenAICompletionChunkResponse, None, None]:
        token_queue: queue.Queue[str | None] = queue.Queue(5)

        def put_queue(w: str | None):
            # token_queue.put_nowait(w)
            token_queue.put(w, timeout=10)

        stop_stream_handling: queue.Queue[bool | None] = queue.Queue()

        def run_inference():
            class Streamer(ov_genai.StreamerBase):
                def __init__(self):
                    super().__init__()
                    self.counter = 0

                def end(self) -> None:
                    log.debug("stream end")

                def write(self, tokens: Sequence[typing.SupportsInt]) -> StreamingStatus:
                    decoded_tokens: list[str] = tokenizer.decode(tokens=[tokens], skip_special_tokens=False)
                    try:
                        self.counter += 1
                        log_stream.info(f"decoded '{decoded_tokens}' from {tokens}, num {self.counter}")
                        # if is_disconnected():
                        #     log_stream.info("stream finished by user disconnected")
                        #     return StreamingStatus.STOP
                        # if not stop_stream_handling.empty() and stop_stream_handling.get_nowait():
                        #     log.debug("stream finished by stop signal")
                        #     return StreamingStatus.STOP
                        while True:
                            try:
                                for t in decoded_tokens:
                                    put_queue(t)
                                break
                            except queue.Full:
                                log.info("steam queue full")
                            finally:
                                if is_disconnected():
                                    log_stream.info("stream finished by user disconnected")
                                    return StreamingStatus.STOP
                                if not stop_stream_handling.empty() and stop_stream_handling.get_nowait():
                                    log.debug("stream finished by stop signal")
                                    return StreamingStatus.STOP

                        return StreamingStatus.RUNNING
                    except Exception as e:
                        log_stream.error(f"streamer error: {e}", exc_info=e)
                        return StreamingStatus.CANCEL

            result: VLMDecodedResults
            try:
                pipe.start_chat()
                log.info(f"inference starting")
                before_generate_mem = get_current_memory()
                streamer = Streamer()
                result = pipe.generate(prompt=full_prompt, generation_config=generation_config, streamer=streamer)

                after_generate_mem = get_current_memory()
                generate_cost = after_generate_mem - before_generate_mem
                log.debug(f"consumed memory: {after_generate_mem:.2f} MB, generate cost: {generate_cost:.2f} MB")

                if log.isEnabledFor(logging.DEBUG):
                    log.debug(f"inference finished reason: {result.finish_reasons}, result:{result.texts}")
                else:
                    log.info(f"inference finished")
            except Exception as e:
                log.error(f"inference error: {e}", exc_info=e)
                raise
            finally:
                # send stop stream token
                # token_queue.put_nowait(None)
                pipe.finish_chat()

        inference_task = executor.submit(run_inference)

        log_inference_processing = logging.getLogger("inference.processing")

        thinking_progress_counter = 1 if is_prompt_start_thinking(full_prompt) else 0
        role = ROLE_ASSISTANT

        if thinking_progress_counter:
            log_inference_processing.debug("start with thinking")

        possible_tool_call_expression = ""
        possible_call_in_progress = False
        tool_call_count = 0

        class State(Enum):
            CONVERSATION = 1
            THINK = 2
            TOOL_CALL = 3

        states: list[State] = []

        def get_last_state():
            return states[-1] if states else None

        def remove_state(expected_state: State):
            s = get_last_state()
            if s == expected_state:
                states.pop()
            else:
                log.error(f"unexpected state {get_last_state}, expected {expected_state}")

        states.append(State.CONVERSATION)  # by default conversation is opened by assistant in chat template
        in_conversation = True  # by default conversation is opened by assistant in chat template
        if thinking_progress_counter > 0:
            states.append(State.THINK)

        try:
            token_number = 0
            token_conversation_start_number: int = -1
            expect_role = False
            phrase_tick: float | None = None
            possible_tool_call_expression_tick: float | None = None
            phrase = ""
            empty_conversation_counter = 0

            no_conversation_counter = 0

            stop_inference = False
            while not stop_inference:
                if is_disconnected():
                    break

                raw_token = token_queue.get()
                token_number += 1
                log_inference_processing.info(f"next token: {raw_token}, num: {token_number}")

                if raw_token is None:
                    stop_chunk = new_chunk(role=role, finish_reason="tool_calls" if tool_call_count > 0 else "stop")
                    yield stop_chunk
                    break

                if is_conversation_start(raw_token):
                    states.append(State.CONVERSATION)
                    in_conversation = True
                    # role = ""
                    no_conversation_counter = 0
                    token_conversation_start_number = token_number
                    expect_role = True

                    phrase = phrase.rstrip()
                    if len(phrase) > 0:
                        log_inference_processing.info(
                            f"phrase before conversation: {phrase}, last token number: {token_number}")

                    phrase = ""
                    phrase_tick = None

                elif is_conversation_end(raw_token):
                    in_conversation = False
                    remove_state(expected_state=State.CONVERSATION)

                    token_conversation_start_number = -1
                    expect_role = False

                    phrase = phrase.rstrip()
                    if len(phrase) == 0:
                        log.debug(f"empty conversation end, role {role}")
                        empty_conversation_counter += 1
                    else:
                        empty_conversation_counter = 0
                        log_inference_processing.info(
                            f"phrase before conversation end: {phrase}, last token number: {token_number}")
                        phrase = ""

                    if empty_conversation_counter > 20:
                        log.warning(f"many empty conversations ({empty_conversation_counter}), abort inference")
                        stop_inference = True

                    if not stop_inference:
                        if tool_call_count > 0:
                            stop_inference = True
                            log.debug(
                                f"stop inference by ending conversation with tool calling (count {tool_call_count})")

                    phrase = ""
                    phrase_tick = None

                elif expect_role and in_conversation and token_number - token_conversation_start_number == 1:
                    if len(raw_token.rstrip()) > 0:  # conversation role
                        expect_role = False
                        prev_role = role
                        role = raw_token
                        log.debug(f"apply conversation role {role}, prev {prev_role}")
                    else:
                        log.debug("empty token where expected role")

                elif think_is_started(raw_token):
                    states.append(State.THINK)
                    # log.debug(f"start thinking by {role}")
                    if len(possible_tool_call_expression) > 0:
                        log.warning(f"start think token inside tool_call {possible_tool_call_expression}")
                    thinking_progress_counter += 1
                    if thinking_progress_counter == 1:
                        log_inference_processing.debug("thinking is starting")
                    else:
                        log_inference_processing.debug(
                            f"More thinking for God of thinking!!! {thinking_progress_counter}")
                elif think_is_over(raw_token):
                    remove_state(expected_state=State.THINK)
                    if len(possible_tool_call_expression) > 0:
                        log.warning(
                            f"stop think token inside tool_call: '{possible_tool_call_expression}', phrase: {phrase}")

                    if thinking_progress_counter > 0:
                        thinking_progress_counter -= 1
                        if thinking_progress_counter == 0:
                            log_inference_processing.debug("thinking is over")
                        else:
                            log_inference_processing.debug(
                                f"intensity of thinking decreased {thinking_progress_counter}")
                    # pass
                elif is_possible_tool_call_start(raw_token):
                    states.append(State.TOOL_CALL)
                    log_inference_processing.debug(f"possible tool call start: {raw_token}")
                    log_inference_processing.info(f"phrase of {role} before tool call: {phrase.rstrip()}")
                    phrase = ""
                    possible_tool_call_expression_tick = time.time()
                    possible_call_in_progress = True
                    possible_tool_call_expression += raw_token
                elif is_possible_tool_call_end(raw_token):
                    remove_state(expected_state=State.TOOL_CALL)
                    log_inference_processing.debug(f"possible tool call end: {raw_token}")
                    possible_tool_call_expression += raw_token
                    parsed_tool_calls: List[ToolCall] = parse_tool_calls(possible_tool_call_expression)
                    if not parsed_tool_calls:
                        log_inference_processing.info(f"phrase like tool calls: {possible_tool_call_expression}")
                        chunk: OpenAICompletionChunkResponse = new_chunk(role=role,
                                                                         content=possible_tool_call_expression)
                        yield chunk
                    else:
                        if log_inference_processing.isEnabledFor(logging.INFO):
                            adapter = TypeAdapter(List[ToolCall])
                            log_inference_processing.info(
                                f"tool call: {adapter.dump_json(parsed_tool_calls).decode("utf-8")}")
                        chunk: OpenAICompletionChunkResponse = new_chunk(role=role, tool_calls=parsed_tool_calls)
                        yield chunk
                    possible_call_in_progress = False
                    possible_tool_call_expression = ""
                    possible_tool_call_expression_tick = None
                    tool_call_count += 1
                else:
                    last_state = get_last_state()
                    if possible_call_in_progress and last_state == State.TOOL_CALL:
                        possible_tool_call_expression += raw_token
                        now_time = time.time()
                        possible_tool_call_expression_time = now_time - possible_tool_call_expression_tick
                        if possible_tool_call_expression_time >= 10:
                            possible_tool_call_expression_tick = now_time
                            log_inference_processing.debug(f"possible tool call part: {possible_tool_call_expression}")
                    else:

                        if ROLE_ASSISTANT != role:
                            log.warning(f"unexpected role {role}")
                            # stop_inference = True

                        phrase = phrase + raw_token

                        now_time = time.time()
                        if phrase_tick is None:
                            phrase_tick = now_time

                        phrase_time = now_time - phrase_tick
                        if phrase_time >= 10:
                            phrase_tick = now_time
                            log_inference_processing.debug(f"phrase part: {phrase}")
                        phrase_end = phrase.endswith("\n")
                        if phrase_end:
                            phrase = phrase.rstrip()
                            if len(phrase) > 0:
                                log_inference_processing.info(
                                    f"phrase of {role}: '{phrase.rstrip()}', last token number: {token_number}")
                                phrase = ""

                        if not last_state:
                            log.debug("no more conversations")
                            no_conversation_counter += 1
                            if no_conversation_counter > 5:
                                log.debug(
                                    f"empty conversations limits exceed ({no_conversation_counter}), abort inference")
                                stop_inference = True
                        else:
                            if in_conversation:
                                chunk = new_chunk(role=role, content=raw_token, thinking=thinking_progress_counter > 0)
                                yield chunk
                            else:
                                log.warning(f"strange phrase of role {role}: {phrase}")

        except Exception as e:
            log.error(f"inference result processing error: {e}", exc_info=e)
        finally:
            stop_stream_handling.put_nowait(True)
            if not inference_task.done():
                log_inference_processing.info("waiting for inference to complete")
                try:
                    r = inference_task.result(timeout=20)
                except Exception as e:
                    log.error(f"waiting inference completion error: {e}", exc_info=e)
            log.info("inference handling is done")

    if body.stream:
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        content = ""
        reasoning_content = ""
        tool_calls: list[ToolCall] = []
        finish_reason = "stop"

        for chunk_data in chunk_generator():
            choices = chunk_data.choices
            if choices:
                finish_reason = choices[-1].finish_reason | finish_reason
                for choice in choices:
                    delta = choice.delta
                    content += delta.content
                    reasoning_content += delta.reasoning_content
                    tool_calls += delta.tool_calls

        unique_id = str(uuid.uuid4())
        message = ChatCompletionMessage()
        message.role = ROLE_ASSISTANT
        message.content = content
        message.reasoning_content = reasoning_content
        message.tool_calls = tool_calls
        choice = ChatCompletionChoice(index=0, finish_reason=finish_reason, message=message)
        return OpenAICompletionResponse(id=unique_id, created=int(time.time()), model=model_name,
                                        choices=[choice])


def think_is_over(subword: str) -> bool:
    return subword.strip() == REASONING_END


def think_is_started(subword: str) -> bool:
    return subword.strip() == REASONING_START


def new_chunk(role: str, content: str | None = None, thinking: bool = False,
              tool_calls: Optional[List[ToolCall]] = None,
              finish_reason: str | None = None) -> OpenAICompletionChunkResponse:
    delta = ChatCompletionChunkDelta()
    delta.role = role
    if thinking:
        delta.reasoning_content = content
    else:
        delta.content = content

    if tool_calls:
        delta.tool_calls = tool_calls
    return new_choices(delta=delta, finish_reason=finish_reason)


def new_choices(delta: ChatCompletionChunkDelta, finish_reason: str | None = None) -> OpenAICompletionChunkResponse:
    choice = ChatCompletionChunkChoice(index=0, finish_reason=finish_reason, delta=delta)
    choices = [choice]
    unique_id = str(uuid.uuid4())
    return OpenAICompletionChunkResponse(id=unique_id, model=model_name, created=int(time.time()), choices=choices)


def is_conversation_start(text: str) -> bool:
    return CONVERSATION_START == text.strip()


def is_conversation_end(text: str) -> bool:
    return CONVERSATION_END == text.strip()


def is_possible_tool_call_start(text: str) -> bool:
    return TOOL_CALL_START == text.strip()


def is_possible_tool_call_end(text: str) -> bool:
    return TOOL_CALL_END == text.strip()


def is_prompt_start_thinking(prompt: str) -> bool:
    return prompt.endswith(REASONING_START, 0, len(prompt) - 1) if prompt.endswith(
        '\n') else prompt.endswith(REASONING_START)


def parse_tool_calls(text: str) -> List[ToolCall]:
    """Parses Qwen3-Coder style unique XML tool call blocks."""
    tool_call_blocks = re.findall(f"{TOOL_CALL_START}(.*?){TOOL_CALL_END}", text, re.DOTALL)
    ts = int(time.time())

    parsed_calls: list[ToolCall] = []
    for i, call_block in enumerate(tool_call_blocks):
        func_name = get_func_name(call_block)
        if func_name is None:
            # log
            continue
        arguments = get_arguments(call_block)
        call_id = next(tool_call_counter)
        parsed_calls.append(ToolCall(id=f"call_{ts}_{call_id}_{func_name}",
                                     function=FunctionCall(name=func_name, arguments=arguments)))
    return parsed_calls


def get_arguments(call_block) -> str:
    param_pattern = r"<parameter=(.*?)>(.*?)</parameter>"
    parameters = re.findall(param_pattern, call_block, re.DOTALL)

    arguments = {}
    for param_name, param_value in parameters:
        arguments[param_name.strip()] = param_value.strip()
    if isinstance(arguments, dict):
        arguments_str = json.dumps(arguments, ensure_ascii=False)
    else:
        arguments_str = str(arguments)
    return arguments_str


def get_func_name(call_block) -> str | None:
    func_match = re.search(r"<function=(.*?)>", call_block)
    func_name = func_match.group(1).strip() if func_match else None
    return func_name


if __name__ == "__main__":
    log.info("server starting")
    log_config = uvicorn.config.LOGGING_CONFIG

    log_config["formatters"]["default"]["format"] = log_format_simple
    log_config["formatters"]["access"]["format"] = (
            log_format_prefix + " - %(client_addr)s - '%(request_line)s' %(status_code)s"
    )

    uvicorn.run(app, host="127.0.0.1", port=8888)
