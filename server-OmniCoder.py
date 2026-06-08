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
import uuid

from pydantic import TypeAdapter

from commom_metric_mem import get_current_memory
from common_openapi_model import OpenAIChatCompletionRequest, ToolDefinition, ChatCompletionMessageParam, \
    OpenAICompletionChunkResponse, \
    ChatCompletionChunkChoice, ChatCompletionChunkDelta, ToolCall, FunctionCall, OpenAICompletionResponse, \
    ChatCompletionChoice, ChatCompletionMessage

model_name = "OmniCoder-9B-int4-sym-g128"
model_path = f"./models/{model_name}/1"

reasoning_supported = True

tool_call_start = "<tool_call>"
tool_call_end = "</tool_call>"
reasoning_start = "<think>"
reasoning_end = "</think>"

device_name = "GPU"
model_cache_dir = f"./models_cache/{model_name}"

import openvino_genai as ov_genai

default_max_new_tokens = 65536

scheduler_config = ov_genai.SchedulerConfig()
scheduler_config.enable_prefix_caching = True
scheduler_config.max_num_batched_tokens = 256
# scheduler_config.num_kv_blocks = 4096
scheduler_config.max_num_seqs = 2
scheduler_config.cache_interval_multiplier = None  # 2
scheduler_config.dynamic_split_fuse = True
scheduler_config.use_sparse_attention = False

# scheduler_config.cache_size = 8
# scheduler_config.use_cache_eviction = True

config = {
    "CACHE_DIR": model_cache_dir,
    # "GPU_ENABLE_LARGE_ALLOCATIONS": "YES",
    "KV_CACHE_PRECISION": "u4",
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

app = FastAPI()
app.router.route_class = LoggingRoute

log.info(f"model loading {model_path}, device: {device_name}, scheduler_config: {scheduler_config.to_string()}")

start_mem = get_current_memory()
log.debug(f"consumed memory: {start_mem:.2f} MB")

try:
    pipe = ov_genai.VLMPipeline(models_path=model_path, device=device_name, **config)
    default_generation_config = pipe.get_generation_config()
    log.info(f"model loaded successfully")

    loaded_pipe_mem = get_current_memory()
    pipe_cost = loaded_pipe_mem - start_mem

    log.debug(f"consumed memory: {loaded_pipe_mem:.2f} MB, pipe loading cost: {pipe_cost:.2f} MB")
except Exception as e:
    log.error(f"instantiate pipeline error: {e}", exc_info=e)
    sys.exit(1)


def is_erase(word: str) -> bool:
    service_token: bool = word.strip() in [tool_call_start, tool_call_end, reasoning_start, reasoning_end]
    return service_token


@app.post("/v1/chat/completions")
async def chat(body: OpenAIChatCompletionRequest, request: Request):
    loop = asyncio.get_event_loop()
    is_reasoning_enabled: bool = reasoning_supported and (body.model_config.get("reasoning") or True)

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

    tokenizer = pipe.get_tokenizer()
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
    generation_config.apply_chat_template = True  # False if full_prompt else True

    temp = body.temperature or 0.8
    if temp < 0.05:
        # Greedy Search
        generation_config.do_sample = False
    else:
        generation_config.do_sample = True
        generation_config.temperature = temp
        generation_config.top_p = body.top_p or 0.9
        generation_config.top_k = 40

    generation_config.repetition_penalty = 1.1

    def stream_generator() -> Generator[str, None, None]:
        for chunk in chunk_generator():
            yield f"data: {chunk.model_dump_json()}\n\n"

    def chunk_generator() -> Generator[OpenAICompletionChunkResponse, None, None]:
        token_queue: queue.Queue[str | None] = queue.Queue()

        stop_stream_handling: queue.Queue[bool | None] = queue.Queue()

        def run_inference():
            def streamer(token: str) -> bool:
                log_stream = logging.getLogger("inference.stream")

                def put_queue(w: str | None):
                    token_queue.put_nowait(w)

                try:
                    log_stream.debug(f"stream: {token}")
                    put_queue(token)
                    if is_disconnected():
                        log_stream.info("stream finished by user disconnected")
                        return True
                    if not stop_stream_handling.empty() and stop_stream_handling.get_nowait():
                        log.debug("stream finished by stop signal")
                        return True
                    # elif token is None:
                    #     log.debug("stream finished by None token")
                    #     return True
                    else:
                        return False
                except Exception as e:
                    log_stream.error(f"streamer error: {e}", exc_info=e)
                    # put_queue(None)
                    return True

            result: VLMDecodedResults
            try:
                pipe.start_chat()
                log.info(f"inference starting")
                before_generate_mem = get_current_memory()
                result = pipe.generate(prompt=full_prompt, generation_config=generation_config, streamer=streamer)

                after_generate_mem = get_current_memory()
                generate_cost = after_generate_mem - before_generate_mem
                log.debug(f"consumed memory: {after_generate_mem:.2f} MB, generate cost: {generate_cost:.2f} MB")

                if log.isEnabledFor(logging.DEBUG):
                    log.debug(f"inference finished reason {result.finish_reasons}, result:\n{result.texts}")
                else:
                    log.info(f"inference finished")
            except Exception as e:
                log.error(f"inference error: {e}", exc_info=e)
                raise
            finally:
                # send stop stream token
                token_queue.put_nowait(None)
                pipe.finish_chat()

        inference_task = executor.submit(run_inference)

        log_inference_processing = logging.getLogger("inference.processing")

        thinking_in_progress = 1 if is_prompt_start_thinking(full_prompt) else 0

        if thinking_in_progress:
            log_inference_processing.debug("start with thinking")

        possible_tool_call_expression = ""
        possible_call_in_progress = False
        tool_called = False

        try:
            start_inference_time = time.time()
            start_phrase_time: float | None = None
            phrase_tick: float | None = None
            phrase = ""
            while True:
                if is_disconnected():
                    break

                token = token_queue.get()
                log_inference_processing.debug(f"next token: {token}")

                if token is None:
                    stop_chunk = new_chunk(finish_reason="stop")
                    yield stop_chunk
                    break

                if start_phrase_time is None:
                    start_phrase_time = time.time()
                    phrase_tick = start_phrase_time

                phrase_ended = token.endswith("\n")
                phrase = phrase + token
                if phrase_ended:
                    log_inference_processing.info(f"phrase: {phrase.rstrip()}")
                    phrase = ""
                    start_phrase_time = None
                else:
                    now_time = time.time()
                    phrase_time = now_time - phrase_tick
                    if phrase_time >= 10:
                        phrase_tick = now_time
                        log_inference_processing.debug(f"phrase part: {phrase}")

                if is_reasoning_enabled:
                    if think_is_started(token):
                        if len(possible_tool_call_expression) > 0:
                            log.warning(f"start think token inside tool_call {possible_tool_call_expression}")
                        thinking_in_progress += 1
                        if thinking_in_progress == 1:
                            log_inference_processing.debug("thinking is starting")
                        else:
                            log_inference_processing.debug(
                                f"More thinking for God of thinking!!! {thinking_in_progress}")
                    if thinking_in_progress > 0 and think_is_over(token):
                        thinking_in_progress -= 1
                        if thinking_in_progress == 0:
                            log_inference_processing.debug("thinking is over")
                        else:
                            log_inference_processing.debug(f"intensity of thinking decreased {thinking_in_progress}")

                if is_possible_tool_call_start(token):
                    log_inference_processing.debug(f"possible tool call start: {token}")
                    possible_call_in_progress = True
                    possible_tool_call_expression += token
                elif is_possible_tool_call_end(token):
                    log_inference_processing.debug(f"possible tool call end: {token}")
                    possible_tool_call_expression += token
                    parsed_tool_calls: List[ToolCall] = parse_tool_calls(possible_tool_call_expression)
                    if not parsed_tool_calls:
                        log_inference_processing.info(f"phrase like tool calls: {possible_tool_call_expression}")
                        chunk = new_chunk(content=possible_tool_call_expression)
                        yield chunk
                    else:
                        if log_inference_processing.isEnabledFor(logging.INFO):
                            adapter = TypeAdapter(List[ToolCall])
                            log_inference_processing.info(
                                f"tool call: {adapter.dump_json(parsed_tool_calls).decode("utf-8")}")
                        chunk = new_chunk(tool_calls=parsed_tool_calls)
                        yield chunk
                    possible_call_in_progress = False
                    possible_tool_call_expression = ""
                    tool_called = True
                elif possible_call_in_progress:
                    possible_tool_call_expression += token
                    # log.debug(f"form tool call expression '{possible_tool_call_expression}'")
                else:
                    if tool_called:
                        log_inference_processing.warning(
                            "model trying to generate tokens after tool call, inference is aborted.")
                        stop_chunk = new_chunk(finish_reason="tool_call")
                        yield stop_chunk
                        break
                    else:
                        if not is_erase(token):
                            chunk = new_chunk(token, thinking=thinking_in_progress > 0)
                            yield chunk

        except Exception as e:
            log.error(f"inference result processing error: {e}", exc_info=e)
        finally:
            stop_stream_handling.put_nowait(True)
            if not inference_task.done():
                log_inference_processing.info("waiting for inference to complete")
                try:
                    r = inference_task.result()
                except Exception as e:
                    log.error(f"waiting inference completion error: {e}", exc_info=e)
            log.info("inference handling is done")

    def is_disconnected() -> bool:
        disconnected = False
        try:
            disconnected = asyncio.run_coroutine_threadsafe(request.is_disconnected(), loop).result(0.5)
            if disconnected:
                log.debug(f"disconnected http request")
        except asyncio.TimeoutError:
            log.debug(f"disconnected http request check timeout")
        return disconnected

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
        message.role = "assistant"
        message.content = content
        message.reasoning_content = reasoning_content
        message.tool_calls = tool_calls
        choice = ChatCompletionChoice(index=0, finish_reason=finish_reason, message=message)
        return OpenAICompletionResponse(id=unique_id, created=int(time.time()), model=model_name,
                                        choices=[choice])


def think_is_over(subword: str) -> bool:
    return subword.strip() == reasoning_end


def think_is_started(subword: str) -> bool:
    return subword.strip() == reasoning_start


def new_chunk(content: str | None = None, thinking: bool = False,
              tool_calls: Optional[List[ToolCall]] = None,
              finish_reason: str | None = None) -> OpenAICompletionChunkResponse:
    delta = ChatCompletionChunkDelta()

    if thinking:
        delta.reasoning_content = content
    else:
        delta.content = content

    if tool_calls:
        # def tool_call_convert(index: int, tool_call: ToolCall) -> dict[str, Any]:
        #     tool_call_function_ = tool_call["function"]
        #     tool_call_id_ = tool_call["id"]
        #     return {
        #         "index": index,
        #         "id": tool_call_id_,
        #         "type": "function",
        #         "function": {
        #             "name": tool_call_function_["name"],
        #             "arguments": tool_call_function_["arguments"]
        #         }
        #     }

        # tool_calls = list(map(lambda p: tool_call_convert(p[0], p[1]), enumerate(tool_calls)))
        delta.tool_calls = tool_calls
    return new_choices(delta=delta, finish_reason=finish_reason)


def new_choices(delta: ChatCompletionChunkDelta, finish_reason: str | None = None) -> OpenAICompletionChunkResponse:
    choice = ChatCompletionChunkChoice(index=0, finish_reason=finish_reason, delta=delta)
    choices = [choice]
    unique_id = str(uuid.uuid4())
    return OpenAICompletionChunkResponse(id=unique_id, model=model_name, created=int(time.time()), choices=choices)


def is_possible_tool_call_start(text: str) -> bool:
    return tool_call_start == text.strip()


def is_possible_tool_call_end(text: str) -> bool:
    return tool_call_end == text.strip()


def is_prompt_start_thinking(prompt: str) -> bool:
    return prompt.endswith(reasoning_start, 0, len(prompt) - 1) if prompt.endswith(
        '\n') else prompt.endswith(reasoning_start)


def parse_tool_calls(text: str) -> List[ToolCall]:
    """Parses Qwen3-Coder style unique XML tool call blocks."""
    tool_call_blocks = re.findall(f"{tool_call_start}(.*?){tool_call_end}", text, re.DOTALL)
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
