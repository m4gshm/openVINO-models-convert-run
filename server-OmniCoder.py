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

scheduler_config = ov_genai.SchedulerConfig()
scheduler_config.enable_prefix_caching = True
scheduler_config.cache_size = 8
scheduler_config.max_num_batched_tokens = 1024
# scheduler_config.num_kv_blocks = 4096
# scheduler_config.max_num_seqs = 256
scheduler_config.cache_interval_multiplier = None # 2
scheduler_config.dynamic_split_fuse = True
# scheduler_config.use_cache_eviction = True
scheduler_config.use_sparse_attention = False

config = {
    # ov.properties.log.level(): ov.properties.log.Level.DEBUG,
    # "OPENVINO_LOG_LEVEL": "DEBUG",
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
from typing import List, Optional, Any, Dict

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from openvino_genai.py_openvino_genai import ChatHistory, VLMDecodedResults
from pydantic import BaseModel
from pydantic.json import pydantic_encoder

from common_log import log_format_simple, log_format_prefix, LoggingRoute

executor = ThreadPoolExecutor()
tool_call_counter = itertools.count(start=0)

os.environ["OPENVINO_LOG_LEVEL"] = "4"
os.environ["ONEDNN_VERBOSE"] = "ON"
os.environ["ONEDNN_VERBOSE_TIMESTAMP"] = "1"

log = logging.getLogger(__name__)

app = FastAPI()
app.router.route_class = LoggingRoute

log.info(f"model loading {model_path}, device: {device_name}, scheduler_config: {scheduler_config.to_string()}")
try:
    pipe = ov_genai.VLMPipeline(models_path=model_path, device=device_name, **config)
    log.info("model loaded successfully")
except Exception as e:
    log.error(e)
    sys.exit(1)


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[dict] = []
    tools: Optional[List[dict]] = None
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    reasoning: Optional[int] = True
    top_p: Optional[float] = None
    temperature: Optional[float] = None


@app.post("/v1/chat/completions")
async def chat(body: ChatCompletionRequest, request: Request):
    loop = asyncio.get_event_loop()
    is_reasoning_enabled = reasoning_supported and (body.reasoning or True)

    messages = body.messages

    log.info(f"inbound history messages {len(messages)}")

    chat_history = ChatHistory()
    chat_history_messages = messages
    chat_history.set_messages(chat_history_messages)

    tools = body.tools
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

    log.debug(f"prompt:\n{full_prompt}")

    generation_config = ov_genai.GenerationConfig()
    generation_config.max_new_tokens = body.max_tokens or 4096
    generation_config.apply_chat_template = False if full_prompt else True

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

    def stream_generator():
        word_queue: queue.Queue[str | None] = queue.Queue()

        stop_stream_handling: queue.Queue[bool | None] = queue.Queue()

        def run_inference():
            def streamer(word: str) -> bool:
                log_stream = logging.getLogger("inference.stream")

                def put_queue(w: str | None):
                    word_queue.put_nowait(w)

                try:
                    log_stream.debug(f"stream: {word}")
                    put_queue(word)
                    if is_disconnected():
                        log_stream.info("stream finished by user disconnected")
                        return True
                    if not stop_stream_handling.empty() and stop_stream_handling.get_nowait():
                        log.debug("stream finished by stop signal")
                        return True
                    # elif word is None:
                    #     log.debug("stream finished by None word")
                    #     return True
                    else:
                        return False
                except Exception as e:
                    log_stream.error(f"streamer error: {e}")
                    # put_queue(None)
                    return True

            result: VLMDecodedResults
            try:
                log.info(f"inference starting")
                result = pipe.generate(prompt=full_prompt, generation_config=generation_config, streamer=streamer)

                if log_inference.isEnabledFor(logging.DEBUG):
                    log_inference.debug(f"inference finished reason {result.finish_reasons}, result:\n{result.texts}")
                else:
                    log.info(f"inference finished")
            except Exception as e:
                log.error(f"inference error: {e}")
                raise
            finally:
                # send stop stream word
                word_queue.put_nowait(None)

        inference_task = executor.submit(run_inference)

        thinking_in_progress = 1 if is_prompt_start_thinking(full_prompt) else 0

        possible_call_expression = ""
        possible_call_in_progress = False
        tool_called = False
        log_inference = logging.getLogger("inference")
        try:
            while True:
                if is_disconnected():
                    break

                # log.debug(f"waiting next word")
                word = word_queue.get()
                log_inference.debug(f"next word: {word}")

                if word is None:
                    stop_chunk = new_chunk(finish_reason="stop")
                    yield f"data: {json.dumps(stop_chunk, ensure_ascii=False)}\n\n"
                    break

                if is_reasoning_enabled:
                    if think_is_started(word):
                        # log, assert thinking_in_progress == False
                        thinking_in_progress += 1
                    if thinking_in_progress > 0 and think_is_over(word):
                        # log, assert
                        thinking_in_progress -= 1

                if is_possible_tool_call_start(word):
                    # log trace
                    # todo need assert possible_call_in_progress == False
                    possible_call_in_progress = True
                    possible_call_expression += word
                elif is_possible_tool_call_end(word):
                    possible_call_expression += word
                    tool_calls = parse_tool_calls(possible_call_expression)
                    if not tool_calls:
                        log_inference.debug(f"fake tool call: {tool_calls}")
                        chunk = new_chunk(text=possible_call_expression)
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    else:
                        log_inference.debug(f"tool call: {tool_calls}")
                        chunk = new_chunk(tool_calls=tool_calls)
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    possible_call_in_progress = False
                    possible_call_expression = ""
                    tool_called = True
                elif possible_call_in_progress:
                    # log
                    possible_call_expression += word
                    # log.debug(f"form tool call expression '{possible_call_expression}'")
                else:
                    if tool_called:
                        log_inference.warning("model trying to generate tokens after tool call, inference is aborted.")
                        stop_chunk = new_chunk(finish_reason="tool_call")
                        yield f"data: {json.dumps(stop_chunk, ensure_ascii=False)}\n\n"
                        break
                    chunk = new_chunk(word, thinking=thinking_in_progress > 0)
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        except Exception as e:
            log.error(f"inference result processing error: {e}")
        finally:
            stop_stream_handling.put_nowait(True)
            if not inference_task.done():
                log_inference.info("waiting for inference to complete")
                try:
                    r = inference_task.result()
                except Exception as e:
                    log.error(e)
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

    # if body.stream:
    return StreamingResponse(stream_generator(), media_type="text/event-stream")
    # else:
    #     full_content = ""
    #     final_tool_calls = None
    #     final_finish_reason = "stop"
    #
    #     async for chunk_str in stream_generator():
    #         if not chunk_str.startswith("data: "):
    #             continue
    #
    #         clean_json = chunk_str.replace("data: ", "").strip()
    #         if not clean_json:
    #             continue
    #
    #         try:
    #             chunk_data = json.loads(clean_json)
    #             choice = chunk_data["choices"][0]
    #             delta = choice.get("delta", {})
    #
    #             # 1. Собираем обычный текст, если он есть
    #             if "content" in delta and delta["content"]:
    #                 full_content += delta["content"]
    #
    #             # 2. Перехватываем tool_calls, если они пришли
    #             if "tool_calls" in delta:
    #                 if final_tool_calls is None:
    #                     final_tool_calls = []
    #
    #                 # Собираем части аргументов/имени (в стриминге OpenAI они могут дополняться)
    #                 for tc in delta["tool_calls"]:
    #                     # Для простоты, так как ваш генератор сразу отдает готовый tool_call:
    #                     final_tool_calls.append(tc)
    #
    #             # 3. Запоминаем причину остановки
    #             if choice.get("finish_reason"):
    #                 final_finish_reason = choice["finish_reason"]
    #
    #         except Exception as e:
    #             logger.error(f"Ошибка разбора чанка при сборке: {e}")
    #
    #     # Формируем итоговый ответ для клиента
    #     message_payload = {"role": "assistant"}
    #
    #     if final_tool_calls:
    #         message_payload["content"] = None
    #         message_payload["tool_calls"] = final_tool_calls
    #     else:
    #         message_payload["content"] = full_content
    #
    #     return {
    #         "choices": [{
    #             "index": 0,
    #             "message": message_payload,
    #             "finish_reason": final_finish_reason
    #         }]
    #     }


def think_is_over(subword: str) -> bool:
    return subword.strip() == reasoning_end


def think_is_started(subword: str) -> bool:
    return subword.strip() == reasoning_start


def new_chunk(text: str | None = None, thinking: bool = False, tool_calls: Optional[list[dict[str, Any]]] = None,
              finish_reason=None) -> \
        dict[str, Any]:
    delta = {}
    if text:
        delta["reasoning_content" if thinking else "content"] = text
    elif tool_calls:
        def tool_call_convert(index: int, tool_call: dict[str, Any]) -> dict[str, Any]:
            tool_call_function_ = tool_call["function"]
            tool_call_id_ = tool_call["id"]
            return {
                "index": index,
                "id": tool_call_id_,
                "type": "function",
                "function": {
                    "name": tool_call_function_["name"],
                    "arguments": tool_call_function_["arguments"]
                }
            }

        tool_calls = list(map(lambda p: tool_call_convert(p[0], p[1]), enumerate(tool_calls)))
        delta["tool_calls"] = tool_calls
    return {"choices": [{"index": 0, "delta": delta, "model": model_name, "finish_reason": finish_reason}]}


def is_possible_tool_call_start(text: str) -> bool:
    return tool_call_start == text.strip()


def is_possible_tool_call_end(text: str) -> bool:
    return tool_call_end == text.strip()


def is_prompt_start_thinking(prompt: str) -> bool:
    return prompt.endswith(reasoning_start, 0, len(prompt) - 1) if prompt.endswith(
        '\n') else prompt.endswith(reasoning_start)


def parse_tool_calls(text: str) -> Optional[List[Dict[str, Any]]]:
    """Parses Qwen3-Coder style unique XML tool call blocks."""
    tool_call_blocks = re.findall(f"{tool_call_start}(.*?){tool_call_end}", text, re.DOTALL)

    parsed_calls = []
    for i, call_block in enumerate(tool_call_blocks):
        func_name = get_func_name(call_block)
        if func_name is None:
            # log
            continue
        arguments = get_arguments(call_block)
        call_id = next(tool_call_counter)
        parsed_calls.append({
            "id": f"call_{call_id}_{func_name}",
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": arguments,
            },
        })
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
