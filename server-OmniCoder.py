from click import shell_completion

model_name = "OmniCoder-9B-int4-sym-g128"
model_path = f"./models/{model_name}/1"

reasoning_supported = True

tool_call_start = "<tool_call>"
tool_call_end = "</tool_call>"
reasoning_start = "<think>"
reasoning_end = "</think>"

device_name = "GPU"
kv_cache_size = 8

model_cache_dir = f"./models_cache/{model_name}"

import asyncio
import json
import logging.config
import os
import queue as q
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Any, Dict, Literal

import openvino_genai as ov_genai
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from openvino_genai.py_openvino_genai import ChatHistory, VLMDecodedResults
from pydantic import BaseModel
from pydantic.json import pydantic_encoder

executor = ThreadPoolExecutor(max_workers=4)

os.environ["OPENVINO_LOG_LEVEL"] = "4"
os.environ["ONEDNN_VERBOSE"] = "ON"
os.environ["ONEDNN_VERBOSE_TIMESTAMP"] = "1"

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # Keeps non-root loggers alive
    "formatters": {
        "simple": {
            "format": "%(asctime)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "detailed",
            "filename": "app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        },
        "file_http": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "detailed",
            "filename": "http.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        },
    },
    "loggers": {
        "http": {
            "level": "DEBUG",
            "handlers": ["console", "file_http"],
            "propagate": True,
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"],
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

logger.debug("Database connection initialized.")


class LoggingRoute(APIRoute):
    logger = logging.getLogger("http")

    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Any:
            body_bytes = await request.body()
            body_str = body_bytes.decode("utf-8") if body_bytes else "Пусто"
            logger.info(f"--> inbound {request.method} {request.url.path}")
            logger.info(f"body: {body_str}")

            response: Response = await original_route_handler(request)

            logger.info(f"<-- outbound {response.status_code}")

            # if isinstance(response, StreamingResponse):
            #     response_body_bytes = b""
            #     chunks = []
            #     async for chunk in response.body_iterator:
            #         chunks.append(chunk)
            #
            #         if isinstance(chunk, str):
            #             chunk_bytes = chunk.encode("utf-8")
            #         else:
            #             chunk_bytes = chunk
            #
            #         response_body_bytes += chunk_bytes
            #
            #     res_body_str = response_body_bytes.decode("utf-8", errors="ignore")
            #     logger.info(f"Тело стрим-ответа: {res_body_str}")
            #
            #     async def re_iterator():
            #         for chunk in chunks:
            #             yield chunk
            #
            #     response.body_iterator = re_iterator()
            # else:
            #     res_body_str = response.body.decode("utf-8", errors="ignore") if response.body else "Пусто"
            #     logger.info(f"response body: {res_body_str}")

            return response

        return custom_route_handler


app = FastAPI()
app.router.route_class = LoggingRoute

scheduler_config = ov_genai.SchedulerConfig()
scheduler_config.cache_size = kv_cache_size
# scheduler_config.max_num_batched_tokens = 1024
scheduler_config.max_num_seqs = 1
scheduler_config.cache_interval_multiplier = 2
scheduler_config.dynamic_split_fuse = True
# scheduler_config.use_cache_eviction = True
scheduler_config.use_sparse_attention = False
scheduler_config.enable_prefix_caching = True


# SchedulerConfig {
# max_num_batched_tokens: 18446744073709551615
# num_kv_blocks: 0
# cache_size: 0
# num_linear_attention_blocks: 0
# cache_interval_multiplier: unset
# dynamic_split_fuse: true
# use_cache_eviction: false
# max_num_seqs: 256
# enable_prefix_caching: true
# use_sparse_attention: false
# }

config = {
    # ov.properties.log.level(): ov.properties.log.Level.DEBUG,
    # "OPENVINO_LOG_LEVEL": "DEBUG",
    "CACHE_DIR": model_cache_dir,
    # "GPU_ENABLE_LARGE_ALLOCATIONS": "YES",
    # "KV_CACHE_PRECISION": "u4",
    "PERFORMANCE_HINT": "LATENCY", # THROUGHPUT crash process
    "scheduler_config": scheduler_config,
}

logger.info(f"model loading {model_path}, device: {device_name}, scheduler_config: {scheduler_config.to_string()}")
try:

    pipe = ov_genai.VLMPipeline(models_path=model_path, device=device_name, **config)
    logger.info("model loaded successfully")
except Exception as e:
    logger.error(e)
    sys.exit(1)


class ChatMessage(BaseModel):
    role: str
    content: str
    tool_call_id: str | None = None
    tool_calls: list[dict] | None = None


class FunctionModel(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolModel(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionModel


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
    # loop = asyncio.get_running_loop()

    is_reasoning_enabled = reasoning_supported and (body.reasoning or True)

    messages = body.messages

    logger.info(f"inbound history messages {len(messages)}")

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

    logger.debug(f"prompt:\n{full_prompt}")

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

    async def stream_generator():
        word_queue: q.Queue[str | None] = q.Queue()  # asyncio.Queue[str | None] = asyncio.Queue()
        stop_generation: q.Queue[bool | None] = q.Queue()

        def run_inference():
            def streamer(word: str) -> bool:
                try:
                    logger.debug(f"stream: {word}")
                    if not stop_generation.empty() and stop_generation.get_nowait():
                        word_queue.put_nowait(None)
                        # log
                        return True
                    elif word is None:
                        word_queue.put_nowait(None)
                        return True
                    else:
                        word_queue.put_nowait(word)  # loop.call_soon_threadsafe(queue.put_nowait, subword)
                        return False
                except Exception as e:
                    logger.error(f"streamer error: {e}")
                    word_queue.put_nowait(None)
                    return True

            result: VLMDecodedResults
            try:
                logger.info(f"inference starting")
                # generation_config.apply_chat_template = True
                # result = pipe.generate(history=chat_history, generation_config=generation_config, streamer=streamer)
                result = pipe.generate(prompt=full_prompt, generation_config=generation_config, streamer=streamer)

                # for text in result.texts:
                #     words = re.split(r" |\n", text)
                #     for word in words:
                #         word_queue.put_nowait(word)

                if logger.level == logging.DEBUG:
                    logger.debug(f"inference finished reason {result.finish_reasons}, result:\n{result.texts}")
                else:
                    logger.info(f"inference finish reason {result.finish_reasons}")
                word_queue.put_nowait(None)
            except Exception as e:
                logger.error(f"inference error: {e}")
                word_queue.put_nowait(None)
                raise e

        # inference_task = loop.run_in_executor(None, run_inference)

        inference_task = executor.submit(run_inference)

        thinking_in_progress = 1 if is_prompt_start_thinking(full_prompt) else 0

        possible_call_expression = ""
        possible_call_in_progress = False
        try:
            while True:
                is_disconnected = await request.is_disconnected()
                if is_disconnected:  # await request.is_disconnected():
                    stop_generation.put_nowait(True)
                    break

                try:
                    # subword = await asyncio.wait_for(queue.get(), timeout=0.1)
                    subword = word_queue.get()
                except asyncio.TimeoutError:
                    # todo log
                    continue

                if subword is None:
                    stop_chunk = new_chunk(None)
                    yield f"data: {json.dumps(stop_chunk, ensure_ascii=False)}\n\n"
                    break

                if is_reasoning_enabled:
                    if think_is_started(subword):
                        # log, assert thinking_in_progress == False
                        thinking_in_progress += 1
                    if thinking_in_progress > 0 and think_is_over(subword):
                        # log, assert
                        thinking_in_progress -= 1

                # if thinking_in_progress > 0:
                #     chunk = new_chunk(subword, thinking=thinking_in_progress>0)
                #     yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                if is_possible_tool_call_start(subword):
                    # log trace
                    # todo need assert possible_call_in_progress == False
                    possible_call_in_progress = True
                    possible_call_expression += subword
                elif is_possible_tool_call_end(subword):
                    possible_call_expression += subword
                    tool_calls = parse_tool_calls(possible_call_expression)
                    if not tool_calls:
                        logger.debug(f"fake tool call: {tool_calls}")
                        chunk = new_chunk(text=possible_call_expression)
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    else:
                        logger.debug(f"tool call: {tool_calls}")
                        chunk = new_chunk(tool_calls=tool_calls)
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    possible_call_in_progress = False
                    possible_call_expression = ""
                elif possible_call_in_progress:
                    # log
                    possible_call_expression += subword
                    logger.debug(f"form tool call expression '{possible_call_expression}'")
                else:
                    chunk = new_chunk(subword, thinking=thinking_in_progress > 0)
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(e)
        finally:
            stop_generation.put_nowait(True)
            if not inference_task.done():
                logger.info("waiting for inference to complete")
                try:
                    r = inference_task.result()  # await inference_task
                except Exception as e:
                    logger.error(e)
            logger.info("inference handling is done")

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


def new_chunk(text: str | None = None, thinking: bool = False, tool_calls: Optional[list[dict[str, Any]]] = None) -> \
        dict[str, Any]:
    finish_reason = None
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
        # finish_reason = "tool_calls"
    else:
        finish_reason = "stop"
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
    for idx, call_block in enumerate(tool_call_blocks):
        func_name = get_func_name(call_block)
        if func_name is None:
            # log
            continue
        arguments = get_arguments(call_block)
        parsed_calls.append({
            "id": f"call_{idx}_{func_name}",
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
    logger.info("Запуск API сервера...")
    uvicorn.run(app, host="127.0.0.1", port=8888)
