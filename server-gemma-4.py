import asyncio
import json
import logging
import re
import sys
from typing import Callable, List, Optional, Any, Dict, Literal

import openvino_genai as ov_genai
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from openvino_genai.py_openvino_genai import ChatHistory
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LoggingRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Any:
            body_bytes = await request.body()
            body_str = body_bytes.decode("utf-8") if body_bytes else "Пусто"

            logger.info(f"--> ВХОДЯЩИЙ {request.method} {request.url.path}")
            logger.info(f"Тело запроса: {body_str}")

            response = await original_route_handler(request)
            return response

        return custom_route_handler


app = FastAPI()
app.router.route_class = LoggingRoute

model_name = "gemma-4-E2B-it-int8-asym"
model_path = f"./models/{model_name}/1"
config = {"CACHE_DIR": f"./models_cache/{model_name}"}

device_name = "GPU"
logger.info(f"Загрузка модели на {device_name} ...")
try:
    pipe = ov_genai.VLMPipeline(model_path, device_name, **config)
    logger.info("Мультимодальная модель успешно загружена!")
except Exception as e:
    logger.error(f"Критическая ошибка инициализации VLMPipeline: {e}")
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
    messages: List[dict]
    tools: Optional[List[ToolModel]] = None
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 1024


@app.post("/v1/chat/completions")
async def chat(body: ChatCompletionRequest, request: Request):
    messages = body.messages
    logger.info(f"--- Входящий запрос (Сообщений в истории: {len(messages)}) ---")

    chat_history = ChatHistory()
    chat_history_messages = messages  # [msg.model_dump() for msg in messages]
    chat_history.set_messages(chat_history_messages)

    body_tools = body.tools
    tools = [tool.model_dump() for tool in body_tools] if body_tools else None
    if tools:
        chat_history.set_tools(tools)

    # tokenizer = pipe.get_tokenizer()
    # full_prompt = tokenizer.apply_chat_template(history=chat_history, add_generation_prompt=True, tools=tools)

    generation_config = ov_genai.GenerationConfig()
    generation_config.max_new_tokens = body.max_tokens or 1024
    generation_config.temperature = 0.1
    generation_config.top_p = 0.95

    if not body.stream:
        # reuse Stream
        result = await asyncio.to_thread(pipe.generate, history=chat_history, generation_config=generation_config)
        generated_text = result.texts if hasattr(result, 'texts') and result.texts else str(result)

        tool_calls = parse_tool_calls(generated_text)
        message_payload = {"role": "assistant"}

        if tool_calls:
            message_payload["tool_calls"] = tool_calls
            message_payload["content"] = None
            finish_reason = "tool_calls"
        else:
            message_payload["content"] = generated_text
            finish_reason = "stop"

        return {
            "choices": [{
                "index": 0,
                "message": message_payload,
                "finish_reason": finish_reason
            }]
        }

    async def real_time_generator():
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        stop_generation = False

        def my_streamer(subword: str) -> bool:
            if stop_generation:
                return True
            loop.call_soon_threadsafe(queue.put_nowait, subword)
            return False

        async def run_inference():
            try:
                await asyncio.to_thread(
                    pipe.generate,
                    history=chat_history,
                    generation_config=generation_config,
                    streamer=my_streamer
                )
            except Exception as e:
                logger.error(f"Ошибка в инференсе: {e}")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        inference_task = asyncio.create_task(run_inference())

        call_expression = ""
        call_prefix = "call"
        call_in_progress = False
        try:
            while True:
                if await request.is_disconnected():
                    stop_generation = True
                    break

                try:
                    subword = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                if subword is None:
                    break

                if subword == call_prefix:
                    call_in_progress = True

                if call_in_progress:
                    call_expression += subword
                    continue
                else:
                    chunk = {
                        "choices": [{"index": 0, "delta": {"content": subword}, "finish_reason": None}]
                    }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            tool_calls = parse_tool_calls(call_expression, call_prefix) if call_in_progress else None
            if tool_calls:
                logger.info(f"Стрим завершен. Обнаружен инструмент: {tool_calls}")

                # Имитируем для клиента OpenAI стриминговую передачу tool_calls
                for tool_call in tool_calls:
                    chunk = {
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": 0,
                                    "id": tool_call["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tool_call["function"]["name"],
                                        "arguments": tool_call["function"]["arguments"]
                                    }
                                }]
                            },
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                # Закрывающий чанк с корректным finish_reason
                stop_chunk = {
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
                }
                yield f"data: {json.dumps(stop_chunk, ensure_ascii=False)}\n\n"
            else:
                stop_chunk = {
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(stop_chunk, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"Исключение во время стриминга: {e}")
        finally:
            stop_generation = True
            if not inference_task.done():
                logger.info("Ожидание завершения фонового инференса...")
                try:
                    await inference_task
                except Exception as e:
                    logger.error(f"Ошибка при закрытии задачи: {e}")
            logger.info("Ресурсы очищены, сессия стриминга закрыта.")

    return StreamingResponse(real_time_generator(), media_type="text/event-stream")


def parse_tool_calls(text: str, call_prefix: str = "call") -> Optional[List[Dict[str, Any]]]:
    """Парсит сгенерированный текст в поисках вызова инструментов."""
    text_clean = text.strip()

    call_function_part = r":([a-zA-Z0-9_-]+)\s*(\{.*\})"
    call_pattern = f"{re.escape(call_prefix)}{call_function_part}"
    match = re.search(call_pattern, text_clean, re.DOTALL)

    if match:
        func_name = match.group(1).strip()
        raw_args = match.group(2).strip()

        try:
            valid_json_args = re.sub(r'([{,]\s*)([a-zA-Z0-9_-]+)\s*:', r'\1"\2":', raw_args)
            valid_json_args = valid_json_args.replace("'", '"')
            valid_json_args = re.sub(r'\bTrue\b', 'true', valid_json_args)
            valid_json_args = re.sub(r'\bFalse\b', 'false', valid_json_args)
            valid_json_args = re.sub(r'\bNone\b', 'null', valid_json_args)

            arguments_dict = json.loads(valid_json_args)
            arguments_str = json.dumps(arguments_dict, ensure_ascii=False)
        except Exception:
            arguments_str = raw_args

        return [{
            "id": f"call_{func_name}_{int(asyncio.get_event_loop().time())}",
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": arguments_str
            }
        }]

    # todo is it need for gemma 4?
    call_match = re.search(r"<tool_call>(.*?)</tool_call>", text_clean, re.DOTALL)
    if call_match:
        text_clean = call_match.group(1).strip()

    # 3. Извлечение JSON из markdown-блоков
    md_match = re.search(r"```json\s*(.*?)\s*```", text_clean, re.DOTALL)
    if md_match:
        text_clean = md_match.group(1).strip()

    if not (text_clean.startswith("{") or text_clean.startswith("[")):
        return None

    try:
        parsed = json.loads(text_clean)
        tool_calls = []
        calls = parsed if isinstance(parsed, list) else [parsed]

        for idx, call in enumerate(calls):
            name = call.get("name") or call.get("function", {}).get("name")
            arguments = call.get("arguments") or call.get("function", {}).get("arguments") or {}

            if name:
                if isinstance(arguments, dict):
                    arguments_str = json.dumps(arguments, ensure_ascii=False)
                else:
                    arguments_str = str(arguments)

                tool_calls.append({
                    "id": f"call_{idx}_{int(asyncio.get_event_loop().time())}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments_str
                    }
                })
        return tool_calls if tool_calls else None
    except Exception:
        return None


if __name__ == "__main__":
    logger.info("Запуск API сервера...")
    uvicorn.run(app, host="127.0.0.1", port=8888)
