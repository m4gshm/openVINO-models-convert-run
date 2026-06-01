# Сохраните как server.py и запустите (требуется pip install fastapi uvicorn)
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import openvino_genai as ov_genai

app = FastAPI()

# # Инициализируем модель автокомплита строго на NPU
# pipe_autocomplete = ov_genai.LLMPipeline("./models/qwen-1.5b-ov", "NPU")

# Инициализируем модель рассуждений на iGPU (Gemma)
pipe_reasoning = ov_genai.LLMPipeline("./models/gemma-4-E2B-it-int8-asym/1", "GPU")

# Описываем структуру, которую РЕАЛЬНО присылает VS Code (OpenAI формат)
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 512

# @app.post("/v1/autocomplete")
# def autocomplete(query: Query):
#     # Этот эндпоинт работает на энергоэффективном NPU
#     result = pipe_autocomplete.generate(query.prompt, max_new_tokens=64)
#     return {"choices": [{"text": result}]}

@app.post("/v1/chat/completions")
def chat(body: ChatCompletionRequest):
    # Разворачиваем историю сообщений от VS Code в плоский текст для модели
    full_prompt = ""
    for msg in body.messages:
        if msg.role == "user":
            full_prompt += f"<start_of_turn>user\n{msg.content}<end_of_turn>\n"
        elif msg.role == "assistant":
            full_prompt += f"<start_of_turn>model\n{msg.content}<end_of_turn>\n"
            
    full_prompt += "<start_of_turn>model\n" # Токен начала ответа для Gemma
    
    # Этот эндпоинт задействует мощную графику Arc
    result = pipe_reasoning.generate(full_prompt, max_new_tokens=body.max_tokens or 512)
    
    # Возвращаем ответ в формате, который ожидает плагин Continue / VS Code
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop"
            }
        ]
    }
