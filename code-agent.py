import time
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import openvino_genai as ov_genai

# ==========================================
# 1. КОНФИГУРАЦИЯ И ИНИЦИАЛИЗАЦИЯ OPENVINO
# ==========================================
MODEL_PATH = "./qwen_openvino"  # Путь к вашей сконвертированной модели
DEVICE = "GPU"                  # Можно изменить на "GPU" или "NPU"

print(f"Loading OpenVINO GenAI model from {MODEL_PATH} to {DEVICE}...")
# Используем `ov_genai.LLMPipeline` для работы с текстовыми моделями
pipe = ov_genai.LLMPipeline(MODEL_PATH, DEVICE)
print("Model loaded successfully!")

app = FastAPI(title="OpenVINO GenAI OpenAI-Compatible Server")

# ==========================================
# 2. СХЕМЫ ДАННЫХ (OPENAI COMPLIANT REQ/RES)
# ==========================================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "openvino-model"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False  # В данном примере обрабатываем non-stream

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

# ==========================================
# 3. ЭНДПОИНТ API
# ==========================================
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not implemented in this basic example.")

    try:
        # Настройка параметров генерации под запрос пользователя
        config = ov_genai.GenerationConfig()
        config.temperature = request.temperature
        config.max_new_tokens = request.max_tokens
        
        # Важно для детерминированного поведения при температуре > 0
        if request.temperature == 0:
            config.do_sample = False
        else:
            config.do_sample = True

        # OpenVINO GenAI LLMPipeline принимает на вход как чистую строку (prompt), 
        # так и структурированный диалог. Формируем диалог:
        ov_messages = []
        for msg in request.messages:
            # Преобразуем входящие сообщения в формат словарей, совместимый с шаблоном чата модели
            ov_messages.append({"role": msg.role, "content": msg.content})

        # Запускаем локальную генерацию (генерация синхронная, но выполняется быстро)
        # Использование шаблона чата (chat template) происходит автоматически внутри OpenVINO
        response_text = pipe.generate(ov_messages, config)

        # Формируем стандартный OpenAI-совместимый ответ
        response_id = f"chatcmpl-{int(time.time())}"
        
        return ChatCompletionResponse(
            id=response_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            # Заглушка для usage (OpenVINO GenAI возвращает только текст, 
            # для подсчета токенов потребовался бы отдельный вызов токенизатора)
            usage=ChatCompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenVINO Generation Error: {str(e)}")

# Эндпоинт для проверки работоспособности
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "openvino-model",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openvino"
            }
        ]
    }

# ==========================================
# 4. ЗАПУСК СЕРВЕРА
# ==========================================
if __name__ == "__main__":
    # Запуск сервера на порту 8080
    uvicorn.run(app, host="0.0.0.0", port=8080)
