from openai import BaseModel

from agent.openai.chat_completions_api import ChatCompletionMessageParam


class UserContext(BaseModel):
    os: str | None = None
    workdir: str | None = None
    model_architectures: set[str] = {}
    messages: list[ChatCompletionMessageParam] = []
