from typing import List, Optional, Union, Dict, Any

from openai.types.chat import ChatCompletionContentPartTextParam, ChatCompletionToolChoiceOptionParam
from pydantic import BaseModel, ConfigDict, Field

CHAT_COMPLETION_CHUNK = "chat.completion.chunk"
CHAT_COMPLETION = "chat.completion"


class Function(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    arguments: str  # JSON string


class ChatCompletionMessageFunctionToolCallParam(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    type: str = "function"
    function: Function


# --- Request Components ---

class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "text"  # Can be "text" or "json_object"


class ChatCompletionMessageParam(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str  # "system", "user", "assistant", "tool", or "function"
    content: Union[str, List[ChatCompletionContentPartTextParam]]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageFunctionToolCallParam]] = None


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]  # JSON Schema object


class ChatCompletionFunctionToolParam(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "function"
    function: FunctionDefinition


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")
    include_usage: Optional[bool] = None


# --- Main Request Schema ---

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = ""
    messages: List[ChatCompletionMessageParam] = []

    # Common Parameters
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20)
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    n: Optional[int] = Field(default=1, ge=1)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = True
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    user: Optional[str] = None

    tools: List[ChatCompletionFunctionToolParam] = []
    tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None

    metadata: Optional[Dict[str, str]] = None
