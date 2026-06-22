from typing import List, Optional, Union, Dict, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CHAT_COMPLETION_CHUNK = "chat.completion.chunk"
CHAT_COMPLETION = "chat.completion"


# --- Request Components ---

class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "text"  # Can be "text" or "json_object"


class ChatCompletionMessageParam(BaseModel):
    """Represents an item in the 'messages' array."""
    model_config = ConfigDict(extra="allow")
    role: str  # "system", "user", "assistant", "tool", or "function"
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]  # JSON Schema object


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "function"
    function: FunctionDefinition


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")
    include_usage: Optional[bool] = None


# --- Main Request Schema ---

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: List[ChatCompletionMessageParam]

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

    # Tools and Functions
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    metadata: Optional[Dict[str, Any]] = None


# --- Response Components ---

class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    type: str = "function"
    function: FunctionCall


class ChatCompletionMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Optional[str] = None
    content: Optional[str] = None
    refusal: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    reasoning_content: Optional[str] = None


class ChoiceLogprobs(BaseModel):
    model_config = ConfigDict(extra="allow")
    content: Optional[List[Dict[str, Any]]] = None


class CompletionUsage(BaseModel):
    model_config = ConfigDict(extra="allow")
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # Accounts for newer models containing reasoning or caching metrics
    prompt_tokens_details: Optional[Dict[str, int]] = None
    completion_tokens_details: Optional[Dict[str, int]] = None


# --- Main Response Schema ---

class ChatCompletionChoice(BaseModel):
    model_config = ConfigDict(extra="allow")
    index: int = 0
    delta: ChatCompletionMessage | None = None
    message: ChatCompletionMessage | None = None
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None
    logprobs: Optional[ChoiceLogprobs] = None


class CompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    object: str = CHAT_COMPLETION_CHUNK
    id: str | None = None
    created: int = None
    model: str | None = None
    usage: Optional[CompletionUsage] = None
    system_fingerprint: Optional[str] = None
    choices: List[ChatCompletionChoice]
    metadata: Optional[Dict[str, Any]] = None
