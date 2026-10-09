from typing import List, Optional, Union, Dict, Any

from openai.types.chat import ChatCompletionToolChoiceOptionParam
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


class ContentPartUrl(BaseModel):
    """``image_url`` / ``video_url`` / ``audio_url`` nested object."""
    model_config = ConfigDict(extra="allow")
    url: str = ""
    detail: Optional[str] = None
    media_type: Optional[str] = None


class ContentPartAudio(BaseModel):
    """``input_audio`` nested object with inline base64 payload."""
    model_config = ConfigDict(extra="allow")
    data: str = ""
    format: Optional[str] = None
    media_type: Optional[str] = None


class ChatCompletionContentPartParam(BaseModel):
    """One content part of a message: text, image, audio or video.

    Declared leniently (extra fields allowed) to accept vendor specific shapes:
    OpenAI ``image_url``/``input_audio``, HuggingFace style ``image``/``audio``/``video``.
    """
    model_config = ConfigDict(extra="allow")
    type: str = "text"  # "text", "image_url", "input_audio", "video_url", "image", "audio", "refusal"
    text: Optional[str] = None
    refusal: Optional[str] = None
    image_url: Optional[ContentPartUrl] = None
    video_url: Optional[ContentPartUrl] = None
    audio_url: Optional[ContentPartUrl] = None
    input_audio: Optional[ContentPartAudio] = None
    image: Optional[str] = None
    video: Optional[str] = None
    audio: Optional[str] = None


class ChatCompletionMessageParam(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str  # "system", "user", "assistant", "tool", or "function"
    content: Optional[Union[str, List[ChatCompletionContentPartParam]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageFunctionToolCallParam]] = None


class FunctionDefinitionParameters(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "object"
    properties: Dict[str, Any]
    required: set[str] | None = None
    additionalProperties: bool | None = False


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    description: Optional[str] = None
    parameters: FunctionDefinitionParameters


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
