"""Pydantic models of the Ollama-compatible API."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict

Role = Literal["system", "user", "assistant", "tool"]


class OllamaMessage(BaseModel):
    """One message of an Ollama chat request."""
    model_config = ConfigDict(extra="allow")

    role: Role = "user"
    content: str = ""
    images: Optional[List[str]] = None


class OllamaOptions(BaseModel):
    """Generation options of an Ollama request."""
    model_config = ConfigDict(extra="allow")

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    num_predict: Optional[int] = None
    num_ctx: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    repeat_penalty: Optional[float] = None


class OllamaChatRequest(BaseModel):
    """Request body of POST /api/chat."""
    model_config = ConfigDict(extra="allow")

    model: str = ""
    messages: List[OllamaMessage] = []
    stream: bool = True
    options: Optional[OllamaOptions] = None
    tools: Optional[List[Dict[str, Any]]] = None
    format: Optional[str] = None
    keep_alive: Optional[str] = None


class OllamaGenerateRequest(BaseModel):
    """Request body of POST /api/generate."""
    model_config = ConfigDict(extra="allow")

    model: str = ""
    prompt: str = ""
    system: Optional[str] = None
    images: Optional[List[str]] = None
    stream: bool = True
    options: Optional[OllamaOptions] = None
    keep_alive: Optional[str] = None


class OllamaShowRequest(BaseModel):
    """Request body of POST /api/show."""
    model_config = ConfigDict(extra="allow")

    model: str = ""
    verbose: bool = False


class OllamaModelDetails(BaseModel):
    """Details of a model entry of GET /api/tags."""
    parent_model: str = ""
    format: str = "gguf"
    family: str = ""
    families: List[str] = []
    parameter_size: str = ""
    quantization_level: str = ""


class OllamaModelEntry(BaseModel):
    """One model entry of GET /api/tags."""
    name: str
    model: str
    modified_at: str
    size: int = 0
    digest: str = ""
    details: OllamaModelDetails = OllamaModelDetails()


class OllamaTagsResponse(BaseModel):
    """Response body of GET /api/tags."""
    models: List[OllamaModelEntry] = []


class OllamaShowResponse(BaseModel):
    """Response body of POST /api/show."""
    model_config = ConfigDict(extra="allow")

    model: str = ""
    license: str = ""
    modelfile: str = ""
    parameters: str = ""
    template: str = ""
    details: OllamaModelDetails = OllamaModelDetails()
    model_info: Dict[str, Any] = {}
    capabilities: List[str] = ["completion"]


class OllamaChatResponseMessage(BaseModel):
    """The message of an Ollama chat response."""
    model_config = ConfigDict(extra="allow")

    role: Role = "assistant"
    content: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    thinking: Optional[str] = None


class OllamaChatResponse(BaseModel):
    """Response body of POST /api/chat."""
    model_config = ConfigDict(extra="allow")

    model: str = ""
    created_at: str = ""
    message: OllamaChatResponseMessage = OllamaChatResponseMessage()
    done: bool = True
    done_reason: Optional[str] = None
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0


class OllamaGenerateResponse(BaseModel):
    """Response body of POST /api/generate."""
    model_config = ConfigDict(extra="allow")

    model: str = ""
    created_at: str = ""
    prompt: str = ""
    response: str = ""
    done: bool = True
    done_reason: Optional[str] = None
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0


class OllamaVersionResponse(BaseModel):
    """Response body of GET /api/version."""
    version: str = "0.0.0"
