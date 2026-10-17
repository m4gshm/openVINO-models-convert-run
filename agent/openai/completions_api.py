"""Pydantic models of the OpenAI legacy text-completions endpoint."""
from typing import List, Optional, Any, Literal

from pydantic import BaseModel

TEXT_COMPLETION_CHUNK = "text_completion.chunk"

TEXT_COMPLETION = "text_completion"


class CompletionRequest(BaseModel):
    """Request body of POST /v1/completions."""
    model: Optional[str] = None
    prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = None
    stop: Optional[List[str]] = None


class CompletionChoice(BaseModel):
    """One completion variant returned in a completions response."""
    text: str
    index: int = 0
    logprobs: Optional[Any] = None
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = None


class CompletionResponse(BaseModel):
    """Response body of POST /v1/completions."""
    id: str | None
    object: str = TEXT_COMPLETION_CHUNK
    created: int | None = None
    model: str | None = None
    choices: List[CompletionChoice]
