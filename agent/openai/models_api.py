"""Pydantic models of the OpenAI GET /v1/models endpoint."""
from typing import List, Optional

from pydantic import BaseModel


class ModelPermission(BaseModel):
    """Permission entry attached to a served model."""
    id: str
    object: str = "model_permission"
    created: int
    allow_create_engine: bool
    allow_sampling: bool
    allow_logprobs: bool
    allow_search_indices: bool
    allow_view: bool
    allow_fine_tuning: bool
    organization: str
    group: Optional[str] = None
    is_blocking: bool


class ModelObject(BaseModel):
    """Metadata of a single served model."""
    id: Optional[str] = None
    object: str = "model"
    created: Optional[int] = None
    owned_by: Optional[str] = None
    permission: Optional[List[ModelPermission]] = None
    root: Optional[str] = None
    parent: Optional[str] = None
    max_model_len: Optional[int] = None
    supported_modalities: Optional[List[str]] = None


class ModelsListResponse(BaseModel):
    """Response body of GET /v1/models."""
    object: str = "list"
    data: List[ModelObject]
