"""OpenAI proxy controller: forwards requests to a remote OpenAI-compatible endpoint."""
import importlib
import logging
import threading
import uuid
from typing import Iterable

from openai.types.chat import ChatCompletionChunk

from agent.openai.chat_completions_api import ChatCompletionRequest
from agent.openai.models_api import ModelsListResponse, ModelObject

# Import openai package explicitly to avoid conflict with local openai folder
_openai_pkg = importlib.import_module('openai')
openai = _openai_pkg
_openai_types_chat = importlib.import_module('openai.types.chat')
_chunk_choice = importlib.import_module('openai.types.chat.chat_completion_chunk').Choice
_chunk_delta = importlib.import_module('openai.types.chat.chat_completion_chunk').ChoiceDelta
# The imports below must run after the openai importlib workaround above.
# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports
from starlette.requests import Request

from agent.inference.token_handler import TokenHandlerConfig
from agent.openai import GenerateOpts, completions_api
from agent.openai.chat_api import new_stop_response, ROLE_ASSISTANT, new_chat_completion_chunk, new_usage_chunk
from agent.openai.engine_rest_common import (ControllerConfig, BaseController,
                                             is_stream_request, new_http_response_completions)

log = logging.getLogger(__name__)


def remote_modalities(model: object) -> list[str] | None:
    """Modalities declared by the remote endpoint for one model entry.

    The OpenAI response schema has no such field (``Model`` is id/object/created/owned_by), but
    servers commonly add it as an extension, and the SDK keeps unknown fields. When the remote
    stays silent the result is None, so ``/v1/models`` omits the field instead of guessing it.
    """
    declared = getattr(model, "supported_modalities", None)
    if declared is None:
        declared = (getattr(model, "model_extra", None) or {}).get("supported_modalities")
    if not isinstance(declared, (list, tuple, set)):
        return None
    names = sorted({name for name in declared if isinstance(name, str) and name})
    return names or None


class OpenAiController(BaseController):
    """
    OpenAI-compatible controller that uses OpenAI Python SDK to call
    remote/inference endpoints without requiring OpenVINO libraries.
    
    This enables running on machines without GPU (e.g., macOS) where
    OpenVINO cannot load native accelerators.
    """

    # Wired from server args; grouping the parameters into a config is a separate refactor.
    def __init__(self,  # pylint: disable=too-many-arguments,too-many-positional-arguments
                 config: ControllerConfig, api_key: str,
                 base_url: str, handler_config: TokenHandlerConfig,
                 stop_signal: threading.Event, generate_opts: GenerateOpts):
        """
        Initialize OpenAI controller.

        Args:
            config: Controller configuration (model_name, max_prompt_len, etc.)
            api_key: OpenAI API key for authentication
            base_url: OpenAI-compatible API endpoint URL
            handler_config: Token handler configuration
            stop_signal: Threading event for stopping inference
            generate_opts: Generation options (temperature, max_tokens, etc.)
        """
        super().__init__(config, handler_config, generate_opts, stop_signal)

        self.api_key = api_key
        self.base_url = base_url

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        log.info("OpenAI client initialized: base_url=%s", base_url)

    async def models(self) -> ModelsListResponse:
        models = self.client.models.list()
        data = models.data
        result = [ModelObject(created=m.created, id=m.id, owned_by=m.owned_by,
                              supported_modalities=remote_modalities(m)) for m in data]
        return ModelsListResponse(data=result)

    async def completions(self, body: completions_api.CompletionRequest, request: Request):
        """Forward a text-completions request to the remote OpenAI-compatible API."""
        remote = self.client.completions.create(
            model=body.model,
            prompt=body.prompt,
            stream=True,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            stop=body.stop,
        )

        def as_chat_chunks() -> Iterable[ChatCompletionChunk]:
            """Re-shape remote completion chunks into the chat-chunk form the responder consumes."""
            response_id = str(uuid.uuid4())
            for chunk in remote:
                yield ChatCompletionChunk(
                    id=chunk.id or response_id,
                    created=chunk.created or 0,
                    model=chunk.model or body.model or "",
                    object="chat.completion.chunk",
                    choices=[_chunk_choice(
                        index=choice.index,
                        delta=_chunk_delta(content=choice.text),
                        finish_reason=choice.finish_reason,
                    ) for choice in chunk.choices],
                )

        return await new_http_response_completions(is_stream_request(body), as_chat_chunks())

    async def chat_chunks(self, body: ChatCompletionRequest,
                           request: Request) -> Iterable[ChatCompletionChunk]:

        response_id = str(uuid.uuid4())

        body_model = body.model
        # Content parts (images, audio) are forwarded as they came: plain dicts keep the payload intact.
        messages = [message.model_dump(mode="json", exclude_none=True) for message in body.messages]
        # The proxy generator understands only streamed deltas, so the remote call always streams;
        # a client that asked for a plain response gets the chunks aggregated below instead.
        remote_stream = self.client.chat.completions.create(model=body_model,
                                                            stream=True,
                                                            stream_options={"include_usage": True},
                                                            messages=messages)

        def chunk_generator() -> Iterable[ChatCompletionChunk]:
            try:
                for chunk in remote_stream:
                    if chunk.usage:
                        yield new_usage_chunk(
                            response_id=response_id,
                            model=chunk.model or body_model,
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens)
                    if chunk.choices:
                        choice = chunk.choices[0]
                        delta = choice.delta
                        content = delta.content if hasattr(delta, 'content') and delta.content else None
                        has_reasoning = hasattr(delta, 'reasoning_content') and delta.reasoning_content
                        reasoning_content = delta.reasoning_content if has_reasoning else None
                        thinking = not reasoning_content is None
                        role = delta.role if hasattr(delta, 'role') else None

                        our_chunk = new_chat_completion_chunk(
                            response_id=response_id,
                            model=chunk.model,
                            content=content if content else reasoning_content,
                            thinking=thinking,
                            role=role,
                            finish_reason=choice.finish_reason
                        )

                        yield our_chunk

                        if choice.finish_reason:
                            log.debug("generation finished: finish_reason=%s", choice.finish_reason)
                            # break

            except Exception as e:  # pylint: disable=broad-exception-caught  # safety net: any failure of the remote stream must become an error stop-response
                log.error("inference error: %s", e, exc_info=e)
                yield new_stop_response(
                    content=str(e),
                    finish_reason="stop",
                    model=body_model,
                    role=ROLE_ASSISTANT,
                    response_id=response_id
                )

        return chunk_generator()

    def shutdown(self):
        """Clean up OpenAI client."""
        super().shutdown()
        # Note: AsyncOpenAI client cleanup is handled by garbage collection
        log.info("OpenAI controller shutdown")
