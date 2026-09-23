import importlib
import logging
import threading
import time
import uuid
from typing import Iterable

# Import openai package explicitly to avoid conflict with local openai folder
_openai_pkg = importlib.import_module('openai')
openai = _openai_pkg
_openai_types_chat = importlib.import_module('openai.types.chat')
ChatCompletionChunk = _openai_types_chat.ChatCompletionChunk
from starlette import status
from starlette.requests import Request
from starlette.responses import JSONResponse

from agent.common.metric_mem import get_current_memory
from agent.inference.token_handler import TokenHandler, TokenHandlerConfig, StopSignal
from agent.openai import GenerateOpts
from agent.openai.chat_api import new_stop_response, ROLE_ASSISTANT, new_chat_completion_chunk
from agent.openai.engine_rest_common import BaseController, ControllerConfig, add_stop_signal, get_tokens_size
from agent.parser import Parser

log = logging.getLogger(__name__)


class OpenAiController(BaseController):
    """
    OpenAI-compatible controller that uses OpenAI Python SDK to call
    remote/inference endpoints without requiring OpenVINO libraries.
    
    This enables running on machines without GPU (e.g., macOS) where
    OpenVINO cannot load native accelerators.
    """

    def __init__(self, config: ControllerConfig, parser: Parser, api_key: str,
                 base_url: str, handler_config: TokenHandlerConfig,
                 stop_signal: threading.Event, generate_opts: GenerateOpts,
                 model_name_override: str | None = None):
        """
        Initialize OpenAI controller.
        
        Args:
            config: Controller configuration (model_name, max_prompt_len, etc.)
            parser: Parser for model-specific token handling
            api_key: OpenAI API key for authentication
            base_url: OpenAI-compatible API endpoint URL
            handler_config: Token handler configuration
            stop_signal: Threading event for stopping inference
            generate_opts: Generation options (temperature, max_tokens, etc.)
            model_name_override: Optional override for model name in responses
        """
        super().__init__(config, parser, None, handler_config, generate_opts, stop_signal)
        
        self.api_key = api_key
        self.base_url = base_url
        self.model_name_override = model_name_override or config.model_name
        
        # Initialize OpenAI client
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        log.info(f"OpenAI client initialized: base_url={base_url}, model={self.model_name_override}")

    def chunk_generator(self, prompt: str, generation_config,
                        token_handler: TokenHandler) -> Iterable[ChatCompletionChunk]:
        """
        Generate chunks using OpenAI API.
        
        This method:
        1. Prepares the chat completion request
        2. Calls the OpenAI API with streaming
        3. Yields ChatCompletionChunk objects for SSE delivery
        """
        response_id = str(uuid.uuid4())
        
        # Get prompt token count (estimate)
        prompt_tokens_amount = len(prompt.split()) if self.tokenizer is None else get_tokens_size(self.tokenizer, prompt)
        max_length = generation_config.max_length if hasattr(generation_config, 'max_length') else None
        
        over_limit_response = self.check_prompt_limit(max_length=max_length, encode_size=prompt_tokens_amount,
                                                      response_id=response_id)
        if over_limit_response:
            yield over_limit_response
            return

        before_generate_mem = get_current_memory()
        
        log.info(f"inference start: request={response_id}, model={self.model_name_override}")
        
        # Prepare chat completion parameters
        temperature = generation_config.temperature if hasattr(generation_config, 'temperature') else None
        max_completion_tokens = generation_config.max_new_tokens if hasattr(generation_config, 'max_new_tokens') else None
        top_p = generation_config.top_p if hasattr(generation_config, 'top_p') else None
        frequency_penalty = generation_config.frequency_penalty if hasattr(generation_config, 'frequency_penalty') else None
        stop = generation_config.stop_strings if hasattr(generation_config, 'stop_strings') else None
        
        # Convert stop_strings to list if needed
        if isinstance(stop, set):
            stop = list(stop)
        
        # Build the chat completion request
        chat_completion_params = {
            "model": self.model_name_override,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_completion_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "stop": stop,
        }
        
        # Remove None values
        chat_completion_params = {k: v for k, v in chat_completion_params.items() if v is not None}
        
        log.debug(f"chat completion request: params={chat_completion_params}")
        
        try:
            # Call OpenAI API with streaming
            stream = self.client.chat.completions.create(**chat_completion_params)
            
            # Process stream chunks
            for chunk in stream:
                # Convert OpenAI chunk to our ChatCompletionChunk format
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    # Create our ChatCompletionChunk
                    our_chunk = new_chat_completion_chunk(
                        response_id=response_id,
                        model=self.model_name_override,
                        content=delta.content if hasattr(delta, 'content') and delta.content else None,
                        role="assistant" if hasattr(delta, 'role') and delta.role else None,
                        finish_reason=choice.finish_reason
                    )
                    
                    yield our_chunk
                    
                    # Handle stop signal
                    if choice.finish_reason:
                        log.debug(f"generation finished: finish_reason={choice.finish_reason}")
                        break
                        
        except Exception as e:
            log.error(f"inference error: {e}", exc_info=e)
            yield new_stop_response(
                content=str(e),
                finish_reason="stop",
                model=self.model_name_override,
                role=ROLE_ASSISTANT,
                response_id=response_id
            )
        
        after_generate_mem = get_current_memory()
        generate_cost = after_generate_mem - before_generate_mem
        log.debug(f"consumed memory: {after_generate_mem:.2f} MB, generate delta: {generate_cost:.2f} MB")
        log.info(f"inference finished: request={response_id}")

    def slots_sync(self) -> JSONResponse:
        """Sync version of slots for testing."""
        return JSONResponse(content={
            "slots": [],
            "kv_cache_size_mb": 0,
            "cache_size_mb": 0,
            "cache_usage": 0,
            "max_cache_usage": 0,
            "cache_utilization_pct": 0,
            "active_requests": 0,
            "scheduled_requests": 0,
            "active_slots": 0
        })

    def shutdown(self):
        """Clean up OpenAI client."""
        super().shutdown()
        # Note: AsyncOpenAI client cleanup is handled by garbage collection
        log.info("OpenAI controller shutdown")