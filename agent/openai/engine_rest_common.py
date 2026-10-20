"""Shared plumbing of the REST engines: config, request preparation, generation config and streaming."""
import asyncio
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Literal, Iterable

from fastapi.exceptions import RequestValidationError
from openai.types.chat import ChatCompletionChunk, ChatCompletion
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, Choice
from openvino_genai import ChatHistory
from openvino_genai import Tokenizer
from openvino_genai.py_openvino_genai import GenerationConfig  # pylint: disable=no-name-in-module  # compiled extension module
from pydantic import BaseModel
from starlette import status
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from agent import inference
from agent.client.tool_select_options import detect_select_options
from agent.client.user_context import UserContext
from agent.client.veai import is_veai_agent, get_veai_context
from agent.client.veai.tool_call_fixer import veai_fix_tool_definition_optional_property_as_null_type, \
    veai_fix_incorrect_arguments
from agent.common.markdown import markdown_bold, markdown_back_tick
from agent.inference.token_handler import StopSignal, get_finish_str, TokenHandler, \
    TokenHandlerConfig
from agent.multimodal.content_parts import content_text
from agent.multimodal.media_error import MediaError
from agent.multimodal.media_inputs import MediaInputs, MediaLimits
from agent.multimodal.modality import Modalities, Modality
from agent.multimodal.request_media import prepare_messages
from agent.openai import GenerateOpts, completions_api
from agent.openai.chat_api import ROLE_TOOL, ROLE_ASSISTANT, new_stop_response
from agent.openai.chat_api import new_chat_completion, new_tool_call, new_chat_completion_chunk
from agent.openai.chat_completions_api import ChatCompletionRequest, ChatCompletionMessageParam, \
    ChatCompletionFunctionToolParam, FunctionDefinitionParameters
from agent.openai.interrupt import InterruptRegistry
from agent.openai.middleware_checkpoint import is_middleware_checkpoint, new_middleware_call_id
from agent.openai.models_api import ModelsListResponse, ModelObject
from agent.parser import Parser
from agent.preprocess.prompt_escape import PromptEscaper, MEDIA_TAGS_ONLY
from agent.preprocess.tool_call import PreprocessToolCall


STOP: Literal["stop"] = "stop"
LENGTH: Literal["length"] = "length"

log = logging.getLogger(__name__)
log_client_generated = logging.getLogger(log.name + ".client.generated")

WARN_GENERATION_IS_INTERRUPTED_ = "Generating is interrupted."

USER_SELECT_CONTINUE = "continue"
USER_SELECT_INTERRUPT = "interrupt"


class TokenizeRequest(BaseModel):
    """Request body of the /tokenize endpoint."""
    input: str


class DetokenizeRequest(BaseModel):
    """Request body of the /detokenize endpoint."""
    tokens: list[int]


class ControllerConfig(BaseModel):
    """Static per-model configuration shared by all endpoints."""
    model_name: str | None = None
    max_prompt_len: int | None = None
    model_architectures: set[str]
    response_timeout: timedelta = timedelta(minutes=20)
    is_fix_tool_type: bool = True
    is_detect_cycled_tool_call: bool = True
    chat_template: str = ''
    modalities: set[str] = {Modality.TEXT.value}
    max_media_bytes: int = 20 * 1024 * 1024
    max_media_items: int = 16
    allow_local_media_files: bool = False
    media_http_timeout: float = 30.0
    media_audio_sample_rate: int = 16000
    interrupt_completions: bool = False

    def modalities_of(self) -> Modalities:
        """Accepted input modalities of the configured model."""
        return Modalities(frozenset(Modality(name) for name in self.modalities | {Modality.TEXT.value}))

    def media_limits(self) -> MediaLimits:
        """Media size and count limits of a single request."""
        return MediaLimits(max_bytes=self.max_media_bytes, max_items=self.max_media_items,
                           allow_local_files=self.allow_local_media_files,
                           http_timeout=self.media_http_timeout,
                           audio_sample_rate=self.media_audio_sample_rate)


def is_stream_request(body: completions_api.CompletionRequest | ChatCompletionRequest) -> bool:
    """Return True when the client asked for a streaming response."""
    return body.stream is True


async def new_http_response_chat(stream: bool,
                                 chunk_generator: Iterable[ChatCompletionChunk]) -> StreamingResponse | ChatCompletion:
    """Wrap chunks into an SSE streaming or aggregated chat completion response."""
    if stream:
        return StreamingResponse(stream_generator(chunk_generator), media_type="text/event-stream")
    # Aggregate off the event loop: pumping the inference generator here stalls the loop, and the
    # per-token is_disconnected() check (0.5s timeout) then throttles every token of the generation.
    return await asyncio.to_thread(new_chat_completion_union, chunk_generator)


async def new_http_response_completions(stream: bool,
                                        chunk_generator: Iterable[ChatCompletionChunk]) -> (
        StreamingResponse | ChatCompletion | completions_api.CompletionResponse):
    """Wrap chunks into an SSE streaming or aggregated text-completions response."""
    def chunk_converter(chunk_generator: Iterable[ChatCompletionChunk]) -> Iterable[
        completions_api.CompletionResponse]:
        """Convert chat chunks into text-completion responses."""
        def convert_response(r: ChatCompletionChunk) -> completions_api.CompletionResponse:
            """Map one chat chunk to a completions response."""
            return completions_api.CompletionResponse(model=r.model, id=r.id, choices=[
                convert_choice(c) for c in r.choices])

        def convert_choice(chat_completion_choice: Choice) -> completions_api.CompletionChoice:
            """Map one chat choice to a text choice carrying the delta content."""
            delta = chat_completion_choice.delta
            content = delta.content if delta and delta.content else ""
            reason: Literal["stop"] | None = "stop" if chat_completion_choice.finish_reason else None
            return completions_api.CompletionChoice(text=content, finish_reason=reason)

        for c in chunk_generator:
            yield convert_response(c)

    if stream:
        return StreamingResponse(stream_generator(chunk_converter(chunk_generator)), media_type="text/event-stream")

    def aggregate() -> completions_api.CompletionResponse:
        # completions endpoint must answer the completions schema (choices[].text), not the chat one
        finish_reason, full_content, _, _ = make_union(chunk_generator)
        return completions_api.CompletionResponse(id=str(uuid.uuid4()), created=int(time.time()),
                                                  choices=[completions_api.CompletionChoice(
                                                      text=full_content,
                                                      finish_reason="stop" if finish_reason else None)])

    # same as new_http_response_chat: keep the event loop free while the generator is consumed
    response: completions_api.CompletionResponse = await asyncio.to_thread(aggregate)
    return response


def new_chat_completion_union(chunk_generator: Iterable[ChatCompletionChunk]) -> ChatCompletion:
    """Aggregate a chunk stream into a single non-streaming ChatCompletion."""
    finish_reason, full_content, full_reasoning_content, full_tool_calls = make_union(chunk_generator)
    response_id = str(uuid.uuid4())
    return new_chat_completion(response_id=response_id, finish_reason=finish_reason, content=full_content,
                               reasoning_content=full_reasoning_content, tool_calls=full_tool_calls)


@dataclass
class GenerationRequest:  # pylint: disable=too-many-instance-attributes  # flat DTO of the request fields
    """Generation parameters of a single request, as sent by the client.

    A request overrides the server defaults only for the fields it actually sets;
    every unset field falls back to the matching field of :class:`GenerateOpts`.
    """

    temperature: float | None = None
    max_completion_tokens: int | None = None
    max_prompt_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    apply_chat_template: bool = False
    logprobs: bool | None = None
    stop: list[str] | str | None = None

    @staticmethod
    def of_chat_request(body: ChatCompletionRequest) -> "GenerationRequest":
        """Generation parameters of a chat completion request."""
        return GenerationRequest(temperature=body.temperature,
                                 max_completion_tokens=body.max_tokens or body.max_completion_tokens,
                                 top_p=body.top_p,
                                 frequency_penalty=body.frequency_penalty,
                                 logprobs=body.logprobs,
                                 stop=body.stop)

    @staticmethod
    def of_completion_request(body: completions_api.CompletionRequest) -> "GenerationRequest":
        """Generation parameters of a legacy text completion request."""
        return GenerationRequest(temperature=body.temperature, max_completion_tokens=body.max_tokens)

    def stop_strings(self) -> set[str]:
        """Stop phrases of the request, normalized to a set."""
        stop = self.stop
        if isinstance(stop, list):
            return set(stop)
        if isinstance(stop, str):
            return {stop}
        return set()


def new_generation_config(generate_opts: GenerateOpts, request: GenerationRequest) -> GenerationConfig:
    """Translate the request parameters and the server defaults into a GenerationConfig.

    The request wins over the server defaults; the defaults fill in what the request left unset.
    """
    generation_config = GenerationConfig()
    _apply_generate_opts(generation_config, generate_opts)
    _apply_request(generation_config, request)
    return generation_config


def _apply_generate_opts(generation_config: GenerationConfig, generate_opts: GenerateOpts) -> None:
    """Copy the server defaults into the generation config.

    Only the options that are actually set are copied, so that the defaults of
    the GenerationConfig itself stay in place for the rest.
    """
    generation_config.do_sample = generate_opts.do_sample
    _set_if_set(generation_config, "max_new_tokens", generate_opts.max_new_tokens)
    _set_if_set(generation_config, "max_length", generate_opts.max_prompt_tokens)
    _set_if_set(generation_config, "temperature", generate_opts.temperature)
    _set_if_set(generation_config, "top_p", generate_opts.top_p)
    _set_if_set(generation_config, "top_k", generate_opts.top_k)
    _set_if_set(generation_config, "min_p", generate_opts.min_p)
    _set_if_set(generation_config, "repetition_penalty", generate_opts.repetition_penalty)
    _set_if_set(generation_config, "frequency_penalty", generate_opts.frequency_penalty)
    _set_if_set(generation_config, "presence_penalty", generate_opts.presence_penalty)
    _set_if_set(generation_config, "num_assistant_tokens", generate_opts.num_assistant_tokens)
    _set_if_set(generation_config, "tree_depth", generate_opts.tree_depth)
    _set_if_set(generation_config, "branching_factor", generate_opts.branching_factor)
    _set_if_set(generation_config, "max_ngram_size", generate_opts.max_ngram_size)
    _set_if_set(generation_config, "num_return_sequences", generate_opts.num_return_sequences)
    _set_if_set(generation_config, "num_beams", generate_opts.num_beams)
    _set_if_set(generation_config, "num_beam_groups", generate_opts.num_beam_groups)
    _set_if_set(generation_config, "diversity_penalty", generate_opts.diversity_penalty)
    _set_if_set(generation_config, "length_penalty", generate_opts.length_penalty)
    _set_if_set(generation_config, "no_repeat_ngram_size", generate_opts.no_repeat_ngram_size)


def _set_if_set(generation_config: GenerationConfig, name: str, value) -> None:
    """Copy the value into the generation config unless it is unset."""
    if not value is None:
        setattr(generation_config, name, value)


def _apply_request(generation_config: GenerationConfig, request: GenerationRequest) -> None:
    """Override the generation config with the parameters of the request."""
    if request.max_completion_tokens:
        generation_config.max_new_tokens = request.max_completion_tokens
    if request.max_prompt_tokens:
        generation_config.max_length = request.max_prompt_tokens
    if request.temperature:
        generation_config.temperature = request.temperature
    if request.top_p:
        generation_config.top_p = request.top_p
    if not request.frequency_penalty is None:
        generation_config.frequency_penalty = request.frequency_penalty
    if request.logprobs:
        generation_config.logprobs = 1
    generation_config.apply_chat_template = request.apply_chat_template
    generation_config.stop_strings = request.stop_strings()


def get_tokens_size(tokenizer: Tokenizer, prompt: str) -> int:
    """Return the number of tokens the tokenizer produces for a prompt."""
    encode_size = tokenizer.encode(prompt).input_ids.size
    return encode_size


class BaseController(ABC):  # pylint: disable=too-many-instance-attributes  # dependency bundle of shared state
    """Endpoint-independent plumbing shared by all engines."""
    def __init__(self, config: ControllerConfig,
                 handler_config: TokenHandlerConfig,
                 generate_opts: GenerateOpts,
                 stop_signal: threading.Event):
        """Store config, handlers and the stop signal shared by the endpoints."""
        # self.parser = parser
        self.generate_opts = generate_opts
        self.handler_config = handler_config
        self.config = config
        # self.tokenizer = tokenizer
        self.log_inference_prompt = logging.getLogger(inference.log.name + ".prompt")
        self.log_inference_token_metrics = logging.getLogger(inference.log.name + ".token_metrics")
        self.log_inference = inference.log
        self.closed = threading.Event()
        self.stop_signal = stop_signal
        self.interrupt_registry = InterruptRegistry()

    def begin_interruptible_request(self) -> threading.Event | None:
        """Start an interruptible request.

        Returns the stop event of the new request when the interrupt mode is on: a newer
        request sets the event of the running one, so the older generation stops and
        releases the device. Returns None when the mode is off.
        """
        if not self.config.interrupt_completions:
            return None
        return self.interrupt_registry.start()

    def guard_generation(self, chunk_generator: Iterable[ChatCompletionChunk],
                         interrupt_event: threading.Event) -> Iterable[ChatCompletionChunk]:
        """Wrap the chunk generator so the interrupt slot is released when it ends."""
        def guarded() -> Iterable[ChatCompletionChunk]:
            try:
                yield from chunk_generator
            finally:
                self.interrupt_registry.finish(interrupt_event)

        return guarded()

    async def health(self) -> JSONResponse:
        """Simple health check — always returns 200 OK."""
        return JSONResponse(content={"status": "ok"})

    def shutdown(self):
        """Mark the controller closed so stop checks start returning True."""
        self.closed.set()

    # `request` is part of the starlette exception-handler contract even though unused.
    def validation_exception_handler(self, request: Request,  # pylint: disable=unused-argument
                                     exc: RequestValidationError):
        """Convert request validation errors into a 422 JSON response."""
        log.error("request validation error: %s", exc.errors())

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors()},
        )

    def validate_messages(self, messages, tools) -> ChatCompletionChunk | None:
        """Return an intervention chunk for cycled tool calls, or None when the request is fine."""
        if self.config.is_detect_cycled_tool_call:
            request_user_select = detect_select_options(tools)
            preprocess_tool_call = PreprocessToolCall()
            looped_function, count = preprocess_tool_call.check_loop_tool_calls(messages)
            if looped_function:
                # log
                msg = looped_function.render_markdown()
                tool_calls = []
                content = ""
                if request_user_select:
                    # log
                    question = request_user_select.new_call(
                        msg +
                        "\n\n" +
                        "Repeated: " + markdown_back_tick(str(count) + " " + ("time" if count == 1 else "times")) +
                        "\n\n" + markdown_bold("What to do next?"),
                        [USER_SELECT_CONTINUE, USER_SELECT_INTERRUPT])
                    tool_call = new_tool_call(call_id=new_middleware_call_id(),
                                              function=question.to_openai_function_call())
                    tool_calls.append(tool_call)
                else:
                    content = msg + "\n\n" + WARN_GENERATION_IS_INTERRUPTED_
                return new_chat_completion_chunk(finish_reason=STOP,
                                                 role=ROLE_ASSISTANT,
                                                 content=content,
                                                 tool_calls=tool_calls)
        return None

    async def chat(self, body: ChatCompletionRequest, request: Request):
        """Shared /v1/chat/completions entry: checkpoint guard, validation, dispatch."""
        headers = request.headers
        host = headers.get("host")
        user_agent = headers.get("user-agent")
        x_device_id = headers.get("x-device-id")
        x_request_id = headers.get("x-request-id")

        log.debug("http request: host='%s', user_agent='%s', x_device_id=%s, x_request_id=%s",
                  host, user_agent, x_device_id, x_request_id)

        stream = is_stream_request(body)
        messages = body.messages
        log.info("inbound history messages %s", len(messages))

        last_message = messages[-1] if messages else None
        if last_message:
            if is_middleware_checkpoint(last_message) and USER_SELECT_INTERRUPT in \
                    content_text(last_message.content).lower():
                return await new_http_response_chat(stream, [
                    new_chat_completion_chunk(content="Interrupted", role=ROLE_ASSISTANT, finish_reason="stop")])
            if last_message.role == ROLE_TOOL:
                log_client_generated.debug(content_text(last_message.content))

        invalid_response = self.validate_messages(messages, body.tools)
        return (await new_http_response_chat(stream, [invalid_response])) if invalid_response \
            else await self.handle_chat_completion(body, request)

    @abstractmethod
    async def handle_chat_completion(self, body: ChatCompletionRequest,
                                     request: Request) -> StreamingResponse | ChatCompletion:
        """Produce chunks for a chat completion request; implemented per engine."""

    @abstractmethod
    async def completions(self, body: completions_api.CompletionRequest, request: Request):
        """Produce chunks for a legacy text completion request; implemented per engine."""

    @abstractmethod
    async def models(self) -> ModelsListResponse:
        """List the served models."""


class BaseOVController(BaseController):
    """Base for OpenVINO-backed engines: prompt building and shared endpoints."""

    # Constructor wires the OpenVINO engine context; grouping into a config is a separate refactor.
    def __init__(self, config: ControllerConfig, parser: Parser, tokenizer: Tokenizer,  # pylint: disable=R0913,R0917
                 handler_config: TokenHandlerConfig,
                 generate_opts: GenerateOpts, stop_signal: threading.Event):
        """Wire parser, tokenizer and the shared controller state."""
        super().__init__(config, handler_config, generate_opts, stop_signal)
        self.parser = parser
        # self.generate_opts = generate_opts
        # self.handler_config = handler_config
        # self.config = config
        self.tokenizer = tokenizer
        self.prompt_escaper = PromptEscaper.of_tokenizer(tokenizer)

        # self.log_inference_prompt = logging.getLogger(inference.log.name + ".prompt")
        # self.log_inference_token_metrics = logging.getLogger(inference.log.name + ".token_metrics")
        # self.log_inference = inference.log
        # self.closed = threading.Event()
        # self.stop_signal = stop_signal

    async def models(self) -> ModelsListResponse:
        """Report the single configured model with its modalities."""
        current_time = int(time.time())
        return ModelsListResponse(data=[ModelObject(
            id=self.config.model_name,
            max_model_len=self.config.max_prompt_len,
            supported_modalities=self.config.modalities_of().names(),
            created=current_time,
        )])

    async def tokenize(self, request: TokenizeRequest) -> JSONResponse:
        """tokenize endpoint — tokenizes text and returns token IDs."""
        tokens = self.tokenizer.encode(request.input).input_ids.data[0].tolist()
        return JSONResponse(content={"tokens": tokens})

    async def detokenize(self, request: DetokenizeRequest) -> JSONResponse:
        """detokenize endpoint — detokenizes token IDs back to text."""
        text = self.tokenizer.decode(request.tokens).strip()
        return JSONResponse(content={"text": text})

    async def handle_chat_completion(self, body: ChatCompletionRequest,
                                     request: Request) -> StreamingResponse | ChatCompletion:
        """Run the full local pipeline: prepare messages, build prompt, generate chunks."""
        stream = is_stream_request(body)
        messages: list[ChatCompletionMessageParam] = body.messages
        is_veai = is_veai_agent(messages)
        tools_raw, function_parameters = self.prepare_tools(body.tools, is_veai)

        try:
            prepared = await asyncio.to_thread(
                prepare_messages, messages, self.config.modalities_of(), self.config.media_limits(),
                self.prompt_escaper)
        except MediaError as e:
            log.info("multimodal request rejected: %s", e)
            return await new_http_response_chat(stream, [new_chat_completion_chunk(
                response_id=str(uuid.uuid4()), role=ROLE_ASSISTANT, model=self.config.model_name or "",
                content=str(e), finish_reason=STOP)])

        prompt = self.build_prompt(prepared.messages, tools_raw)
        generation_config = new_generation_config(self.generate_opts,
                                                  GenerationRequest.of_chat_request(body))
        over_limit_response = self.check_prompt_of(prompt, generation_config, prepared.media)
        if over_limit_response:
            return await new_http_response_chat(stream, [over_limit_response])

        token_handler = self.new_token_handler(prompt, messages, is_veai, function_parameters, request)
        chunk_generator = self.chunk_generator(prompt=prompt, generation_config=generation_config,
                                               token_handler=token_handler, media=prepared.media)
        return await new_http_response_chat(stream, chunk_generator)

    def check_prompt_of(self, prompt: str, generation_config: GenerationConfig,
                        media: MediaInputs) -> ChatCompletionChunk | None:
        """Check the size of the prompt and log the parameters of the generation."""
        prompt_tokens_amount = get_tokens_size(self.tokenizer, prompt)
        over_limit_response = self.check_prompt_limit(max_length=generation_config.max_length,
                                                      encode_size=prompt_tokens_amount,
                                                      response_id=str(uuid.uuid4()))
        if over_limit_response:
            return over_limit_response
        self.log_inference_start(prompt_tokens_amount, generation_config, media)
        return None

    def prepare_tools(self, tools, is_veai: bool) -> tuple[list[dict[str, Any]],
                                                           dict[str, FunctionDefinitionParameters]]:
        """Split the tools of the request into their raw form and their function parameters."""
        tools_raw, function_parameters = get_function_parameters_by_name(tools, is_veai,
                                                                        self.config.is_fix_tool_type)
        return self.prompt_escaper.escape_in(tools_raw), function_parameters

    def build_prompt(self, messages: list[dict], tools_raw: list[dict[str, Any]]) -> str:
        """Render the prompt of the request out of its messages and tools."""
        extra_context = self.generate_opts.model_parameters or {}
        chat_history = new_chat_history(messages, tools_raw)
        log.debug("chat history: messages=%s, tools=%s, extra_context=%s",
                  len(chat_history.get_messages()), len(chat_history.get_tools()), extra_context)
        prompt = self.tokenizer.apply_chat_template(history=chat_history,
                                                    tools=tools_raw,
                                                    add_generation_prompt=True,
                                                    extra_context=extra_context,
                                                    chat_template=self.config.chat_template)
        self.log_inference_prompt.debug(prompt)
        return prompt

    def log_inference_start(self, prompt_tokens_amount: int, generation_config: GenerationConfig,
                            media: MediaInputs) -> None:
        """Report the parameters of the generation before it starts."""
        if self.log_inference.isEnabledFor(logging.DEBUG):
            self.log_inference.debug(
                "inference start: "
                "prompt_tokens_amount=%s, "
                "do_sample=%s, "
                "max_length=%s, "
                "max_new_tokens=%s, "
                "temperature=%.2f, "
                "top_p=%.2f, top_k=%s, "
                "min_p=%.2f, repetition_penalty=%.2f, "
                "presence_penalty=%.2f, "
                "frequency_penalty=%.2f, "
                "media='%s'",
                prompt_tokens_amount, generation_config.do_sample, generation_config.max_length,
                generation_config.max_new_tokens, generation_config.temperature, generation_config.top_p,
                generation_config.top_k, generation_config.min_p, generation_config.repetition_penalty,
                generation_config.presence_penalty, generation_config.frequency_penalty, media.summary())
        else:
            self.log_inference.info("inference start, media='%s'", media.summary())

    def new_token_handler(self, prompt: str, messages: list[ChatCompletionMessageParam], is_veai: bool,
                          function_parameters: dict[str, FunctionDefinitionParameters],
                          request: Request) -> TokenHandler:
        """Build the token handler of the request, wired to the stop conditions."""
        is_client_gone = disconnect_probe(request)

        def is_stop():
            return self.stop_signal.is_set() or self.closed.is_set() or is_client_gone()

        user_context = (get_veai_context(messages) or UserContext()) if is_veai else UserContext()
        user_context.messages = messages
        user_context.model_architectures = self.config.model_architectures
        return TokenHandler(tokenizer=self.tokenizer,
                            prompt=prompt,
                            parser=self.parser,
                            init_chat_events=True,
                            is_stop=is_stop,
                            config=self.handler_config,
                            tool_fixer=(veai_fix_incorrect_arguments if is_veai else None),
                            user_context=user_context,
                            supported_functions=function_parameters,
                            )

    @abstractmethod
    def chunk_generator(self, prompt: str, generation_config: GenerationConfig, token_handler: TokenHandler,
                        media: MediaInputs) -> Iterable[ChatCompletionChunk]:
        """Run generation for the prepared prompt and stream chat completion chunks.

        Args:
            prompt: rendered prompt text, may contain OpenVINO media tags.
            generation_config: generation parameters of this call.
            token_handler: streaming token handler of the response.
            media: decoded image/video/audio inputs referenced by the prompt tags.
        """

    async def completions(self, body: completions_api.CompletionRequest, request: Request):
        """Run the legacy completions pipeline on top of the same chunk generator."""
        # a raw prompt is the client's own template, so special tokens are intended there,
        # but no media is attached to it and a media tag would only fail generation
        prompt = MEDIA_TAGS_ONLY.escape(body.prompt or "")

        self.log_inference_prompt.debug(prompt)

        # in the interrupt mode a new request stops the running one, so the device is
        # freed for the fresh request instead of being shared by two generations
        interrupt_event = self.begin_interruptible_request()

        generation_config = new_generation_config(self.generate_opts,
                                                  GenerationRequest.of_completion_request(body))

        token_handler = self.new_completion_token_handler(prompt, request, interrupt_event)

        stream = is_stream_request(body)
        prompt_tokens_amount = get_tokens_size(self.tokenizer, prompt)
        over_limit_response = self.check_prompt_limit(max_length=generation_config.max_length,
                                                      encode_size=prompt_tokens_amount,
                                                      response_id=str(uuid.uuid4()))
        if over_limit_response:
            return await new_http_response_chat(stream, [over_limit_response])

        chunk_generator = self.chunk_generator(prompt=prompt, generation_config=generation_config,
                                               token_handler=token_handler, media=MediaInputs.empty())
        if interrupt_event is not None:
            chunk_generator = self.guard_generation(chunk_generator, interrupt_event)
        return await new_http_response_completions(stream, chunk_generator)

    def new_completion_token_handler(self, prompt: str, request: Request,
                                     interrupt_event: threading.Event | None) -> TokenHandler:
        """Build the token handler of a completion request, wired to the stop conditions."""
        is_client_gone = disconnect_probe(request)

        def is_stop():
            return (self.stop_signal.is_set() or self.closed.is_set() or is_client_gone()
                    or (interrupt_event is not None and interrupt_event.is_set()))

        return TokenHandler(tokenizer=self.tokenizer,
                            prompt=prompt,
                            parser=self.parser,
                            init_chat_events=True,
                            is_stop=is_stop,
                            config=self.handler_config,
                            )

    def check_prompt_limit(self, max_length: int, encode_size: int, response_id: str) -> ChatCompletionChunk | None:
        """Return a length-stop chunk when the prompt does not fit, else None."""
        if encode_size >= max_length:
            return new_chat_completion_chunk(response_id=response_id, role=ROLE_ASSISTANT,
                                             model=self.config.model_name or "", finish_reason=LENGTH,
                                             content=f"prompt exceeds limit: {encode_size} >= {max_length}")
        return None


def make_union(chunk_generator: Iterable[ChatCompletionChunk]) -> tuple[
    Literal["stop", "length", "tool_calls", "content_filter"], str, str, list[ChoiceDeltaToolCall]]:
    """Aggregate chunks into finish reason, content, reasoning content and tool calls."""
    full_content = ""
    full_reasoning_content = ""
    full_tool_calls: list[ChoiceDeltaToolCall] = []
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] = "stop"

    for chunk_data in chunk_generator:
        choices = chunk_data.choices
        if choices:
            raw_reason = choices[-1].finish_reason
            # legacy remote chunks finish tool calls with 'function_call' — same meaning as tool_calls
            finish_reason = "tool_calls" if raw_reason == "function_call" else (raw_reason or finish_reason)
            for choice in choices:
                delta = choice.delta
                delta_content = delta.content
                if delta_content:
                    full_content += delta_content
                # reasoning_content is an optional extra field (openai models use extra="allow"):
                # plain deltas never carry it and pydantic __getattr__ raises AttributeError
                delta_reasoning_content = getattr(delta, "reasoning_content", None)
                if delta_reasoning_content:
                    full_reasoning_content += delta_reasoning_content
                delta_tool_calls = delta.tool_calls
                if delta_tool_calls:
                    full_tool_calls += delta_tool_calls
    return finish_reason, full_content, full_reasoning_content, full_tool_calls


def is_disconnected(request: Request) -> bool:
    """Check with a short timeout whether the client connection is gone."""
    disconnected = False
    try:
        loop = request.app.state.main_loop
        disconnected = asyncio.run_coroutine_threadsafe(request.is_disconnected(), loop).result(0.1)
        if disconnected:
            log.debug("disconnected http request")
    except asyncio.TimeoutError:
        pass
        # log.debug(f"disconnected http request check timeout")
    return disconnected


def disconnect_probe(request: Request, interval: float = 0.25) -> Callable[[], bool]:
    """Throttled client-disconnect probe for the per-token stop check.

    is_disconnected() costs a roundtrip to the event loop; probing it on every token lets a busy
    loop stall the generation. The probe checks at most once per `interval` and latches the first
    positive result. stop_signal/closed are checked by the caller on every call regardless.
    """
    next_check = 0.0
    disconnected = False

    def probe() -> bool:
        nonlocal next_check, disconnected
        if disconnected:
            return True
        now = time.monotonic()
        if now < next_check:
            return False
        next_check = now + interval
        disconnected = is_disconnected(request)
        return disconnected

    return probe


def new_chat_history(messages: list[ChatCompletionMessageParam] | list[dict],
                     tools_raw: list[dict[str, Any]] | None = None) -> ChatHistory:
    """Build an openvino_genai.ChatHistory from request messages and tools."""
    chat_history = ChatHistory()
    for message in messages:
        model_dump = message.model_dump() if hasattr(message, "model_dump") else message
        chat_history.append(model_dump)
    if tools_raw:
        chat_history.set_tools(tools_raw)
    return chat_history


def is_function_tool(tool: Any) -> bool:
    """Return True when the tool entry is a function tool."""
    return getattr(tool, "type", None) == "function"


def get_function_parameters_by_name(tools: list[ChatCompletionFunctionToolParam] | None, is_veai: bool,
                                    is_fix_tool_type: bool = False) -> tuple[
    list[dict[str, Any]], dict[str, FunctionDefinitionParameters]]:
    """Split tools into raw dicts and a name-to-parameters map, applying the veai fix when asked."""
    function_parameters: dict[str, FunctionDefinitionParameters] = {}
    tools_raw: list[dict[str, Any]] = []
    is_fix = is_veai and is_fix_tool_type
    for tool in (tools or []):
        tool_ = veai_fix_tool_definition_optional_property_as_null_type(tool) if is_fix else tool
        tools_raw.append(tool_.model_dump(mode="json"))
        if is_function_tool(tool_):
            function = tool_.function
            parameters = function.parameters
            if parameters:
                # parameters = deepcopy(parameters)
                # for param_desc in parameters.values():
                #     if "description" in param_desc:
                #         del param_desc["description"]
                function_name = function.name
                function_parameters[function_name] = parameters
    return tools_raw, function_parameters


def stream_generator(chunk_generator: Iterable[BaseModel]) -> Iterable[str]:
    """Serialize chunks as SSE data lines."""
    for chunk in chunk_generator:
        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"


def add_stop_signal(responses: list[ChatCompletionChunk], stop_signal: StopSignal):
    """Attach the finish reason to the last choice or append a final stop response."""
    choices = responses[-1].choices if responses else None
    finish_reason = get_finish_str(stop_signal)
    if choices:
        last_choice = choices[-1]
        last_choice.finish_reason = finish_reason
        log.debug("add finish_reason to the last choice, finish_reason=%s", finish_reason)
    else:
        log.debug("add stop message at the end, finish_reason=%s", finish_reason)
        responses.append(new_stop_response(finish_reason=finish_reason, role=None))
    return responses
