import asyncio
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Callable, Literal, Iterable

from fastapi.exceptions import RequestValidationError
from openai.types import FunctionDefinition
from openai.types.chat import ChatCompletionChunk, ChatCompletion, ChatCompletionToolUnionParam
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, Choice
from openvino_genai import ChatHistory
from openvino_genai import Tokenizer
from openvino_genai.py_openvino_genai import GenerationConfig
from pydantic import BaseModel
from starlette import status
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from agent import inference
from agent.client.tool_select_options import detect_select_options
from agent.client.user_context import UserContext
from agent.client.veai import is_veai_agent, get_veai_context, read_list_dir
from agent.client.veai.tool.list_dir import ListDir
from agent.client.veai.tool_call_fixer import veai_fix_tool_definition_optional_property_as_null_type
from agent.inference.token_handler import markdown_bold, markdown_back_tick
from agent.openai import GenerateOpts, completions_api
from agent.openai.chat_api import ROLE_TOOL, ROLE_ASSISTANT
from agent.openai.chat_api import new_chat_completion, new_tool_call, new_chat_completion_chunk
from agent.openai.chat_completions_api import ChatCompletionRequest, ChatCompletionMessageParam
from agent.openai.models_api import ModelsListResponse, ModelObject
from agent.parser import Parser
from agent.preprocess.tool_call import PreprocessToolCall

STOP: Literal["stop"] = "stop"
LENGTH: Literal["length"] = "length"

log = logging.getLogger(__name__)
log_client_generated = logging.getLogger(log.name + ".client.generated")

WARN_GENERATION_IS_INTERRUPTED_ = "Generating is interrupted."

USER_SELECT_CONTINUE = "continue"
USER_SELECT_INTERRUPT = "interrupt"
MIDDLEWARE_CHEKPOINT = "middleware_checkpoint"


class ControllerConfig(BaseModel):
    model_name: str
    max_prompt_len: int
    model_architectures: set[str]
    response_timeout: timedelta = timedelta(minutes=20)


def new_http_response(stream: bool,
                      chunk_generator: Iterable[ChatCompletionChunk]) -> StreamingResponse | ChatCompletion:
    if stream:
        return StreamingResponse(stream_generator(chunk_generator), media_type="text/event-stream")
    else:
        finish_reason, full_content, full_reasoning_content, full_tool_calls = make_union(chunk_generator)
        return new_chat_completion(
            finish_reason=finish_reason,
            content=full_content,
            reasoning_content=full_reasoning_content,
            tool_calls=full_tool_calls)


class BaseController(ABC):
    def __init__(self, config: ControllerConfig, parser: Parser, tokenizer: Tokenizer,
                 generate_config: GenerateOpts, is_fix_tool_type: bool, stop_signal: threading.Event,
                 chat_template: str = ''):
        self.parser = parser
        self.generate_config = generate_config
        self.config = config
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.log_inference_prompt = logging.getLogger(inference.log.name + ".prompt")
        self.log_inference_token_metrics = logging.getLogger(inference.log.name + ".token_metrics")
        self.log_inference = inference.log
        self.is_fix_tool_type = is_fix_tool_type
        self.closed = threading.Event()
        self.stop_signal = stop_signal

    def shutdown(self):
        self.closed.set()

    async def models(self) -> ModelsListResponse:
        current_time = int(time.time())
        return ModelsListResponse(data=[ModelObject(
            id=self.config.model_name,
            max_model_len=self.config.max_prompt_len,
            created=current_time,
        )])

    async def validation_exception_handler(self, request: Request, exc: RequestValidationError):
        log.error(f"request validation error: {exc.errors()}")

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors()},
        )

    def new_generation_config(self,
                              temperature: float | None,
                              max_completion_tokens: int | None,
                              max_prompt_tokens: int | None = None,
                              top_p: float | None = None,
                              frequency_penalty: float | None = None,
                              apply_chat_template: bool = False,
                              logprobs: bool | None = None,
                              stop: list[str] | str | None = None,
                              ) -> GenerationConfig:
        generation_config = GenerationConfig()
        max_new_tokens = max_completion_tokens or self.generate_config.max_new_tokens
        if max_new_tokens:
            generation_config.max_new_tokens = max_new_tokens
        max_length = max_prompt_tokens or self.generate_config.max_prompt_tokens
        if max_length:
            generation_config.max_length = max_length
        generation_config.apply_chat_template = apply_chat_template

        temp = temperature or self.generate_config.temperature
        if not temp or temp <= 0.0:
            # Greedy Search
            generation_config.do_sample = False
        else:
            generation_config.do_sample = True
            generation_config.temperature = temp
            generation_config.top_p = top_p or self.generate_config.top_p
            generation_config.top_k = self.generate_config.top_k
            generation_config.min_p = self.generate_config.min_p

            if frequency_penalty:
                generation_config.frequency_penalty = frequency_penalty
            else:
                frequency_penalty = self.generate_config.frequency_penalty
                if frequency_penalty:
                    generation_config.frequency_penalty = frequency_penalty

            if logprobs:
                generation_config.logprobs = 1

        repetition_penalty = self.generate_config.repetition_penalty
        if repetition_penalty:
            generation_config.repetition_penalty = repetition_penalty

        presence_penalty = self.generate_config.presence_penalty
        if presence_penalty:
            generation_config.presence_penalty = presence_penalty

        stop_set: set[str] = set(stop) if isinstance(stop, list) else {stop} if isinstance(stop, str) else set()
        generation_config.stop_strings = stop_set
        return generation_config

    async def chat(self, body: ChatCompletionRequest, request: Request):
        headers = request.headers
        host = headers.get("host")
        user_agent = headers.get("user-agent")
        x_device_id = headers.get("x-device-id")
        x_request_id = headers.get("x-request-id")

        log.debug(f"http request: host='{host}', user_agent='{user_agent}', "
                  f"x_device_id={x_device_id}, x_request_id={x_request_id}")

        stream = body.stream == True

        messages = body.messages
        tools = body.tools

        log.info(f"inbound history messages {len(messages)}")

        is_veai = is_veai_agent(messages)

        # if is_veai:
        #     for message in messages:
        #         if message.name == ListDir.name():
        #             dumped_dirs = read_list_dir(message)
        #             if dumped_dirs:
        #                 result_json = ListDir.new_result_json(dumped_dirs)
        #                 message.content = result_json
        #             pass

        user_context = get_veai_context(messages) if is_veai else UserContext()
        user_context.messages = messages
        user_context.model_architectures = self.config.model_architectures

        last_message = messages[-1] if messages else None

        if last_message:
            if is_middleware_checkpoint(last_message) and USER_SELECT_INTERRUPT in str(
                    last_message.content).lower():
                return new_http_response(stream, [
                    new_chat_completion_chunk(content="Interrupted", role=ROLE_ASSISTANT, finish_reason="stop")])
            elif last_message.role == ROLE_TOOL:
                log_client_generated.debug(last_message.content)

        invalid_response = self.validate_messages(messages, tools)
        if invalid_response:
            return new_http_response(stream, [invalid_response])

        tools_raw, function_by_name = group_function_by_name(tools, is_veai, self.is_fix_tool_type)

        tokenizer = self.tokenizer
        extra_context = {}
        model_parameters = self.generate_config.model_parameters
        if model_parameters:
            extra_context = model_parameters

        chat_history = new_chat_history(messages, tools_raw)
        history_get_messages = chat_history.get_messages()
        log.debug(f"chat history: messages={len(history_get_messages)}, tools={len(chat_history.get_tools())}, "
                  f"extra_context={extra_context}")

        full_prompt = tokenizer.apply_chat_template(history=chat_history,
                                                    tools=tools_raw,
                                                    add_generation_prompt=True,
                                                    extra_context=extra_context,
                                                    chat_template=self.chat_template)

        self.log_inference_prompt.debug(full_prompt)

        def is_stop():
            return self.stop_signal.is_set() or self.closed.is_set() or is_disconnected(request)

        chunk_generator = self.chunk_generator(
            prompt=full_prompt, generation_config=(
                self.new_generation_config(temperature=body.temperature,
                                           max_completion_tokens=(body.max_tokens or body.max_completion_tokens),
                                           top_p=body.top_p, frequency_penalty=body.frequency_penalty,
                                           logprobs=body.logprobs, stop=body.stop)), tokenizer=tokenizer,
            init_chat_events=True, is_stop=is_stop,
            is_veai=is_veai, user_context=user_context, function_by_name=function_by_name)
        return new_http_response(stream, chunk_generator)

    @abstractmethod
    def chunk_generator(self, prompt: str, generation_config: GenerationConfig,
                        tokenizer: Tokenizer, init_chat_events: bool, is_stop: Callable[[], bool], is_veai: bool,
                        function_by_name: dict[str, FunctionDefinition] | None = None,
                        user_context: UserContext | None = None,
                        ) -> Iterable[ChatCompletionChunk]:
        pass

    def validate_messages(self, messages, tools) -> ChatCompletionChunk | None:
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
                tool_call = new_tool_call(call_id=MIDDLEWARE_CHEKPOINT + "_" + str(uuid.uuid4()),
                                          function=question.to_openai_function_call())
                tool_calls.append(tool_call)
            else:
                content = (msg + "\n\n" + WARN_GENERATION_IS_INTERRUPTED_)
            return new_chat_completion_chunk(finish_reason=STOP,
                                             role=ROLE_ASSISTANT,
                                             content=content,
                                             tool_calls=tool_calls)
        return None

    async def completions(self, body: completions_api.CompletionRequest, request: Request):
        prompt = body.prompt
        if not prompt:
            prompt = ""

        self.log_inference_prompt.debug(prompt)

        generation_config = self.new_generation_config(temperature=body.temperature,
                                                       max_completion_tokens=body.max_tokens)
        response_id = str(uuid.uuid4())

        def is_stop():
            return self.stop_signal.is_set() or self.closed.is_set() or is_disconnected(request)

        stream = body.stream
        chunk_generator = self.chunk_generator(prompt=prompt, generation_config=generation_config,
                                               tokenizer=self.tokenizer, init_chat_events=True,
                                               is_stop=is_stop, is_veai=False)

        def chunk_converter(chunk_generator: Iterable[ChatCompletionChunk]) -> Iterable[
            completions_api.CompletionResponse]:
            def convert_response(r: ChatCompletionChunk) -> completions_api.CompletionResponse:
                return completions_api.CompletionResponse(model=r.model, id=r.id, choices=[
                    convert_choice(c) for c in r.choices])

            def convert_choice(chat_completion_choice: Choice) -> completions_api.CompletionChoice:
                delta = chat_completion_choice.delta
                content = delta.content if delta and delta.content else ""
                reason: Literal["stop"] | None = "stop" if chat_completion_choice.finish_reason else None
                return completions_api.CompletionChoice(text=content, finish_reason=reason)

            for c in chunk_generator:
                yield convert_response(c)

        if stream:
            return StreamingResponse(stream_generator(chunk_converter(chunk_generator)), media_type="text/event-stream")
        else:
            finish_reason, full_content, full_reasoning_content, full_tool_calls = make_union(chunk_generator)
            return new_chat_completion(response_id=response_id,
                                       finish_reason=finish_reason, content=full_content,
                                       reasoning_content=full_reasoning_content, tool_calls=full_tool_calls)

    def check_prompt_limit(self, max_length: int, encode_size: int, response_id: str) -> ChatCompletionChunk | None:
        if encode_size >= max_length:
            return new_chat_completion_chunk(response_id=response_id, role=ROLE_ASSISTANT,
                                             model=self.config.model_name, finish_reason=LENGTH,
                                             content=f"prompt exceeds limit: {encode_size} >= {max_length}")
        return None

    def get_tokens_size(self, prompt: str) -> int:
        encode_size = self.tokenizer.encode(prompt).input_ids.size
        return encode_size


def make_union(chunk_generator: Iterable[ChatCompletionChunk]) -> tuple[
    Literal["stop", "length", "tool_calls", "content_filter"], str, str, list[ChoiceDeltaToolCall]]:
    full_content = ""
    full_reasoning_content = ""
    full_tool_calls: list[ChoiceDeltaToolCall] = []
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] = "stop"

    for chunk_data in chunk_generator:
        choices = chunk_data.choices
        if choices:
            finish_reason = choices[-1].finish_reason or finish_reason
            for choice in choices:
                delta = choice.delta
                delta_content = delta.content
                if delta_content:
                    full_content += delta_content
                delta_reasoning_content = delta.reasoning_content
                if delta_reasoning_content:
                    full_reasoning_content += delta_reasoning_content
                delta_tool_calls = delta.tool_calls
                if delta_tool_calls:
                    full_tool_calls += delta_tool_calls
    return finish_reason, full_content, full_reasoning_content, full_tool_calls


def is_disconnected(request: Request) -> bool:
    disconnected = False
    try:
        loop = request.app.state.main_loop
        disconnected = asyncio.run_coroutine_threadsafe(request.is_disconnected(), loop).result(0.5)
        if disconnected:
            log.debug(f"disconnected http request")
    except asyncio.TimeoutError:
        pass
        # log.debug(f"disconnected http request check timeout")
    return disconnected


def new_chat_history(messages: list[ChatCompletionMessageParam],
                     tools_raw: list[dict[str, Any]] | None = None) -> ChatHistory:
    chat_history = ChatHistory()
    for message in messages:
        model_dump = message.model_dump()
        chat_history.append(model_dump)
    if tools_raw:
        chat_history.set_tools(tools_raw)
    return chat_history


def group_function_by_name(tools: list[ChatCompletionToolUnionParam] | None, is_veai: bool,
                           is_fix_tool_type: bool = False) -> tuple[
    list[dict[str, Any]], dict[str, FunctionDefinition]]:
    function_by_name: dict[str, FunctionDefinition] = {}
    tools_raw: list[dict[str, Any]] = []
    is_fix = is_veai and is_fix_tool_type
    for tool in (tools or []):
        tool_ = veai_fix_tool_definition_optional_property_as_null_type(tool) if is_fix else tool
        tools_raw.append(tool_.model_dump())
        function = tool.function
        function_by_name[function.name] = function
    return tools_raw, function_by_name


def stream_generator(chunk_generator: Iterable[BaseModel]) -> Iterable[str]:
    for chunk in chunk_generator:
        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"


def is_middleware_checkpoint(last_message: ChatCompletionMessageParam) -> str | None | bool:
    is_tool = last_message.role == ROLE_TOOL
    if not is_tool:
        return False
    tool_call_id = last_message.tool_call_id
    return tool_call_id and tool_call_id.startswith(MIDDLEWARE_CHEKPOINT)
