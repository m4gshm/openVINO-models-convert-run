import asyncio
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Callable, Literal, Iterable

from openvino_genai import ChatHistory
from openvino_genai import Tokenizer
from openvino_genai.py_openvino_genai import GenerationConfig
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import StreamingResponse

from agent import inference
from agent.client.tool_select_options import detect_select_options
from agent.client.user_context import UserContext
from agent.client.veai import is_veai_agent, get_veai_context
from agent.client.veai.tool_call_fixer import veai_fix_tool_definition_optional_property_as_null_type
from agent.common.roles import ROLE_TOOL, ROLE_ASSISTANT
from agent.inference.token_handler import markdown_bold, markdown_back_tick
from agent.openai import GenerateOpts, completions_api
from agent.openai.chat_api import new_response, new_message, new_tool_call, new_stop_response
from agent.openai.chat_completions_api import CompletionResponse, ToolCall, ToolDefinition, FunctionDefinition, \
    ChatCompletionMessageParam, ChatCompletionChoice, ChatCompletionRequest
from agent.openai.completions_api import CompletionChoice
from agent.openai.models_api import ModelsListResponse, ModelObject
from agent.parser import Parser
from agent.preprocess.tool_call import PreprocessToolCall

STOP: Literal["stop"] = "stop"
LENGTH: Literal["length"] = "length"

log = logging.getLogger(__name__)

WARN_GENERATION_IS_INTERRUPTED_ = "Generation is interrupted."

USER_SELECT_CONTINUE = "continue"
USER_SELECT_INTERRUPT = "interrupt"
MIDDLEWARE_CHEKPOINT = "middleware_checkpoint"


class ControllerConfig(BaseModel):
    model_name: str
    response_timeout: timedelta = timedelta(minutes=20)


def new_http_response(stream: bool,
                      chunk_generator: Iterable[CompletionResponse]) -> StreamingResponse | CompletionResponse:
    if stream:
        return StreamingResponse(stream_generator(chunk_generator), media_type="text/event-stream")
    else:
        finish_reason, full_content, full_reasoning_content, full_tool_calls = make_union(chunk_generator)
        return new_response(
            message=new_message(ROLE_ASSISTANT, full_content, full_reasoning_content, full_tool_calls),
            finish_reason=finish_reason, stream=False)


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
            created=current_time,
        )])

    def new_generation_config(self,
                              temperature: float | None,
                              max_tokens: int | None,
                              max_completion_tokens: int | None,
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
        max_length = max_tokens or self.generate_config.max_tokens
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

            if logprobs:
                generation_config.logprobs = 1

        repetition_penalty = self.generate_config.repetition_penalty
        if repetition_penalty:
            generation_config.repetition_penalty = repetition_penalty

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
        user_context = get_veai_context(messages) if is_veai else None

        last_message = messages[-1] if messages else None

        if last_message and is_middleware_checkpoint(last_message) and USER_SELECT_INTERRUPT in str(
                last_message.content).lower():
            return new_http_response(stream, [new_response(
                message=new_message(role=ROLE_ASSISTANT, content="Interrupted"),
                stream=stream, finish_reason="stop")])

        invalid_response = self.validate_messages(messages, tools)
        if invalid_response:
            return new_http_response(stream, [invalid_response])

        tools_raw, function_by_name = group_function_by_name(tools, is_veai, self.is_fix_tool_type)

        tokenizer = self.tokenizer
        extra_context = {}
        # if self.generate_config.reasoning_supported:
        #     extra_context["enable_thinking"] = is_reasoning_enabled

        chat_history = new_chat_history(messages, tools_raw)
        full_prompt = tokenizer.apply_chat_template(history=chat_history,
                                                    add_generation_prompt=True,
                                                    extra_context=extra_context,
                                                    chat_template=self.chat_template)

        self.log_inference_prompt.debug(full_prompt)

        def is_stop():
            return self.stop_signal.is_set() or self.closed.is_set() or is_disconnected(request)

        chunk_generator = self.chunk_generator(
            prompt=full_prompt, generation_config=(
                self.new_generation_config(temperature=body.temperature,
                                           max_tokens=body.max_tokens,
                                           max_completion_tokens=body.max_completion_tokens,
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
                        ) -> Iterable[CompletionResponse]:
        pass

    def validate_messages(self, messages, tools) -> CompletionResponse | None:
        request_user_select = detect_select_options(tools)
        preprocess_tool_call = PreprocessToolCall()
        looped_function, count = preprocess_tool_call.check_loop_tool_calls(messages)
        if looped_function:
            # log
            msg = looped_function.render_markdown()
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
                completion_message = new_message(tool_calls=[tool_call])
            else:
                completion_message = new_message(content=(msg + WARN_GENERATION_IS_INTERRUPTED_))
            return new_response(message=completion_message, finish_reason=STOP)
        return None

    async def completions(self, body: completions_api.CompletionRequest, request: Request):
        prompt = body.prompt
        if not prompt:
            prompt = ""

        self.log_inference_prompt.debug(prompt)

        generation_config = self.new_generation_config(temperature=body.temperature,
                                                       max_tokens=body.max_tokens,
                                                       max_completion_tokens=self.generate_config.max_new_tokens)
        response_id = str(uuid.uuid4())

        encode_size = self.get_tokens_size(prompt)
        max_length = generation_config.max_length
        over_limit_response = self.check_prompt_limit(max_length, encode_size, response_id)
        if over_limit_response:
            return over_limit_response

        def is_stop():
            return self.stop_signal.is_set() or self.closed.is_set() or is_disconnected(request)

        stream = body.stream
        chunk_generator = self.chunk_generator(prompt=prompt, generation_config=generation_config,
                                               tokenizer=self.tokenizer, init_chat_events=True,
                                               is_stop=is_stop, is_veai=False)

        def chunk_converter(chunk_generator: Iterable[CompletionResponse]) -> Iterable[
            completions_api.CompletionResponse]:
            def convert_response(r: CompletionResponse) -> completions_api.CompletionResponse:
                return completions_api.CompletionResponse(model=r.model, id=r.id, choices=[
                    convert_choice(c) for c in r.choices])

            def convert_choice(c: ChatCompletionChoice) -> CompletionChoice:
                delta = c.delta
                content = delta.content if delta and delta.content else ""
                reason: Literal["stop"] | None = "stop" if c.finish_reason else None
                return completions_api.CompletionChoice(text=content, finish_reason=reason)

            for c in chunk_generator:
                yield convert_response(c)

        chunk_converter = chunk_converter(chunk_generator)
        if stream:
            return StreamingResponse(stream_generator(chunk_converter), media_type="text/event-stream")
        else:
            finish_reason, full_content, full_reasoning_content, full_tool_calls = make_union(chunk_converter)
            return new_response(response_id=response_id,
                                message=new_message(ROLE_ASSISTANT, full_content, full_reasoning_content,
                                                    full_tool_calls),
                                finish_reason=finish_reason, stream=False)

    def check_prompt_limit(self, max_length: int, encode_size: int,
                           response_id: str) -> CompletionResponse | None:
        model_name = self.config.model_name
        if encode_size >= max_length:
            return new_stop_response(response_id, model_name, finish_reason=LENGTH,
                                     content=f"prompt exceeds limit: {encode_size} >= {max_length}")
        return None

    def get_tokens_size(self, prompt: str) -> int:
        encode_size = self.tokenizer.encode(prompt).input_ids.size
        return encode_size


def make_union(chunk_generator: Iterable[CompletionResponse]) -> tuple[
    Literal["stop", "length", "tool_calls", "content_filter"], str, str, list[ToolCall]]:
    full_content = ""
    full_reasoning_content = ""
    full_tool_calls: list[ToolCall] = []
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


def new_chat_history(messages: list[BaseModel], tools_raw: list[dict[str, Any]] | None = None) -> ChatHistory:
    chat_history = ChatHistory()
    chat_history.set_messages(list(map(BaseModel.model_dump, messages)) if messages else [])
    if tools_raw:
        chat_history.set_tools(tools_raw)
    return chat_history


def group_function_by_name(tools: list[ToolDefinition] | None, is_veai: bool, is_fix_tool_type: bool = False) -> tuple[
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


def stream_generator(chunk_generator: Iterable[CompletionResponse]) -> Iterable[str]:
    for chunk in chunk_generator:
        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"


def is_middleware_checkpoint(last_message: ChatCompletionMessageParam) -> str | None | bool:
    is_tool = last_message.role == ROLE_TOOL
    if not is_tool:
        return False
    tool_call_id = last_message.tool_call_id
    return tool_call_id and tool_call_id.startswith(MIDDLEWARE_CHEKPOINT)
