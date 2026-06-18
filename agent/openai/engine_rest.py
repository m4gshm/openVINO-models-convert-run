import asyncio
import itertools
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from os import pipe
from typing import Any, Generator

import openvino_genai
from fastapi.routing import APIRouter
from openvino_genai import ChatHistory
from openvino_genai.py_openvino_genai import ContinuousBatchingPipeline, GenerationHandle, GenerationFinishReason
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import StreamingResponse

from common.metric_mem import get_current_memory
from common.openai_api import new_response, new_tool_call, new_message
from common.openai_model import OpenAIChatCompletionRequest, ChatCompletionMessageParam, ChatCompletionMessage, \
    FunctionDefinition, OpenAICompletionResponse, ToolCall
from common.roles import ROLE_TOOL, ROLE_ASSISTANT
from inference.token_handler import TokenHandler, TokenHandlerConfig, StopSignal, get_stop_signal_by_finish_reason
from openai import GenerateConfig
from parser.base import Parser
from preprocess.tool_call import PreprocessToolCall
from tool_select_options import detect_select_options
from veai.tool_call_fixer import fix_tool_definition_optional_property_as_null_type

log = logging.getLogger(__name__)

log_inference = logging.getLogger("inference")
log_inference_prompt = logging.getLogger("inference.prompt")

WARN_GENERATION_IS_INTERRUPTED_ = "Generation is interrupted."

USER_SELECT_CONTINUE = "continue"
USER_SELECT_INTERRUPT = "interrupt"
MIDDLEWARE_CHEKPOINT = "middleware_checkpoint"

request_counter = itertools.count(start=0)


class ControllerConfig(BaseModel):
    model_name: str
    response_timeout: timedelta = timedelta(minutes=5)


class Controller:
    def __init__(self, config: ControllerConfig, parser: Parser, pipe: ContinuousBatchingPipeline,
                 handler_config: TokenHandlerConfig, generate_config: GenerateConfig, router: APIRouter = APIRouter()):
        router.post("/v1/chat/completions")(self.chat)
        self.router = router
        self.parser = parser
        self.pipe = pipe
        self.handler_config = handler_config
        self.generate_config = generate_config
        self.config = config
        self.executor = ThreadPoolExecutor()
        self.active_handles_lock = threading.Lock()
        self.active_handles: dict[int, GenerationHandle] = {}

        self.stop_event = threading.Event()
        # self.engine_thread = threading.Thread(target=self._engine_loop, daemon=True)
        # self.engine_thread.start()

    # def _engine_loop(self):
    #     pass
        # while not self.stop_event.is_set():
            # if self.pipe.has_non_finished_requests():
            #     self.step()
            # else:
            #     time.sleep(0.5)

            # metrics = self.pipe.get_metrics()
            # log.info(f"metrics: requests={metrics.requests}, scheduled_requests={metrics.scheduled_requests}, "
            #          f"cache_size_in_bytes={metrics.cache_size_in_bytes}, "
            #          f"kv_cache_size_in_bytes={metrics.kv_cache_size_in_bytes}")

    def step(self):
        try:
            self.pipe.step()
        except Exception as e:
            log.error(f"pipe step error {e}")

    def shutdown(self):
        with self.active_handles_lock:
            active_ids = list(self.active_handles.keys())
            for req_id in active_ids:
                handle = self.active_handles.get(req_id)
                if handle:
                    try:
                        # log
                        handle.cancel()
                    except Exception as e:
                        # log
                        pass

        self.stop_event.set()
        self.engine_thread.join(timeout=2.0)

    async def chat(self, body: OpenAIChatCompletionRequest, request: Request):
        def is_middleware_checkpoint(last_message: ChatCompletionMessageParam) -> str | None | bool:
            is_tool = last_message.role == ROLE_TOOL
            if not is_tool:
                return False
            tool_call_id = last_message.tool_call_id
            return tool_call_id and tool_call_id.startswith(MIDDLEWARE_CHEKPOINT)

        loop = asyncio.get_event_loop()

        def is_disconnected() -> bool:
            disconnected = False
            try:
                disconnected = asyncio.run_coroutine_threadsafe(request.is_disconnected(), loop).result(0.5)
                if disconnected:
                    log.debug(f"disconnected http request")
            except asyncio.TimeoutError:
                pass
                # log.debug(f"disconnected http request check timeout")
            return disconnected

        is_reasoning_enabled: bool = self.generate_config.reasoning_supported and (
                body.model_config.get("reasoning") or True)

        messages = body.messages

        last_message = messages[-1] if messages else None
        stream = body.stream == True
        if last_message:
            if is_middleware_checkpoint(last_message):
                if USER_SELECT_INTERRUPT in str(last_message.content).lower():
                    return new_response(
                        chat_completion_message=ChatCompletionMessage(role=ROLE_ASSISTANT, content="Interrupted"),
                        stream=stream, finish_reason="stop")

        log.info(f"inbound history messages {len(messages)}")

        tools = body.tools
        request_user_select = detect_select_options(tools)

        preprocess_tool_call = PreprocessToolCall()
        looped_function = preprocess_tool_call.check_loop_calls(messages)
        if looped_function:
            # log
            msg = f"Multiple calls of the '{looped_function.name}' tool " \
                  f"result in the same response '{looped_function.result}'. "
            if request_user_select:
                # log
                question = request_user_select.new_call(msg + "What to do next?",
                                                        [USER_SELECT_CONTINUE, USER_SELECT_INTERRUPT])
                tool_call = new_tool_call(call_id=MIDDLEWARE_CHEKPOINT + "_" + str(uuid.uuid4()), function=question)
                completion_message = new_message(tool_calls=[tool_call])
            else:
                completion_message = new_message(content=(msg + WARN_GENERATION_IS_INTERRUPTED_))
            return new_response(chat_completion_message=completion_message, finish_reason="stop", stream=stream)

        function_by_name: dict[str, FunctionDefinition] = {}
        tools_raw: list[dict[str, Any]] = []
        for tool in (tools or []):
            fixed_tool = fix_tool_definition_optional_property_as_null_type(tool)
            tools_raw.append(fixed_tool.model_dump())
            function = tool.function
            function_by_name[function.name] = function

        chat_history = ChatHistory()
        chat_history_messages = list(map(ChatCompletionMessageParam.model_dump, messages)) if messages else []
        chat_history.set_messages(chat_history_messages)
        chat_history.set_tools(tools_raw)

        tokenizer = self.pipe.get_tokenizer()
        extra_context = {}
        if self.generate_config.reasoning_supported:
            extra_context["enable_thinking"] = is_reasoning_enabled

        request_id = next(request_counter)

        full_prompt = tokenizer.apply_chat_template(history=chat_history,
                                                    add_generation_prompt=True,
                                                    tools=tools_raw,
                                                    extra_context=extra_context)
        log_inference_prompt.debug(full_prompt)

        generation_config = openvino_genai.py_openvino_genai.GenerationConfig()
        generation_config.max_new_tokens = body.max_completion_tokens or self.generate_config.default_max_new_tokens
        generation_config.max_length = body.max_tokens or self.generate_config.default_max_tokens
        generation_config.apply_chat_template = False if full_prompt else True

        temp = body.temperature or self.generate_config.default_temperature
        if temp < 0.05:
            # Greedy Search
            generation_config.do_sample = False
        else:
            generation_config.do_sample = True
            generation_config.temperature = temp
            generation_config.top_p = body.top_p or self.generate_config.default_top_p
            generation_config.top_k = self.generate_config.default_top_k
            generation_config.min_p = self.generate_config.default_min_p

            if body.frequency_penalty:
                generation_config.frequency_penalty = body.frequency_penalty

            if body.logprobs:
                generation_config.logprobs = 1

        generation_config.repetition_penalty = self.generate_config.default_repetition_penalty

        def stream_generator() -> Generator[str, None, None]:
            for chunk in chunk_generator():
                yield f"data: {chunk.model_dump_json()}\n\n"

        def chunk_generator() -> Generator[OpenAICompletionResponse, None, None]:
            before_generate_mem = get_current_memory()
            try:
                if log_inference.isEnabledFor(logging.DEBUG):
                    log_inference.debug(
                        f"inference starting with parameters: do_sample={generation_config.do_sample},"
                        f" max_length={generation_config.max_length}, "
                        f"max_new_tokens={generation_config.max_new_tokens}, "
                        f"do_sample={generation_config.do_sample}, temperature={generation_config.temperature:.2f}, "
                        f"top_p={generation_config.top_p:.2f}, top_k={generation_config.top_k}, "
                        f"min_p={generation_config.min_p:.2f}, repetition_penalty={generation_config.repetition_penalty:.2f}, "
                        f"presence_penalty={generation_config.presence_penalty:.2f}, "
                        f"frequency_penalty={generation_config.frequency_penalty:.2f}")
                else:
                    log_inference.info(f"inference starting")

                start_thinking = self.parser.is_prompt_start_thinking(full_prompt)
                streamer = (TokenHandler(tokenizer=tokenizer, parser=self.parser, is_stop=is_disconnected,
                                         supported_functions=function_by_name,
                                         start_thinking=start_thinking,
                                         config=self.handler_config))
                generate_result = self.pipe.add_request(request_id=request_id, prompt=full_prompt,
                                                        generation_config=generation_config, images=[], videos=[])
                with self.active_handles_lock:
                    self.active_handles[request_id] = generate_result


                def is_response_timeout(start):
                    now_time = time.perf_counter()
                    duration = timedelta(seconds=(now_time - start))
                    if duration >= self.config.response_timeout:
                        log.warning("inference timeout")
                        generate_result.stop(GenerationFinishReason.NONE)
                        return True
                    return False


                request_start = time.perf_counter()
                response_timeout = False
                while not self.pipe.has_non_finished_requests():
                    self.step()
                    time.sleep(0.2)
                    response_timeout = is_response_timeout(request_start)
                    if response_timeout:
                        break

                if response_timeout:
                    yield new_response(stream=stream, finish_reason=StopSignal.CANCEL.value)
                else:
                    unique_id = str(uuid.uuid4())
                    def read():
                        started = False
                        request_start = time.perf_counter()
                        while True:
                            has_requests = self.pipe.has_non_finished_requests()
                            if not has_requests:
                                yield new_response(stream=stream, finish_reason=StopSignal.STOP.value)
                                return
                            self.step()
                            can_read = generate_result.can_read()
                            if can_read:
                                started = True
                                read_timeout = is_response_timeout(request_start)
                                if read_timeout:
                                    log.warning("read timeout")
                                    yield new_response(stream=stream, finish_reason=StopSignal.CANCEL.value)
                                    return
                                else:
                                    generation_handle = generate_result.read()
                                    items = generation_handle.items()
                                    if len(items)  == 0:
                                        log.error("empty generation")
                                        yield new_response(stream=stream, finish_reason=StopSignal.CANCEL.value)
                                        return
                                    for k, generation_output in items:
                                        responses, stop_signal = streamer.handle_token(generation_output.generated_ids)
                                        for response in responses:
                                            response.id = unique_id
                                            response.model = self.config.model_name
                                            yield response

                                        finish_reason = generation_output.finish_reason
                                        if finish_reason and finish_reason != GenerationFinishReason.NONE:
                                            stop_signal = get_stop_signal_by_finish_reason(finish_reason)
                                            if not stop_signal:
                                                stop_signal = StopSignal.STOP
                                            yield new_response(stream=stream, finish_reason=stop_signal.value)
                                            return
                                        elif stop_signal:
                                            generate_result.stop(stop_signal.finish_reason)
                                            yield new_response(stream=stream, finish_reason=stop_signal.value)
                                            return
                            elif started:
                                yield new_response(stream=stream, finish_reason=StopSignal.STOP.value)
                                return

                    yield from read()

                metrics = self.pipe.get_metrics()
                log.info(f"inference finished: "
                         f"kv_cache_size={metrics.kv_cache_size_in_bytes / 1024:.2f}MB, "
                         f"cache_size={metrics.cache_size_in_bytes / 1024:.2f}MB "
                         f"cache_usage={metrics.cache_usage}, "
                         f"max_cache_usage={metrics.max_cache_usage}, "
                         f"requests={metrics.requests}, "
                         f"scheduled_requests={metrics.scheduled_requests}"
                         )
            except Exception as e:
                log_inference.error(f"inference error: {e}", exc_info=e)
                yield new_response(stream=stream, finish_reason=StopSignal.CANCEL.value)
                raise
            finally:
                with self.active_handles_lock:
                    del self.active_handles[request_id]
                after_generate_mem = get_current_memory()
                delta = after_generate_mem - before_generate_mem
                log.debug(f"consumed memory: {after_generate_mem:.2f} MB, delta: {delta:.2f} MB")

        if stream:
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            full_content = ""
            full_reasoning_content = ""
            full_tool_calls: list[ToolCall] = []
            finish_reason = "stop"

            for chunk_data in chunk_generator():
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

        return new_response(chat_completion_message=
                            new_message(full_content, full_reasoning_content, full_tool_calls),
                            finish_reason=finish_reason, stream=False)
