import asyncio
import collections
import logging
import queue
import typing
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Generator

import openvino_genai
import openvino_genai as ov_genai
from fastapi.routing import APIRouter
from openvino_genai import VLMPipeline, ChatHistory, GenerationFinishReason
from openvino_genai.py_openvino_genai import MeanStdPair, StreamingStatus
from starlette.requests import Request
from starlette.responses import StreamingResponse

from common.metric_mem import get_current_memory
from common.openai_api import new_response, new_tool_call, new_message
from common.openai_model import OpenAIChatCompletionRequest, ChatCompletionMessageParam, ResponseFormat, \
    ChatCompletionMessage, FunctionDefinition, OpenAICompletionResponse, ToolCall
from common.roles import ROLE_TOOL, ROLE_ASSISTANT
from inference.token_handler import TokenHandler, TokenHandlerConfig, StopSignal
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


class VlmController:
    def __init__(self, model_name: str, parser: Parser, pipe: VLMPipeline, handler_config: TokenHandlerConfig,
                 generate_config: GenerateConfig, router: APIRouter = APIRouter()):
        router.post("/v1/chat/completions")(self.chat)
        self.router = router
        self.parser = parser
        self.pipe = pipe
        self.handler_config = handler_config
        self.generate_config = generate_config
        self.model_name = model_name
        self.executor = ThreadPoolExecutor()

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

        body.response_format = ResponseFormat(type="json_object")
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

        full_prompt = tokenizer.apply_chat_template(history=chat_history,
                                                    add_generation_prompt=True,
                                                    tools=tools_raw,
                                                    extra_context=extra_context)

        log_inference_prompt.debug(full_prompt)

        generation_config = ov_genai.GenerationConfig()
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
            chunk_queue: queue.Queue[OpenAICompletionResponse | None] = queue.Queue()
            stop_stream_handling: queue.Queue[bool] = queue.Queue()
            start_stream_handling: queue.Queue[bool] = queue.Queue()

            def run_inference():
                try:
                    self.pipe.start_chat()
                    if log_inference.isEnabledFor(logging.DEBUG):
                        log_inference.debug(
                            f"inference starting with parameters max_length={generation_config.max_length}, "
                            f"max_new_tokens={generation_config.max_new_tokens}, "
                            f"do_sample={generation_config.do_sample}, temperature={generation_config.temperature:.2f}, "
                            f"top_p={generation_config.top_p:.2f}, top_k={generation_config.top_k}, "
                            f"min_p={generation_config.min_p:.2f}, repetition_penalty={generation_config.repetition_penalty:.2f}, "
                            f"presence_penalty={generation_config.presence_penalty:.2f}, "
                            f"frequency_penalty={generation_config.frequency_penalty:.2f}")
                    else:
                        log_inference.info(f"inference starting")

                    before_generate_mem = get_current_memory()

                    start_thinking = self.parser.is_prompt_start_thinking(full_prompt)

                    class StreamerWrapper(openvino_genai.py_openvino_genai.StreamerBase):
                        def __init__(self, streamer: TokenHandler):
                            super().__init__()
                            self.streamer = streamer
                            self.started = False

                        def end(self) -> None:
                            pass

                        def write(self, tokens: collections.abc.Sequence[typing.SupportsInt]) -> StreamingStatus:
                            if not self.started:
                                start_stream_handling.put(True)
                                self.started = True
                            if not stop_stream_handling.empty() and stop_stream_handling.get_nowait():
                                log.debug("stream finished by stop signal")
                                return StreamingStatus.STOP

                            responses, stop_signal = self.streamer.handle_token(tokens)
                            if responses:
                                for response in responses:
                                    chunk_queue.put_nowait(response)

                            if stop_signal == StopSignal.STOP:
                                return StreamingStatus.STOP
                            elif stop_signal == StreamingStatus.TOOL_CALL_STOP:
                                return StreamingStatus.TOOL_CALL_STOP
                            elif stop_signal == StreamingStatus.CANCEL:
                                return StreamingStatus.CANCEL
                            return StreamingStatus.RUNNING

                    streamer = (
                        StreamerWrapper(TokenHandler(tokenizer=tokenizer, parser=self.parser, is_stop=is_disconnected,
                                                     supported_functions=function_by_name, start_thinking=start_thinking,
                                                     config=self.handler_config)))
                    generate_result = self.pipe.generate(prompt=full_prompt, generation_config=generation_config,
                                                         streamer=streamer)
                    chunk_queue.put_nowait(None)
                    after_generate_mem = get_current_memory()
                    generate_cost = after_generate_mem - before_generate_mem
                    log.debug(f"consumed memory: {after_generate_mem:.2f} MB, generate delta: {generate_cost:.2f} MB")

                    metrics = generate_result.perf_metrics

                    def to_str(d: MeanStdPair) -> str:
                        return f"std {d.std} , mean {d.mean}"

                    inference_finish_reasons = generate_result.finish_reasons
                    log_msg = (
                        f"inference finished with reason '{inference_finish_reasons}'\n"
                        f"num_input_tokens:{metrics.get_num_input_tokens()}\n"
                        f"generated_tokens:{metrics.get_num_generated_tokens()}\n"
                        f"generate_duration: {to_str(metrics.get_generate_duration())}\n"
                        f"inference_duration: {to_str(metrics.get_inference_duration())}\n"
                        f"ttft: {to_str(metrics.get_ttft())}\n"
                        f"throughput: {to_str(metrics.get_throughput())}\n"
                    )
                    if log_inference.isEnabledFor(logging.DEBUG):
                        log_inference.debug(f"{log_msg}\nresult: {generate_result.texts}")
                    else:
                        log_inference.info(log_msg)

                    inference_finish_reason = inference_finish_reasons[0] if inference_finish_reasons else None
                    if inference_finish_reason is None or inference_finish_reason == GenerationFinishReason.NONE:
                        log.warning(f"inference finished by unexpected status {inference_finish_reason}")

                except Exception as e:
                    log_inference.error(f"inference error: {e}", exc_info=e)
                    raise
                finally:
                    self.pipe.finish_chat()

            unique_id = str(uuid.uuid4())
            inference_task = self.executor.submit(run_inference)
            try:
                start_stream_handling.get()
                stop_inference = False
                while not stop_inference:
                    if is_disconnected():
                        break
                    try:
                        chunk = chunk_queue.get(timeout=20)
                        if chunk:
                            chunk.id = unique_id
                            chunk.model = self.model_name
                            yield chunk
                        else:
                            stop_inference = True
                    except queue.Empty:
                        pass
                    except TimeoutError:
                        pass

            except Exception as e:
                log.error(f"chunk processing error: {e}", exc_info=e)
            finally:
                stop_stream_handling.put_nowait(True)
                if not inference_task.done():
                    log.info("waiting for inference to complete")
                    try:
                        r = inference_task.result(timeout=20)
                    except Exception as e:
                        log.error(f"waiting inference completion error: {e}", exc_info=e)
                log.info("inference handling is done")

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

    def shutdown(self):
        pass
