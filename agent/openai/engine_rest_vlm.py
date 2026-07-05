import logging
import queue
import threading
import uuid
from collections import abc
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
from typing import Generator
from typing import SupportsInt, Literal

from fastapi.routing import APIRouter
from openvino_genai import ChatHistory
from openvino_genai import VLMPipeline, GenerationFinishReason, py_openvino_genai, StreamingStatus
from openvino_genai.py_openvino_genai import DecodedResults, LLMPipeline, MeanStdPair, \
    Tokenizer, VLMDecodedResults, GenerationConfig

from agent.common.metric_mem import get_current_memory
from agent.inference.token_handler import TokenHandler, TokenHandlerConfig, StopSignal
from agent.openai import GenerateConfig
from agent.openai.chat_api import new_stop_response
from agent.openai.chat_completions_api import CompletionResponse, FunctionDefinition
from agent.openai.engine_rest_common import ControllerConfig, BaseController
from agent.parser import Parser

log = logging.getLogger(__name__)


class VlmController(BaseController):
    def __init__(self, config: ControllerConfig, parser: Parser, pipe: VLMPipeline | LLMPipeline,
                 handler_config: TokenHandlerConfig,
                 generate_config: GenerateConfig, router: APIRouter, chat_template: str = ''):
        super().__init__(config, parser, pipe.get_tokenizer(), generate_config, router, chat_template)
        self.pipe = pipe
        self.handler_config = handler_config
        self.generate_config = generate_config
        self.config = config
        self.executor = ThreadPoolExecutor()
        self.request_lock = threading.Lock()

    def chunk_generator(self, prompt: str, chat_history: ChatHistory, generation_config: GenerationConfig,
                        tokenizer: Tokenizer,
                        init_chat_events: bool, is_stop: Callable[[], bool], is_veai: bool,
                        function_by_name: dict[str, FunctionDefinition] | None = None, user_context=None,
                        ) -> Generator[
        CompletionResponse, None, None]:

        response_id = str(uuid.uuid4())
        encode_size = self.get_tokens_size(prompt)
        max_length = generation_config.max_length
        over_limit_response = self.check_prompt_limit(max_length, encode_size, response_id)
        if over_limit_response:
            yield over_limit_response
            return

        with self.request_lock:
            chunk_queue: queue.Queue[CompletionResponse | None] = queue.Queue()
            stop_stream_handling: queue.Queue[bool] = queue.Queue()
            start_stream_handling: queue.Queue[bool] = queue.Queue()
            before_generate_mem = get_current_memory()

            encode_size = self.get_tokens_size(prompt)
            max_length = generation_config.max_length

            def run_inference():
                try:
                    if self.log_inference.isEnabledFor(logging.DEBUG):
                        self.log_inference.debug(
                            f"inference start: "
                            f"pipe_type={type(self.pipe)}, "
                            f"prompt_tokens={encode_size}, "
                            f"do_sample={generation_config.do_sample}, "
                            f"max_length={max_length}, "
                            f"max_new_tokens={generation_config.max_new_tokens}, "
                            f"temperature={generation_config.temperature:.2f}, "
                            f"top_p={generation_config.top_p:.2f}, top_k={generation_config.top_k}, "
                            f"min_p={generation_config.min_p:.2f}, repetition_penalty={generation_config.repetition_penalty:.2f}, "
                            f"presence_penalty={generation_config.presence_penalty:.2f}, "
                            f"frequency_penalty={generation_config.frequency_penalty:.2f}"
                        )
                    else:
                        self.log_inference.info(f"inference start")

                    token_handler = TokenHandler(tokenizer=tokenizer, parser=self.parser,
                                                 init_chat_events=init_chat_events,
                                                 is_stop=is_stop, is_veai=is_veai, config=self.handler_config,
                                                 supported_functions=function_by_name, user_context=user_context)
                    streamer = StreamerWrapper(token_handler,
                                               start_stream_handling=start_stream_handling,
                                               stop_stream_handling=stop_stream_handling,
                                               chunk_queue=chunk_queue)

                    generate_result = start_generate_result(streamer)
                    chunk_queue.put_nowait(None)

                    metrics = generate_result.perf_metrics if isinstance(generate_result, DecodedResults) else None

                    def to_str(d: MeanStdPair) -> str:
                        return f"std {d.std} , mean {d.mean}"

                    log_msg = f"inference finished: "
                    if isinstance(generate_result, DecodedResults):
                        inference_finish_reasons = generate_result.finish_reasons
                        log_msg += f"reason '{inference_finish_reasons}'"
                    else:
                        inference_finish_reasons = None

                    if metrics:
                        log_msg += (f"num_input_tokens={metrics.get_num_input_tokens()}, "
                                    f"generated_tokens={metrics.get_num_generated_tokens()}, "
                                    f"generate_duration={to_str(metrics.get_generate_duration())}, "
                                    f"inference_duration={to_str(metrics.get_inference_duration())}, "
                                    f"ttft={to_str(metrics.get_ttft())}, "
                                    f"throughput={to_str(metrics.get_throughput())}")
                    if self.log_inference.isEnabledFor(logging.DEBUG):
                        texts = generate_result.texts if isinstance(generate_result,
                                                                    DecodedResults) else generate_result
                        self.log_inference.debug(f"{log_msg}\nresult: {texts}")
                    else:
                        self.log_inference.info(log_msg)

                    after_generate_mem = get_current_memory()
                    generate_cost = after_generate_mem - before_generate_mem
                    log.debug(f"consumed memory: {after_generate_mem:.2f} MB, generate delta: {generate_cost:.2f} MB")

                    self.log_inference_generated.debug("".join(token_handler.phrase.full))
                    self.log_inference_generated.debug("".join(token_handler.tool_call_phrase.full))

                    inference_finish_reason = inference_finish_reasons[0] if inference_finish_reasons else None
                    if inference_finish_reason is None or inference_finish_reason == GenerationFinishReason.NONE:
                        self.log_inference.warning(f"inference finished by unexpected status {inference_finish_reason}")

                except Exception as e:
                    start_stream_handling.put_nowait(True)
                    self.log_inference.error(f"inference error: {e}", exc_info=e)
                    err_str = str(e)
                    finish_reason: Literal[
                        "length", "stop"] = "length" if "<= m_max_prompt_len" in err_str else "stop"
                    chunk_queue.put_nowait(new_stop_response(content=err_str, finish_reason=finish_reason))
                    chunk_queue.put_nowait(None)

            def start_generate_result(streamer: StreamerWrapper) -> VLMDecodedResults:
                pipe = self.pipe
                if isinstance(pipe, VLMPipeline):
                    vlm_pipe: VLMPipeline = pipe
                    generate_result = vlm_pipe.generate(history=chat_history, generation_config=generation_config,
                                                        streamer=streamer)
                elif isinstance(pipe, LLMPipeline):
                    llm_pipe: LLMPipeline = pipe
                    generate_result = llm_pipe.generate(inputs=chat_history, generation_config=generation_config,
                                                        streamer=streamer)
                else:
                    raise NotImplementedError(f"unexpected pipe type {type(pipe)}")
                return generate_result

            try:
                inference_task = self.executor.submit(run_inference)
                start_stream_handling.get()
                stop_inference = False
                while not stop_inference:
                    if is_stop():
                        break
                    try:
                        chunk = chunk_queue.get(timeout=20)
                        if chunk:
                            chunk.id = response_id
                            chunk.model = self.config.model_name
                            yield chunk
                        else:
                            stop_inference = True
                    except queue.Empty:
                        pass
                    except TimeoutError:
                        pass

            except Exception as e:
                log.error(f"chunk processing error: {e}", exc_info=e)

            stop_stream_handling.put_nowait(True)
            if not inference_task.done():
                log.info("waiting for inference to complete")
                try:
                    r = inference_task.result(timeout=20)
                except Exception as e:
                    log.error(f"waiting inference completion error: {e}", exc_info=e)
            log.info("inference handling is done")


class StreamerWrapper(py_openvino_genai.StreamerBase):
    def __init__(self, streamer: TokenHandler, chunk_queue: queue.Queue[CompletionResponse | None],
                 stop_stream_handling: queue.Queue[bool], start_stream_handling: queue.Queue[bool]):
        super().__init__()
        self.streamer = streamer
        self.started = False
        self.stop_stream_handling = stop_stream_handling
        self.start_stream_handling = start_stream_handling
        self.chunk_queue = chunk_queue

    def end(self) -> None:
        pass

    def write(self, tokens: abc.Sequence[SupportsInt]) -> StreamingStatus:
        if not self.started:
            self.start_stream_handling.put(True)
            self.started = True
        if not self.stop_stream_handling.empty() and self.stop_stream_handling.get_nowait():
            log.debug("stream finished by stop signal")
            return StreamingStatus.STOP

        responses, stop_signal = self.streamer.handle_token(tokens)

        if responses:
            for response in responses:
                self.chunk_queue.put_nowait(response)

        if stop_signal == StopSignal.STOP:
            return StreamingStatus.STOP
        elif stop_signal == StopSignal.TOOL_CALL:
            return StreamingStatus.TOOL_CALL_STOP
        elif stop_signal == StreamingStatus.CANCEL:
            return StreamingStatus.CANCEL
        return StreamingStatus.RUNNING
