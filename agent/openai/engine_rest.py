import itertools
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Generator, Callable
from typing import Literal

from fastapi.routing import APIRouter
from openvino_genai.py_openvino_genai import ContinuousBatchingPipeline, GenerationHandle, GenerationFinishReason, \
    GenerationConfig, Tokenizer, GenerationStatus, ChatHistory

from agent.common.metric_mem import get_current_memory
from agent.inference.token_handler import TokenHandler, TokenHandlerConfig, get_stop_signal_by_finish_reason, \
    markdown_bold, get_finish_str, StopSignal
from agent.openai import GenerateOpts
from agent.openai.chat_api import new_stop_response
from agent.openai.chat_completions_api import CompletionResponse, FunctionDefinition
from agent.openai.engine_rest_common import ControllerConfig, BaseController
from agent.parser import Parser, StateEvent

log = logging.getLogger(__name__)

request_counter = itertools.count(start=0)


def add_stop_signal(responses: list[CompletionResponse], stop_signal: StopSignal):
    choices = responses[-1].choices if responses else None
    finish_reason = get_finish_str(stop_signal)
    if choices:
        choices[-1].finish_reason = finish_reason
    else:
        responses.append(new_stop_response(finish_reason=finish_reason))
    return responses


class ContinuousBatchingController(BaseController):
    def __init__(self, config: ControllerConfig, parser: Parser, pipe: ContinuousBatchingPipeline,
                 handler_config: TokenHandlerConfig, stop_signal: threading.Event,
                 generate_config: GenerateOpts, is_fix_tool_type: bool, chat_template: str = ''):
        super().__init__(config, parser, pipe.get_tokenizer(), generate_config, is_fix_tool_type, chat_template)
        self.pipe = pipe
        self.handler_config = handler_config
        self.generate_config = generate_config
        self.config = config
        self.executor = ThreadPoolExecutor()
        self.active_handles_lock = threading.Lock()
        self.active_handles: dict[int, GenerationHandle] = {}
        self.stop_signal = stop_signal

    def step(self):
        try:
            self.pipe.step()
        except Exception as e:
            log.error(f"pipe step error {e}")
            raise e

    def shutdown(self):
        super().shutdown()
        with self.active_handles_lock:
            pipe = self.pipe
            if pipe is None:
                log.info("pipe already closed")
                return
            active_ids = list(self.active_handles.keys())
            log.info(f"canceling of active requests {active_ids}")
            for req_id in active_ids:
                handle = self.active_handles.get(req_id)
                if handle:
                    try:
                        # log
                        handle.cancel()
                    except Exception as e:
                        log.error(f"error on pipe handle cancelling: {e}")
                        pass
            self.pipe = None
            del pipe

    def chunk_generator(self, prompt: str, chat_history: ChatHistory, generation_config: GenerationConfig,
                        tokenizer: Tokenizer,
                        init_chat_events: bool, is_stop: Callable[[], bool], is_veai: bool,
                        function_by_name: dict[str, FunctionDefinition] | None = None, user_context=None,
                        ) -> Generator[
        CompletionResponse, None, None]:
        before_generate_mem = get_current_memory()
        request_id = next(request_counter)
        response_id = str(uuid.uuid4())
        model_name = self.config.model_name

        stop_response = new_stop_response(response_id, model_name)

        encode_size = self.get_tokens_size(prompt)
        max_length = generation_config.max_length
        over_limit_response = self.check_prompt_limit(max_length, encode_size, response_id)
        if over_limit_response:
            yield over_limit_response
            return
        if self.log_inference.isEnabledFor(logging.DEBUG):
            self.log_inference.debug(
                f"inference start: request={request_id}, "
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
            self.log_inference.info(f"inference start: request={request_id}")

        token_handler = TokenHandler(tokenizer=tokenizer,
                                     parser=self.parser,
                                     init_chat_events=init_chat_events,
                                     is_stop=is_stop,
                                     is_veai=is_veai,
                                     config=self.handler_config,
                                     supported_functions=function_by_name,
                                     )

        generation_handle: GenerationHandle
        try:
            generation_handle = self.pipe.add_request(request_id=request_id, prompt=prompt,
                                                      generation_config=generation_config, images=[], videos=[])
        except Exception as e:
            log.error(f"create pipe error {e}")
            raise e

        with self.active_handles_lock:
            self.active_handles[request_id] = generation_handle

        def is_response_timeout(start):
            now_time = time.perf_counter()
            duration = timedelta(seconds=(now_time - start))
            if duration >= self.config.response_timeout:
                log.warning(f"inference timeout: {duration}")
                generation_handle.stop(GenerationFinishReason.NONE)
                return True
            return False

        try:
            request_start = time.perf_counter()
            response_timeout = False
            while not self.pipe.has_non_finished_requests():
                self.step()
                time.sleep(0.2)
                response_timeout = is_response_timeout(request_start)
                if response_timeout:
                    break

            if response_timeout:
                yield stop_response
            else:
                unique_id = str(uuid.uuid4())

                def read():
                    empty_tokens_limit = 100
                    empty_out_counter = 0
                    started = False
                    request_start = time.perf_counter()
                    while True:
                        has_requests = self.pipe.has_non_finished_requests()
                        if not has_requests:
                            if not started:
                                log.debug("has no active requests")
                                yield stop_response
                                return
                            else:
                                log.debug("has no active requests but current has not been finished")

                        self.step()
                        can_read = generation_handle.can_read()
                        if can_read:
                            started = True
                            read_timeout = is_response_timeout(request_start)
                            if read_timeout:
                                yield stop_response
                                return
                            else:
                                generation_outputs = generation_handle.read()
                                items = generation_outputs.items()
                                if len(items) == 0:
                                    self.log_inference_token_metrics.debug("empty generation")
                                    empty_out_counter += 1
                                    if empty_out_counter >= empty_tokens_limit:
                                        self.log_inference.error("empty generation limits exceed")
                                        yield new_stop_response(content=markdown_bold(
                                            f"empty generation limits exceed: {empty_tokens_limit}"),
                                            response_id=response_id, model=model_name)
                                        return
                                for k, generation_output in items:
                                    generated_ids = generation_output.generated_ids
                                    self.log_inference_token_metrics.debug(f"generation_output: ids={generated_ids}, "
                                                                           f"score={generation_output.score}, "
                                                                           f"log_probs={generation_output.generated_log_probs}")

                                    responses, stop_signal = token_handler.handle_tokens(generated_ids)

                                    finish_reason = generation_output.finish_reason
                                    if not stop_signal and finish_reason and finish_reason != GenerationFinishReason.NONE:
                                        stop_signal = get_stop_signal_by_finish_reason(finish_reason)
                                        log.debug(f"generation_output has finish_reason '{finish_reason}' converted to "
                                                  f"stop_signal '{stop_signal}'")

                                    if stop_signal:
                                        add_stop_signal(responses, stop_signal)

                                    for response in responses:
                                        response.id = unique_id
                                        response.model = model_name
                                        yield response
                                    if stop_signal:
                                        return
                        elif started:
                            self.log_inference.debug("no more reads")
                            if token_handler.state.has_event(StateEvent.CONVERSATION):
                                self.log_inference.debug("force end conversation")
                                responses, stop_signal = token_handler.conversation_end(token_handler.state, -1)
                                if stop_signal:
                                    add_stop_signal(responses, stop_signal)
                                for response in responses:
                                    response.id = unique_id
                                    response.model = model_name
                                    yield response
                            return

                yield from read()

            metrics = self.pipe.get_metrics()

            self.log_inference.info(f"inference finished: "
                                    f"reason={generation_handle.get_status()}, "
                                    f"kv_cache_size={metrics.kv_cache_size_in_bytes / 1024 / 1024:.2f}MB, "
                                    f"cache_size={metrics.cache_size_in_bytes / 1024 / 1024:.2f}MB "
                                    f"cache_usage={metrics.cache_usage}, "
                                    f"max_cache_usage={metrics.max_cache_usage}, "
                                    f"requests={metrics.requests}, "
                                    f"scheduled_requests={metrics.scheduled_requests}"
                                    )
        except Exception as e:
            self.log_inference.error(f"inference error: {e}", exc_info=e)
            msg = f"{e.args}"
            finish_reason: Literal["length", "stop"] = "length" if "max_length > prompt_len" in msg else "stop"
            yield new_stop_response(finish_reason=finish_reason, response_id=response_id,
                                    model=self.config.model_name, content=markdown_bold("ERROR: " + msg))

        status = generation_handle.get_status()
        if status == GenerationStatus.RUNNING:
            self.log_inference.debug(f"request {request_id} is cancelled with status {status}")
            generation_handle.cancel()
        else:
            self.log_inference.info(f"request {request_id} finished with status {status}")

        with self.active_handles_lock:
            del self.active_handles[request_id]

        after_generate_mem = get_current_memory()
        delta = after_generate_mem - before_generate_mem
        log.debug(f"consumed memory: {after_generate_mem:.2f} MB, delta: {delta:.2f} MB")
