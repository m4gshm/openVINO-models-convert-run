import itertools
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
    GenerationConfig, Tokenizer

from agent.common.metric_mem import get_current_memory
from agent.inference.token_handler import TokenHandler, TokenHandlerConfig, StopSignal, \
    get_stop_signal_by_finish_reason, get_finish_str
from agent.openai import GenerateConfig
from agent.openai.chat_api import new_stop_response
from agent.openai.chat_completions_api import CompletionResponse, FunctionDefinition
from agent.openai.engine_rest_common import ControllerConfig, BaseController
from agent.parser import Parser

log = logging.getLogger(__name__)

request_counter = itertools.count(start=0)


class ContinuousBatchingController(BaseController):
    def __init__(self, config: ControllerConfig, parser: Parser, pipe: ContinuousBatchingPipeline,
                 handler_config: TokenHandlerConfig,
                 generate_config: GenerateConfig, router: APIRouter):
        super().__init__(config, parser, pipe.get_tokenizer(), generate_config, router)
        self.pipe = pipe
        self.handler_config = handler_config
        self.generate_config = generate_config
        self.config = config
        self.executor = ThreadPoolExecutor()
        self.active_handles_lock = threading.Lock()
        self.active_handles: dict[int, GenerationHandle] = {}

    def step(self):
        try:
            self.pipe.step()
        except Exception as e:
            log.error(f"pipe step error {e}")
            raise e

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

    def chunk_generator(self, prompt: str, generation_config: GenerationConfig, tokenizer: Tokenizer,
                        init_chat_events: bool, is_stop: Callable[[], bool],
                        function_by_name: dict[str, FunctionDefinition] | None = None
                        ) -> Generator[CompletionResponse, None, None]:
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
        try:
            if self.log_inference.isEnabledFor(logging.DEBUG):
                self.log_inference.debug(
                    f"inference starting with parameters: "
                    f"prompt_tokens={encode_size}, "
                    f"do_sample={generation_config.do_sample}, "
                    f"max_length={max_length}, "
                    f"max_new_tokens={generation_config.max_new_tokens}, "
                    f"do_sample={generation_config.do_sample}, temperature={generation_config.temperature:.2f}, "
                    f"top_p={generation_config.top_p:.2f}, top_k={generation_config.top_k}, "
                    f"min_p={generation_config.min_p:.2f}, repetition_penalty={generation_config.repetition_penalty:.2f}, "
                    f"presence_penalty={generation_config.presence_penalty:.2f}, "
                    f"frequency_penalty={generation_config.frequency_penalty:.2f}"
                )
            else:
                self.log_inference.info(f"inference starting")

            streamer = TokenHandler(tokenizer=tokenizer,
                                    parser=self.parser,
                                    init_chat_events=init_chat_events,
                                    is_stop=is_stop,
                                    supported_functions=function_by_name,
                                    config=self.handler_config)

            generate_result = self.pipe.add_request(request_id=request_id, prompt=prompt,
                                                    generation_config=generation_config, images=[], videos=[])

            with self.active_handles_lock:
                self.active_handles[request_id] = generate_result

            def is_response_timeout(start):
                now_time = time.perf_counter()
                duration = timedelta(seconds=(now_time - start))
                if duration >= self.config.response_timeout:
                    log.warning(f"inference timeout: {duration}")
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
                yield stop_response
            else:
                unique_id = str(uuid.uuid4())

                def read():
                    started = False
                    request_start = time.perf_counter()
                    while True:
                        has_requests = self.pipe.has_non_finished_requests()
                        if not has_requests:
                            yield stop_response
                            return
                        self.step()
                        can_read = generate_result.can_read()
                        if can_read:
                            started = True
                            read_timeout = is_response_timeout(request_start)
                            if read_timeout:
                                yield stop_response
                                return
                            else:
                                generation_handle = generate_result.read()
                                items = generation_handle.items()
                                if len(items) == 0:
                                    log.error("empty generation")
                                    yield stop_response
                                    return
                                for k, generation_output in items:
                                    responses, stop_signal = streamer.handle_token(generation_output.generated_ids)
                                    for response in responses:
                                        response.id = unique_id
                                        response.model = model_name
                                        yield response

                                    finish_reason = generation_output.finish_reason
                                    if finish_reason and finish_reason != GenerationFinishReason.NONE:
                                        stop_signal = get_stop_signal_by_finish_reason(finish_reason)
                                        if not stop_signal:
                                            stop_signal = StopSignal.STOP
                                        yield new_stop_response(finish_reason=get_finish_str(stop_signal),
                                                                model=model_name, response_id=response_id)
                                        return
                                    elif stop_signal:
                                        generate_result.stop(stop_signal.finish_reason)
                                        yield new_stop_response(finish_reason=get_finish_str(stop_signal),
                                                                model=model_name, response_id=response_id)
                                        return
                        elif started:
                            yield stop_response
                            return

                yield from read()

            metrics = self.pipe.get_metrics()
            log.info(f"inference finished: "
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
                                    model=self.config.model_name, content="ERROR:" + msg)
        finally:
            with self.active_handles_lock:
                if self.active_handles.get(request_id):
                    del self.active_handles[request_id]
            after_generate_mem = get_current_memory()
            delta = after_generate_mem - before_generate_mem
            log.debug(f"consumed memory: {after_generate_mem:.2f} MB, delta: {delta:.2f} MB")
