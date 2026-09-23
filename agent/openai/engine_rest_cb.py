import itertools
import logging
import threading
import time
import uuid
from datetime import timedelta
from typing import Iterable
from typing import Literal

from openai.types.chat import ChatCompletionChunk
from openvino_genai.py_openvino_genai import ContinuousBatchingPipeline, GenerationHandle, GenerationFinishReason, \
    GenerationConfig, GenerationStatus
from starlette.responses import JSONResponse

from agent.common.metric_mem import get_current_memory
from agent.inference.token_handler import TokenHandler, TokenHandlerConfig, get_stop_signal_by_finish_reason, \
    markdown_bold, StopSignal
from agent.openai import GenerateOpts
from agent.openai.chat_api import new_stop_response, ROLE_ASSISTANT
from agent.openai.engine_rest_common import ControllerConfig, BaseController, add_stop_signal, get_tokens_size
from agent.parser import Parser, StateEvent

log = logging.getLogger(__name__)

request_counter = itertools.count(start=0)


class ContinuousBatchingController(BaseController):
    def __init__(self, config: ControllerConfig, parser: Parser, pipe: ContinuousBatchingPipeline,
                 handler_config: TokenHandlerConfig, stop_signal: threading.Event, generate_opts: GenerateOpts):
        super().__init__(config, parser, pipe.get_tokenizer(), handler_config, generate_opts, stop_signal)
        self.pipe = pipe
        self.active_handles_lock = threading.Lock()
        self.active_handles: dict[int, GenerationHandle] = {}

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

    def chunk_generator(self, prompt: str, generation_config: GenerationConfig, token_handler: TokenHandler) -> \
            Iterable[ChatCompletionChunk]:
        model_name = self.config.model_name

        before_generate_mem = get_current_memory()
        request_id = next(request_counter)

        response_id = str(uuid.uuid4())
        prompt_tokens_amount = get_tokens_size(self.tokenizer, prompt)
        max_length = generation_config.max_length

        over_limit_response = self.check_prompt_limit(max_length=max_length, encode_size=prompt_tokens_amount,
                                                      response_id=response_id)
        if over_limit_response:
            yield over_limit_response
            return

        stop_response = new_stop_response(response_id=response_id, model=model_name, role=None)

        if self.log_inference.isEnabledFor(logging.DEBUG):
            self.log_inference.debug(
                f"inference start: request={request_id}, "
                f"pipe_type={type(self.pipe)}, "
                f"prompt_tokens_amount={prompt_tokens_amount}, "
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
                                            response_id=response_id, model=model_name, role=ROLE_ASSISTANT)
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
                                        response.id = response_id
                                        response.model = model_name
                                        yield response
                                    if stop_signal:
                                        generation_handle.stop(
                                            GenerationFinishReason.STOP if stop_signal == StopSignal.STOP else
                                            GenerationFinishReason.TOOL_CALL if stop_signal == StopSignal.TOOL_CALL else
                                            GenerationFinishReason.LENGTH if stop_signal == StopSignal.LENGTH else
                                            GenerationFinishReason.NONE)
                                        return
                        elif started:
                            self.log_inference.debug("no more reads")
                            if token_handler.state.has_event(StateEvent.CONVERSATION):
                                self.log_inference.debug("force end conversation")
                                responses, stop_signal = token_handler.conversation_end(token_handler.state, -1)
                                if stop_signal:
                                    add_stop_signal(responses, stop_signal)
                                for response in responses:
                                    response.id = response_id
                                    response.model = model_name
                                    yield response
                            return

                yield from read()

            metrics = self.pipe.get_metrics()

            self.log_inference.info(f"inference finished: "
                                    f"token_handler_info='{token_handler.get_stat_info()}', "
                                    f"status={generation_handle.get_status()}, "
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
            yield new_stop_response(finish_reason=finish_reason, response_id=response_id, role=ROLE_ASSISTANT,
                                    model=model_name, content=markdown_bold("ERROR: " + msg))

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

    async def slots(self) -> JSONResponse:
        """slots endpoint — returns detailed slot status for continuous batching.
        
        Uses ContinuousBatchingController's active_handles and pipeline metrics
        to provide real-time slot information.
        """
        metrics = self.pipe.get_metrics()
        
        with self.active_handles_lock:
            active_ids = list(self.active_handles.keys())
            
            # Get real OpenVINO GenAI metrics
            kv_cache_size_mb = metrics.kv_cache_size_in_bytes / 1024 / 1024 if hasattr(metrics, 'kv_cache_size_in_bytes') else 0
            cache_size_mb = metrics.cache_size_in_bytes / 1024 / 1024 if hasattr(metrics, 'cache_size_in_bytes') else 0
            cache_usage = metrics.cache_usage if hasattr(metrics, 'cache_usage') else 0
            max_cache_usage = metrics.max_cache_usage if hasattr(metrics, 'max_cache_usage') else 0
            requests = metrics.requests if hasattr(metrics, 'requests') else 0
            scheduled_requests = metrics.scheduled_requests if hasattr(metrics, 'scheduled_requests') else 0
            
            # Calculate cache utilization percentage
            if cache_size_mb > 0 and cache_usage is not None:
                cache_utilization_pct = (cache_usage / cache_size_mb * 100) if cache_size_mb > 0 else 0.0
            else:
                cache_utilization_pct = 0.0
            
            # Build slot list from active handles
            slots_list = []
            for req_id, handle in self.active_handles.items():
                status = handle.get_status()
                request_id = handle.get_request_id() if hasattr(handle, 'get_request_id') else req_id
                
                slot_info = {
                    "id": req_id,
                    "request_id": request_id,
                    "state": str(status),
                    "prompt_len": 0,
                    "kv_cache_blocks": 0
                }
                
                # Try to get additional info from handle
                if hasattr(handle, 'get_prompt_length'):
                    slot_info["prompt_len"] = handle.get_prompt_length()
                if hasattr(handle, 'get_generated_tokens'):
                    slot_info["generation_tokens"] = handle.get_generated_tokens()
                if hasattr(handle, 'get_kv_cache_blocks'):
                    slot_info["kv_cache_blocks"] = handle.get_kv_cache_blocks()
                
                slots_list.append(slot_info)
        
        return JSONResponse(content={
            "slots": slots_list,
            "kv_cache_size_mb": round(kv_cache_size_mb, 2),
            "cache_size_mb": round(cache_size_mb, 2),
            "cache_usage": cache_usage,
            "max_cache_usage": max_cache_usage,
            "cache_utilization_pct": round(cache_utilization_pct, 2),
            "active_requests": requests,
            "scheduled_requests": scheduled_requests,
            "active_slots": len(slots_list)
        })
