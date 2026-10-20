"""Continuous batching engine controller backed by openvino_genai.ContinuousBatchingPipeline."""
import itertools
import logging
import threading
import time
import uuid
from datetime import timedelta
from typing import Iterable
from typing import Literal

from openai.types.chat import ChatCompletionChunk
# openvino_genai is a compiled extension module; pylint cannot see its members statically.
from openvino_genai.py_openvino_genai import (  # pylint: disable=no-name-in-module
    ContinuousBatchingPipeline, GenerationHandle, GenerationFinishReason,
    GenerationConfig, GenerationStatus)

from agent.common.markdown import markdown_bold
from agent.common.metric_mem import get_current_memory
from agent.inference.token_handler import TokenHandler, TokenHandlerConfig, get_stop_signal_by_finish_reason, \
    StopSignal
from agent.multimodal.media_inputs import MediaInputs
from agent.openai import GenerateOpts
from agent.openai.chat_api import new_stop_response, ROLE_ASSISTANT
from agent.openai.engine_rest_common import ControllerConfig, BaseOVController, add_stop_signal
from agent.parser import Parser, StateEvent

log = logging.getLogger(__name__)

request_counter = itertools.count(start=0)


class ContinuousBatchingController(BaseOVController):
    """Engine controller on top of the openvino_genai continuous batching pipeline."""

    # Wired from server args; grouping the parameters into a config is a separate refactor.
    def __init__(self,  # pylint: disable=too-many-arguments,too-many-positional-arguments
                 config: ControllerConfig, parser: Parser, pipe: ContinuousBatchingPipeline,
                 handler_config: TokenHandlerConfig, stop_signal: threading.Event,
                 generate_opts: GenerateOpts):
        """Wire the CB pipeline and the shared controller state."""
        super().__init__(config, parser, pipe.get_tokenizer(), handler_config, generate_opts, stop_signal)
        self.pipe = pipe
        self.active_handles_lock = threading.Lock()
        self.active_handles: dict[int, GenerationHandle] = {}

    def step(self):
        """Run one pipeline step, logging and re-raising any failure."""
        try:
            self.pipe.step()
        except Exception as e:
            log.error("pipe step error %s", e)
            raise e

    def shutdown(self):
        super().shutdown()
        with self.active_handles_lock:
            pipe = self.pipe
            if pipe is None:
                log.info("pipe already closed")
                return
            active_ids = list(self.active_handles.keys())
            log.info("canceling of active requests %s", active_ids)
            for req_id in active_ids:
                handle = self.active_handles.get(req_id)
                if handle:
                    try:
                        # log
                        handle.cancel()
                    except Exception as e:  # pylint: disable=broad-exception-caught  # one stuck handle must not abort shutdown of the rest
                        log.error("error on pipe handle cancelling: %s", e)
            self.pipe = None
            del pipe

    # Streaming loop over the CB pipeline; splitting it into helpers is a separate refactor.
    def chunk_generator(self, prompt: str,  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
                        generation_config: GenerationConfig, token_handler: TokenHandler,
                        media: MediaInputs) -> Iterable[ChatCompletionChunk]:
        model_name = self.config.model_name or ""

        before_generate_mem = get_current_memory()
        request_id = next(request_counter)

        response_id = str(uuid.uuid4())

        stop_response = new_stop_response(response_id=response_id, model=model_name, role=None)

        generation_handle: GenerationHandle
        try:
            generation_handle = self.pipe.add_request(request_id=request_id, prompt=prompt,
                                                      generation_config=generation_config,
                                                      images=media.images, videos=media.videos)
        except Exception as e:
            log.error("create pipe error %s", e)
            raise e

        with self.active_handles_lock:
            self.active_handles[request_id] = generation_handle

        def is_response_timeout(start):
            now_time = time.perf_counter()
            duration = timedelta(seconds=now_time - start)
            if duration >= self.config.response_timeout:
                log.warning("inference timeout: %s", duration)
                generation_handle.stop(GenerationFinishReason.NONE)
                return True
            return False

        try:
            request_start = time.perf_counter()
            response_timeout = False
            interrupted = False
            while not self.pipe.has_non_finished_requests():
                # an interrupted request (a newer one arrived, the client is gone) must not
                # keep the request scheduled: cancel it to free the device for the new one
                if token_handler.is_stop():
                    log.info("request %s is interrupted before the first token", request_id)
                    generation_handle.cancel()
                    interrupted = True
                    break
                self.step()
                time.sleep(0.2)
                response_timeout = is_response_timeout(request_start)
                if response_timeout:
                    break

            if interrupted or response_timeout:
                yield stop_response
            else:
                def read():  # pylint: disable=too-many-locals,too-many-branches,too-many-statements  # token pump
                    empty_tokens_limit = 100
                    empty_out_counter = 0
                    started = False
                    request_start = time.perf_counter()
                    while True:
                        if token_handler.is_stop():
                            log.info("request %s is interrupted", request_id)
                            generation_handle.cancel()
                            yield stop_response
                            return
                        has_requests = self.pipe.has_non_finished_requests()
                        if not has_requests:
                            if not started:
                                log.debug("has no active requests")
                                yield stop_response
                                return
                            log.debug("has no active requests but current has not been finished")

                        self.step()
                        can_read = generation_handle.can_read()
                        if can_read:
                            started = True
                            if is_response_timeout(request_start):
                                yield stop_response
                                return
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
                            for _k, generation_output in items:
                                generated_ids = generation_output.generated_ids
                                self.log_inference_token_metrics.debug(
                                    "generation_output: ids=%s, score=%s, log_probs=%s",
                                    generated_ids, generation_output.score, generation_output.generated_log_probs)

                                responses, stop_signal = token_handler.handle_tokens(generated_ids)

                                finish_reason = generation_output.finish_reason
                                stop_by_finish = finish_reason and finish_reason != GenerationFinishReason.NONE
                                if not stop_signal and stop_by_finish:
                                    stop_signal = get_stop_signal_by_finish_reason(finish_reason)
                                    log.debug("generation_output has finish_reason '%s' converted to stop_signal '%s'",
                                              finish_reason.name, stop_signal)

                                if stop_signal:
                                    add_stop_signal(responses, stop_signal)

                                for response in responses:
                                    response.id = response_id
                                    response.model = model_name
                                    yield response
                                if stop_signal:
                                    if stop_signal == StopSignal.STOP:
                                        pipe_finish_reason = GenerationFinishReason.STOP
                                    elif stop_signal == StopSignal.TOOL_CALL:
                                        pipe_finish_reason = GenerationFinishReason.TOOL_CALL
                                    elif stop_signal == StopSignal.LENGTH:
                                        pipe_finish_reason = GenerationFinishReason.LENGTH
                                    else:
                                        pipe_finish_reason = GenerationFinishReason.NONE
                                    generation_handle.stop(pipe_finish_reason)
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

            self.log_inference.info(
                "inference finished: token_handler_info='%s', status=%s, kv_cache_size=%.2fMB, "
                "cache_size=%.2fMB cache_usage=%s, max_cache_usage=%s, requests=%s, scheduled_requests=%s",
                token_handler.get_stat_info(), generation_handle.get_status(),
                metrics.kv_cache_size_in_bytes / 1024 / 1024, metrics.cache_size_in_bytes / 1024 / 1024,
                metrics.cache_usage, metrics.max_cache_usage, metrics.requests, metrics.scheduled_requests)
        except Exception as e:  # pylint: disable=broad-exception-caught  # any failure must become an error stop-response
            self.log_inference.error("inference error: %s", e, exc_info=e)
            msg = f"{e.args}"
            finish_reason: Literal["length", "stop"] = "length" if "max_length > prompt_len" in msg else "stop"
            yield new_stop_response(finish_reason=finish_reason, response_id=response_id, role=ROLE_ASSISTANT,
                                    model=model_name, content=markdown_bold("ERROR: " + msg))

        status = generation_handle.get_status()
        if status == GenerationStatus.RUNNING:
            self.log_inference.debug("request %s is cancelled with status %s", request_id, status)
            generation_handle.cancel()
        else:
            self.log_inference.info("request %s finished with status %s", request_id, status)

        with self.active_handles_lock:
            del self.active_handles[request_id]

        after_generate_mem = get_current_memory()
        delta = after_generate_mem - before_generate_mem
        log.debug("consumed memory: %.2f MB, delta: %.2f MB", after_generate_mem, delta)
