"""VLM engine controller backed by openvino_genai VLMPipeline, with an LLM-pipe fallback."""
import logging
import queue
import threading
import uuid
from collections import abc
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from typing import SupportsInt, Literal

from openai.types.chat import ChatCompletionChunk
from openvino_genai import VLMPipeline, GenerationFinishReason, py_openvino_genai, StreamingStatus
# openvino_genai is a compiled extension module; pylint cannot see its members statically.
from openvino_genai.py_openvino_genai import (  # pylint: disable=no-name-in-module
    DecodedResults, LLMPipeline, MeanStdPair, VLMDecodedResults, GenerationConfig)

from agent.common.metric_mem import get_current_memory
from agent.inference.token_handler import TokenHandler, TokenHandlerConfig, StopSignal
from agent.multimodal.media_error import MediaUnsupportedError
from agent.multimodal.media_inputs import MediaInputs
from agent.openai import GenerateOpts
from agent.openai.chat_api import new_stop_response, ROLE_ASSISTANT
from agent.openai.engine_rest_common import ControllerConfig, BaseOVController, add_stop_signal
from agent.parser import Parser

log = logging.getLogger(__name__)

# How long one pass of the post-abandonment wait reports before re-logging a progress line.
RESULT_WAIT_TIMEOUT_SECONDS: float = 30


def generate_vlm(pipe: VLMPipeline, prompt: str, generation_config: GenerationConfig,
                 streamer: "StreamerWrapper", media: MediaInputs) -> VLMDecodedResults:
    """Run VLMPipeline generation passing only the media modalities the request carries.

    Prompt text refers to the tensors by OpenVINO media tags, so empty lists are omitted to
    keep compatibility with pipeline versions that do not declare all media parameters.
    """
    kwargs = media.generate_kwargs()
    if kwargs:
        log.info("vlm generation with media: %s", media.summary())
    try:
        return pipe.generate(prompt=prompt, generation_config=generation_config, streamer=streamer, **kwargs)
    except TypeError as e:
        raise TypeError(f"pipeline does not accept media inputs {sorted(kwargs)}: {e}") from e


def vlm_metrics_str(metrics: Any) -> str:
    """Extra VLM metrics of a generation: vision/audio encoding time and image slices."""
    parts: list[str] = []
    for name, getter in (("prepare_embeddings", "get_prepare_embeddings_duration"),
                         ("vision_encoding", "get_vision_encoding_duration"),
                         ("audio_encoding", "get_audio_encoding_duration"),
                         ("text_embedding", "get_text_embedding_duration")):
        method = getattr(metrics, getter, None)
        if not callable(method):
            continue
        value = method()
        mean = getattr(value, "mean", None)
        if mean:
            parts.append(f"{name}={mean:.2f}ms")
    image_slices = getattr(metrics, "get_total_image_slice_count", None)
    if callable(image_slices):
        slices = image_slices()
        if slices:
            parts.append(f"image_slices={slices}")
    return f", {', '.join(parts)}" if parts else ""


class VlmController(BaseOVController):
    """Engine controller running a VLM (or plain LLM) pipeline in a worker executor."""

    # Wired from server args; grouping the parameters into a config is a separate refactor.
    def __init__(self,  # pylint: disable=too-many-arguments,too-many-positional-arguments
                 config: ControllerConfig, parser: Parser, pipe: VLMPipeline | LLMPipeline,
                 handler_config: TokenHandlerConfig, stop_signal: threading.Event,
                 generate_opts: GenerateOpts):
        """Wire the VLM/LLM pipeline and the shared controller state."""
        super().__init__(config, parser, pipe.get_tokenizer(), handler_config, generate_opts, stop_signal)
        self.pipe = pipe
        self.executor = ThreadPoolExecutor()
        self.request_lock = threading.Lock()

    def generate_result(self, prompt: str, generation_config: GenerationConfig,
                        streamer: "StreamerWrapper", media: MediaInputs) -> Any:
        """Run one blocking pipeline.generate call of any supported pipe type."""
        pipe = self.pipe
        if isinstance(pipe, VLMPipeline):
            return generate_vlm(pipe, prompt, generation_config, streamer, media)
        if isinstance(pipe, LLMPipeline):
            if not media.is_empty():
                raise MediaUnsupportedError(
                    "LLM pipeline cannot take media inputs, load the model with --pipe VLM")
            llm_pipe: LLMPipeline = pipe
            return llm_pipe.generate(inputs=prompt, generation_config=generation_config,
                                     streamer=streamer)
        raise NotImplementedError(f"unexpected pipe type {type(pipe)}")

    def wait_for_first_token(self, token_handler: TokenHandler,
                             start_stream_handling: queue.Queue[bool]) -> bool:
        """Block until generate reports its first token.

        Returns False when the request is abandoned while waiting (client disconnect or server
        stop), which is checked every second: a request may queue here behind another
        long-running generation that still holds request_lock.
        """
        while True:
            if token_handler.is_stop():
                log.info("inference stopped before first token")
                return False
            try:
                start_stream_handling.get(timeout=1)
                return True
            except queue.Empty:
                pass

    # Streaming orchestration; splitting it into helpers is a separate refactor.
    def chunk_generator(self, prompt: str, generation_config: GenerationConfig,  # pylint: disable=too-many-statements
                        token_handler: TokenHandler,
                        media: MediaInputs) -> Generator[ChatCompletionChunk, None, None]:

        response_id = str(uuid.uuid4())

        chunk_queue: queue.Queue[ChatCompletionChunk | None] = queue.Queue()
        stop_stream_handling: queue.Queue[bool] = queue.Queue()
        start_stream_handling: queue.Queue[bool] = queue.Queue()

        def run_inference():
            try:
                # request_lock must be held exactly while generate runs: this generator gives up
                # waiting (client disconnect, response timeout) while the pipeline may still be
                # decoding — during prefill the streamer is not called, so a stop flag cannot
                # interrupt generation. Releasing the lock earlier lets the next request start a
                # second concurrent generate, which the pipeline rejects with
                # 'Generate cannot be called while ContinuousBatchingPipeline is already in running state'
                with self.request_lock:
                    streamer = StreamerWrapper(token_handler,
                                               start_stream_handling=start_stream_handling,
                                               stop_stream_handling=stop_stream_handling,
                                               chunk_queue=chunk_queue)
                    before_generate_mem = get_current_memory()
                    generate_result = self.generate_result(prompt, generation_config, streamer, media)
                    self.queue_natural_stop(streamer.stopped_by_streamer, generate_result, chunk_queue)
                chunk_queue.put_nowait(None)

                metrics = generate_result.perf_metrics if isinstance(generate_result, DecodedResults) else None

                def to_str(d: MeanStdPair) -> str:
                    return f"std {d.std} , mean {d.mean}"

                log_msg = (f"inference finished: "
                           f"token_handler_info='{token_handler.get_stat_info()}', ")
                if isinstance(generate_result, DecodedResults):
                    inference_finish_reasons = generate_result.finish_reasons
                    log_msg += f"reason '{[res.name for res in inference_finish_reasons]
                    if isinstance(inference_finish_reasons, list) else inference_finish_reasons}', "
                else:
                    inference_finish_reasons = None

                if metrics:
                    log_msg += (
                        f"num_input_tokens={metrics.get_num_input_tokens()}, "
                        f"generated_tokens={metrics.get_num_generated_tokens()}, "
                        f"generate_duration={to_str(metrics.get_generate_duration())}, "
                        f"inference_duration={to_str(metrics.get_inference_duration())}, "
                        f"ttft={to_str(metrics.get_ttft())}, "
                        f"throughput={to_str(metrics.get_throughput())}")
                    log_msg += vlm_metrics_str(metrics)
                if self.log_inference.isEnabledFor(logging.DEBUG):
                    texts = generate_result.texts if isinstance(generate_result,
                                                                DecodedResults) else generate_result
                    self.log_inference.debug("%s\nresult: %s", log_msg, texts)
                else:
                    self.log_inference.info(log_msg)

                after_generate_mem = get_current_memory()
                generate_cost = after_generate_mem - before_generate_mem
                log.debug("consumed memory: %.2f MB, generate delta: %.2f MB", after_generate_mem, generate_cost)

                inference_finish_reason = inference_finish_reasons[0] if inference_finish_reasons else None
                if inference_finish_reason is None or inference_finish_reason == GenerationFinishReason.NONE:
                    if streamer.stopped_by_streamer:
                        # a tool call, a stop phrase or a cancelled request ends generation via the
                        # streamer, and then finish_reasons legitimately stays empty/NONE
                        self.log_inference.info(
                            "inference stopped by the streamer: token_handler_info='%s'",
                            token_handler.get_stat_info())
                    else:
                        log.warning("inference finished by unexpected status %s", inference_finish_reason)

            except Exception as e:  # pylint: disable=broad-exception-caught  # any failure must reach the client as an error stop-response
                start_stream_handling.put_nowait(True)
                self.log_inference.error("inference error: %s", e, exc_info=e)
                err_str = str(e)
                finish_reason: Literal[
                    "length", "stop"] = "length" if "<= m_max_prompt_len" in err_str else "stop"
                chunk_queue.put_nowait(
                    new_stop_response(content=err_str, finish_reason=finish_reason, model=self.config.model_name,
                                      role=ROLE_ASSISTANT))
                chunk_queue.put_nowait(None)

        try:
            inference_task = self.executor.submit(run_inference)
            stop_inference = not self.wait_for_first_token(token_handler, start_stream_handling)
            while not stop_inference:
                if token_handler.is_stop():
                    log.info("inference stopped by signal")
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

        except Exception as e:  # pylint: disable=broad-exception-caught  # a chunk-processing failure must not kill the generator
            log.error("chunk processing error: %s", e, exc_info=e)

        stop_stream_handling.put_nowait(True)
        # the request may be abandoned while its generate is still queued behind another one or
        # prefilling: such generate cannot be interrupted, so log progress and keep waiting it out
        while not inference_task.done():
            self.log_inference.info("waiting for inference to complete after the generator stopped")
            try:
                inference_task.result(timeout=RESULT_WAIT_TIMEOUT_SECONDS)
            except TimeoutError:
                pass
        log.info("inference handling is done")


    def queue_natural_stop(self, stopped_by_streamer: bool, generate_result: Any,
                           chunk_queue: "queue.Queue[ChatCompletionChunk]") -> None:
        """Queue a finish chunk when the pipeline itself ended the generation.

        EOS or the token budget ends the generate call without a token-handler stop signal, so the
        streamer has not emitted a finish chunk: the real reason (length on the token budget) is
        reported here so the client sees finish_reason=length instead of a default stop.
        """
        if stopped_by_streamer:
            return
        reasons = getattr(generate_result, "finish_reasons", None)
        pipelines_reason = reasons[0] if reasons else None
        finish = "length" if pipelines_reason == GenerationFinishReason.LENGTH else "stop"
        chunk_queue.put_nowait(new_stop_response(finish_reason=finish, role=None,
                                                 model=self.config.model_name))


class StreamerWrapper(py_openvino_genai.StreamerBase):
    # pylint: disable=c-extension-no-member  # the base class comes from a compiled extension
    """Bridges pipeline token callbacks into the response chunk queue."""

    def __init__(self, streamer: TokenHandler, chunk_queue: queue.Queue[ChatCompletionChunk | None],
                 stop_stream_handling: queue.Queue[bool], start_stream_handling: queue.Queue[bool]):
        """Bind the token handler and the queues shared with the generator."""
        super().__init__()
        self.streamer = streamer
        self.started = False
        self.stopped_by_streamer = False
        self.stop_stream_handling = stop_stream_handling
        self.start_stream_handling = start_stream_handling
        self.chunk_queue = chunk_queue

    def end(self) -> None:
        """Required StreamerBase hook: nothing to do on stream end."""

    def write(self, tokens: abc.Sequence[SupportsInt]) -> StreamingStatus:
        """Handle one token batch: emit chunks and map the stop signal to StreamingStatus."""
        if not self.started:
            self.start_stream_handling.put(True)
            self.started = True
        if not self.stop_stream_handling.empty() and self.stop_stream_handling.get_nowait():
            log.debug("stream finished by stop signal")
            self.stopped_by_streamer = True
            return StreamingStatus.STOP

        responses, stop_signal = self.streamer.handle_tokens(tokens)
        if stop_signal:
            add_stop_signal(responses, stop_signal)

        if responses:
            for response in responses:
                self.chunk_queue.put_nowait(response)

        if stop_signal == StopSignal.STOP:
            self.stopped_by_streamer = True
            return StreamingStatus.STOP
        if stop_signal == StopSignal.TOOL_CALL:
            self.stopped_by_streamer = True
            return StreamingStatus.TOOL_CALL_STOP
        if stop_signal == StopSignal.CANCEL:
            self.stopped_by_streamer = True
            return StreamingStatus.CANCEL
        return StreamingStatus.RUNNING
