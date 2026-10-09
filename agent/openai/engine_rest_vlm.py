import logging
import queue
import threading
import uuid
from collections import abc
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable
from typing import SupportsInt, Literal

from openai.types.chat import ChatCompletionChunk
from openvino_genai import VLMPipeline, GenerationFinishReason, py_openvino_genai, StreamingStatus
from openvino_genai.py_openvino_genai import DecodedResults, LLMPipeline, MeanStdPair, \
    VLMDecodedResults, GenerationConfig

from agent.common.metric_mem import get_current_memory
from agent.inference.token_handler import TokenHandler, TokenHandlerConfig, StopSignal
from agent.multimodal.media_error import MediaUnsupportedError
from agent.multimodal.media_inputs import MediaInputs
from agent.openai import GenerateOpts
from agent.openai.chat_api import new_stop_response, ROLE_ASSISTANT
from agent.openai.engine_rest_common import ControllerConfig, BaseOVController, add_stop_signal, get_tokens_size
from agent.parser import Parser

log = logging.getLogger(__name__)


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
    def __init__(self, config: ControllerConfig, parser: Parser, pipe: VLMPipeline | LLMPipeline,
                 handler_config: TokenHandlerConfig, stop_signal: threading.Event, generate_opts: GenerateOpts):
        super().__init__(config, parser, pipe.get_tokenizer(), handler_config, generate_opts, stop_signal)
        self.pipe = pipe
        self.executor = ThreadPoolExecutor()
        self.request_lock = threading.Lock()

    def chunk_generator(self, prompt: str, generation_config: GenerationConfig,
                        token_handler: TokenHandler, media: MediaInputs) -> Iterable[ChatCompletionChunk]:

        response_id = str(uuid.uuid4())

        with self.request_lock:
            chunk_queue: queue.Queue[ChatCompletionChunk | None] = queue.Queue()
            stop_stream_handling: queue.Queue[bool] = queue.Queue()
            start_stream_handling: queue.Queue[bool] = queue.Queue()
            before_generate_mem = get_current_memory()

            def run_inference():
                try:
                    streamer = StreamerWrapper(token_handler,
                                               start_stream_handling=start_stream_handling,
                                               stop_stream_handling=stop_stream_handling,
                                               chunk_queue=chunk_queue)

                    generate_result = start_generate_result(streamer)
                    chunk_queue.put_nowait(None)

                    metrics = generate_result.perf_metrics if isinstance(generate_result, DecodedResults) else None

                    def to_str(d: MeanStdPair) -> str:
                        return f"std {d.std} , mean {d.mean}"

                    log_msg = (f"inference finished: "
                               f"token_handler_info='{token_handler.get_stat_info()}', ")
                    if isinstance(generate_result, DecodedResults):
                        inference_finish_reasons = generate_result.finish_reasons
                        log_msg += f"reason '{inference_finish_reasons}', "
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
                        self.log_inference.debug(f"{log_msg}\nresult: {texts}")
                    else:
                        self.log_inference.info(log_msg)

                    after_generate_mem = get_current_memory()
                    generate_cost = after_generate_mem - before_generate_mem
                    log.debug(f"consumed memory: {after_generate_mem:.2f} MB, generate delta: {generate_cost:.2f} MB")

                    inference_finish_reason = inference_finish_reasons[0] if inference_finish_reasons else None
                    if inference_finish_reason is None or inference_finish_reason == GenerationFinishReason.NONE:
                        log.info(f"inference finished by unexpected status {inference_finish_reason}")

                except Exception as e:
                    start_stream_handling.put_nowait(True)
                    self.log_inference.error(f"inference error: {e}", exc_info=e)
                    err_str = str(e)
                    finish_reason: Literal[
                        "length", "stop"] = "length" if "<= m_max_prompt_len" in err_str else "stop"
                    chunk_queue.put_nowait(
                        new_stop_response(content=err_str, finish_reason=finish_reason, model=self.config.model_name,
                                          role=ROLE_ASSISTANT))
                    chunk_queue.put_nowait(None)

            def start_generate_result(streamer: StreamerWrapper) -> VLMDecodedResults:
                pipe = self.pipe
                if isinstance(pipe, VLMPipeline):
                    generate_result = generate_vlm(pipe, prompt, generation_config, streamer, media)
                elif isinstance(pipe, LLMPipeline):
                    if not media.is_empty():
                        raise MediaUnsupportedError(
                            "LLM pipeline cannot take media inputs, load the model with --pipe VLM")
                    llm_pipe: LLMPipeline = pipe
                    generate_result = llm_pipe.generate(inputs=prompt, generation_config=generation_config,
                                                        streamer=streamer)
                else:
                    raise NotImplementedError(f"unexpected pipe type {type(pipe)}")
                return generate_result

            try:
                inference_task = self.executor.submit(run_inference)
                start_stream_handling.get()
                stop_inference = False
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
    def __init__(self, streamer: TokenHandler, chunk_queue: queue.Queue[ChatCompletionChunk | None],
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

        responses, stop_signal = self.streamer.handle_tokens(tokens)
        if stop_signal:
            add_stop_signal(responses, stop_signal)

        if responses:
            for response in responses:
                self.chunk_queue.put_nowait(response)

        if stop_signal == StopSignal.STOP:
            return StreamingStatus.STOP
        elif stop_signal == StopSignal.TOOL_CALL:
            return StreamingStatus.TOOL_CALL_STOP
        elif stop_signal == StopSignal.CANCEL:
            return StreamingStatus.CANCEL
        return StreamingStatus.RUNNING
