"""OpenVINO mode entry point: model discovery, engine selection and CLI wiring."""
import logging.config
import sys
import threading
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from openvino_genai import SchedulerConfig, Tokenizer, py_openvino_genai

from agent.common.log import logging_config
from agent.common.metric_mem import get_current_memory
from agent.inference.token_handler import TokenHandlerConfig
from agent.multimodal.model_capabilities import read_media_capabilities, resolve_modalities
from agent.openai import get_default_generate_opts, GenerateOpts, get_default_scheduler_opts, SchedulerOpts
from agent.openai.engine_rest_cb import ContinuousBatchingController
from agent.openai.engine_rest_common import ControllerConfig
from agent.openai.engine_rest_vlm import VlmController
from agent.openvino.model_config import default_pipe_of, detect_parser_type, or_default_pipe, \
    parser_type_of, read_model_config
from agent.common.cli import enum_value, enum_values
from agent.openvino.options import AttentionBackend, AttentionHint, DeviceType, KvCachePrecision, Level, MoEHint, \
    NpuCompilerType, NpuGenerateHint, ParserType, PerformanceHint, Pipe, PrefillHint, Turn, YesNo
from agent.openvino.pipeline_properties import apply_common_properties, build_cpu_properties, \
    build_gpu_properties, build_npu_properties, device_value, select_pipeline_properties
from agent.openvino.scheduler import build_scheduler_config
from agent.parser import Parser
from agent.parser.gemma4 import Gemma4ChannelParser
from agent.parser.lfm2 import Lfm2Parser
from agent.parser.qwen2 import Qwen2Parser
from agent.parser.qwen3 import Qwen3MoeParser
from agent.server import new_app, serve_until_stop, stop_signal as app_stop_signal

log = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = "./models"
DEFAULT_MODELS_CACHE_DIR = "./models_cache"


@dataclass
class ModelContext:
    """The model itself: where it lives and what it declares about itself."""

    model_path: Path
    model_name: str
    model_cache_dir: str
    model_architectures: set[str]
    media_capabilities: Any
    generate_opts: GenerateOpts
    max_prompt_len: int | None


@dataclass
class EngineSetup:  # pylint: disable=too-many-instance-attributes  # config bundle of the engine factories
    """Everything the engine factories of the OpenVINO mode need."""

    context: ModelContext
    device: DeviceType
    pipe: Pipe
    parser_type: ParserType | None
    scheduler_config: SchedulerConfig
    pipeline_properties: dict[str, Any]
    controller_config: ControllerConfig
    handler_config: TokenHandlerConfig

    @property
    def is_device_npu(self) -> bool:
        """Whether the pipeline is compiled for the NPU."""
        return self.device == DeviceType.NPU


def run_openvino(args):
    """Run application in OpenVINO mode with local model loading.

    Entry point for running a local OpenVINO GenAI model. Loads the model
    from disk (GGUF or directory), configures the pipeline, scheduler, and
    parser based on detected architecture, then starts a uvicorn server.
    """
    setup = build_engine_setup(args)
    app = init_sequential_engine(
        controller_config=setup.controller_config,
        model_path=str(setup.context.model_path),
        device=device_value(setup.device),
        vlm=setup.pipe == Pipe.VLM,
        parser_type=setup.parser_type,
        scheduler_config=None if setup.is_device_npu else setup.scheduler_config,
        generate_opts=setup.context.generate_opts,
        handler_config=setup.handler_config,
        pipeline_properties=setup.pipeline_properties,
        stop_signal=app_stop_signal,
    ) if setup.pipe != Pipe.CB else init_continuous_batching_engine(
        controller_config=setup.controller_config,
        model_path=str(setup.context.model_path),
        device=device_value(setup.device),
        parser_type=setup.parser_type,
        generate_opts=setup.context.generate_opts,
        handler_config=setup.handler_config,
        scheduler_config=setup.scheduler_config,
        pipeline_properties=setup.pipeline_properties,
        tokenizer_properties={},
        stop_signal=app_stop_signal,
    )

    # --- Start uvicorn server in a daemon thread ---
    serve_until_stop(app, args.host, args.port, app_stop_signal, log)


def build_engine_setup(args) -> EngineSetup:
    """Resolve the model, the engine options and the properties of the pipeline."""
    context = load_model_context(args)
    device = DeviceType[args.device]

    scheduler_opts = load_scheduler_opts(args.scheduler_config_file)
    scheduler_config = build_scheduler_config(scheduler_opts, context.max_prompt_len)

    pipe, parser_type = resolve_pipe_and_parser(args, context.model_architectures,
                                                context.media_capabilities)
    log.info("model: path='%s', architectures=%s, pipe=%s, parser_type='%s'",
             context.model_path, context.model_architectures, pipe, parser_type)
    log.debug("cache dir %s", context.model_cache_dir)

    pipeline_properties = resolve_pipeline_properties(args, context, device, pipe, parser_type)
    controller_config = resolve_controller_config(args, context, pipe,
                                                   load_chat_template(args.chat_template_file))
    handler_config = TokenHandlerConfig(is_detect_looped_inference=args.detect_looped_inference != Turn.off.value)

    log.info("model=%s, device=%s, properties=%s, generate_opts=%s, scheduler_config=%s",
             context.model_path, device, pipeline_properties,
             context.generate_opts.model_dump(exclude_unset=True, exclude_none=True), scheduler_config)

    return EngineSetup(context=context, device=device, pipe=pipe, parser_type=parser_type,
                       scheduler_config=scheduler_config, pipeline_properties=pipeline_properties,
                       controller_config=controller_config, handler_config=handler_config)


def load_model_context(args) -> ModelContext:
    """Resolve the model folder and read what it declares about itself."""
    model_path, model_name = resolve_model_path(args.model, args.models_dir)
    model_cache_dir = f"{args.models_cache_dir}/{model_name}"
    setup_logging(args, model_name)

    model_config = read_model_config(model_path)
    media_capabilities = read_media_capabilities(model_path)
    log.info("model media capabilities: %s", media_capabilities.describe())

    generate_opts = load_generate_opts(args.generate_config_file)
    max_prompt_len = resolve_max_prompt_len(args.max_prompt_len, generate_opts,
                                             model_config.max_position_embeddings)
    generate_opts.max_prompt_tokens = max_prompt_len

    return ModelContext(model_path=model_path, model_name=model_name, model_cache_dir=model_cache_dir,
                         model_architectures=model_config.architectures,
                         media_capabilities=media_capabilities,
                         generate_opts=generate_opts, max_prompt_len=max_prompt_len)


def setup_logging(args, model_name: str) -> None:
    """Configure the logging of the server into the log directory of the model."""
    logs_dir = f"./logs/{model_name}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"
    log.info("logs dir %s", logs_dir)
    logging.config.dictConfig(logging_config(logs_dir))
    log.info("server starting with arguments application arguments: %s", args)


def resolve_pipeline_properties(args, context: ModelContext, device: DeviceType, pipe: Pipe,
                                parser_type: ParserType | None) -> dict[str, Any]:
    """Build the pipeline properties of the device, including the optional draft model."""
    attention_backend = resolve_attention_backend(args.attention_backend, pipe,
                                                   device == DeviceType.NPU, parser_type)
    pipeline_properties = build_pipeline_properties(args, device, context.model_cache_dir,
                                                    context.max_prompt_len, attention_backend)
    if args.draft_model:
        draft_model_path = str(Path(f"{args.models_dir}/{args.draft_model}"))
        pipeline_properties["draft_model"] = new_draft_model(args, draft_model_path,
                                                             context.model_cache_dir,
                                                             device_value(device))
    if args.prompt_lookup == Turn.on.value:
        pipeline_properties["prompt_lookup"] = True
    return pipeline_properties


def resolve_controller_config(args, context: ModelContext, pipe: Pipe,
                              chat_template: str) -> ControllerConfig:
    """Build the configuration shared by all endpoints of the engine."""
    media_capabilities = context.media_capabilities
    modalities = resolve_modalities(media_capabilities, context.model_architectures, enum_value(pipe))
    log.info("input modalities: %s, pipe=%s, model media: %s",
             modalities.names(), enum_value(pipe), media_capabilities.describe())
    if media_capabilities.has_media and not modalities.media_names():
        log.warning("model has media encoders (%s) but the engine accepts text only, "
                    "media requests will be rejected", media_capabilities.describe())

    interrupt_completions = args.interrupt_completions == enum_value(Turn.on)
    log.info("interrupt mode of /v1/completions: %s", interrupt_completions)

    return ControllerConfig(model_name=args.model, max_prompt_len=context.max_prompt_len,
                            model_architectures=context.model_architectures,
                            is_fix_tool_type=args.fix_tool_type != Turn.off.value,
                            chat_template=chat_template,
                            is_detect_cycled_tool_call=args.detect_cycled_tool_call != Turn.off.value,
                            modalities=set(modalities.names()),
                            max_media_bytes=args.max_media_mb * 1024 * 1024,
                            max_media_items=args.max_media_items,
                            media_audio_sample_rate=media_capabilities.audio_sample_rate,
                            allow_local_media_files=args.allow_local_media_files == enum_value(Turn.on),
                            interrupt_completions=interrupt_completions)


def resolve_model_path(model: str, models_dir: str) -> tuple[Path, str]:
    """Resolve the model path and the display name of the model.

    An absolute path is taken as it is; a relative one is looked up in the
    models directory.
    """
    model_path = Path(model)
    if model_path.is_absolute():
        if model_path.is_file():
            # remove gguf ext
            model_name = model_path.with_suffix("").name
        else:
            model_name = model_path.name
    else:
        model_name = model
        model_path = Path(f"{models_dir}/{model}")
    if not model_path.exists():
        log.error("file or directory doesn't exists: '%s'", model_path)
        sys.exit(1)
    return model_path, model_name


def load_generate_opts(generate_opts_file: str | None) -> GenerateOpts:
    """Load the generation options of the server, or take the built-in defaults."""
    if not generate_opts_file:
        return get_default_generate_opts()
    log.info("load %s", generate_opts_file)
    try:
        with open(generate_opts_file, "r", encoding="utf-8") as file:
            return GenerateOpts.model_validate_json(file.read())
    except FileNotFoundError as e:
        log.error("%s", e)
        raise e


def load_scheduler_opts(scheduler_opts_file: str | None) -> SchedulerOpts:
    """Load the scheduler options of the server, or take the built-in defaults."""
    if not scheduler_opts_file:
        return get_default_scheduler_opts()
    log.info("load %s", scheduler_opts_file)
    try:
        with open(scheduler_opts_file, "r", encoding="utf-8") as file:
            return SchedulerOpts.model_validate_json(file.read())
    except FileNotFoundError as e:
        log.error("%s", e)
        raise e


def load_chat_template(chat_template_file: str | None) -> str:
    """Load the chat template of the model, empty when there is none."""
    if not chat_template_file:
        return ''
    log.info("load %s", chat_template_file)
    try:
        with open(chat_template_file, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError as e:
        log.error("%s", e)
        raise e


def resolve_max_prompt_len(arg_max_prompt_len: int | None, generate_opts: GenerateOpts,
                            max_position_embeddings: int | None) -> int | None:
    """Context length of the server: the CLI switch, the options file or the model itself."""
    if arg_max_prompt_len:
        return arg_max_prompt_len
    default_max_prompt_tokens = get_default_generate_opts().max_prompt_tokens
    max_prompt_len = generate_opts.max_prompt_tokens or default_max_prompt_tokens
    return max_prompt_len if max_prompt_len else max_position_embeddings


def resolve_pipe_and_parser(args, model_architectures: set[str], media_capabilities) -> tuple[Pipe, ParserType | None]:
    """Select the pipeline and the parser of the model.

    The ``--pipe`` and ``--parser`` switches win; otherwise the architecture of
    the model decides, and a model with media encoders is served as a VLM.
    """
    pipe: Pipe | None = Pipe[args.pipe] if args.pipe else None
    parser_type = parser_type_of(args.parser) or detect_parser_type(model_architectures)
    if parser_type:
        pipe = or_default_pipe(pipe, default_pipe_of(parser_type, model_architectures))

    if not pipe and media_capabilities.has_media:
        pipe = or_default_pipe(pipe, Pipe.VLM)
        log.info("pipe selected by model media capabilities: %s", enum_value(pipe))

    if not pipe:
        log.error("need define --pipe for model architectures=%s", model_architectures)
        sys.exit(1)

    return pipe, parser_type


def new_model_parser(parser_type: ParserType | None, tokenizer: Tokenizer | None = None) -> Parser:
    """Instantiate the parser of the detected model family."""
    if parser_type == ParserType.qwen3moe:
        return Qwen3MoeParser(tokenizer)
    if parser_type == ParserType.gemma4:
        return Gemma4ChannelParser(tokenizer)
    if parser_type == ParserType.qwen2:
        return Qwen2Parser(tokenizer)
    if parser_type == ParserType.lfm2:
        return Lfm2Parser(tokenizer)
    return Parser(tokenizer)


def resolve_attention_backend(attention_backend: str | None, pipe: Pipe, is_device_npu: bool,
                               parser_type: ParserType | None) -> str | None:
    """Select the attention kernel of the pipeline.

    The continuous batching pipeline does not support the PA kernel, and the NPU
    and the gemma4 models use their own default, so the kernel is only forced
    for the plain CPU and GPU pipelines.
    """
    is_cb = pipe == Pipe.CB
    is_gemma4 = parser_type == ParserType.gemma4
    if is_cb and attention_backend == enum_value(AttentionBackend.PA):
        log.info("erase attention_backend=%s for unsupported pipe=%s", attention_backend, enum_value(pipe))
        return attention_backend
    if attention_backend is None and not (is_cb or is_device_npu or is_gemma4):
        attention_backend = enum_value(AttentionBackend.PA)
        log.info("force attention_backend=%s", attention_backend)
    return attention_backend


def build_pipeline_properties(args, device: DeviceType, model_cache_dir: str,
                              max_prompt_len: int | None,
                              attention_backend: str | None) -> dict[str, Any]:
    """Build the pipeline properties of the target device."""
    cpu_properties = build_cpu_properties(model_cache_dir, args)
    gpu_properties = build_gpu_properties(model_cache_dir, args)
    npu_properties = build_npu_properties(model_cache_dir, args, max_prompt_len)

    pipeline_properties = select_pipeline_properties(device, cpu_properties, gpu_properties, npu_properties)
    return apply_common_properties(pipeline_properties, args.kv_cache_precision, attention_backend)


def new_draft_model(args, draft_model_path: str, model_cache_dir: str, device: str):
    """Build the optional draft model of the speculative decoding."""
    draft_properties: dict[str, Any] = {
        "CACHE_DIR": model_cache_dir
    }
    if args.eagle3_mode == Turn.on.value:
        draft_properties["eagle3_mode"] = True
    if args.mtp_mode == Turn.on.value:
        draft_properties["mtp_mode"] = True
    if args.dflash_mode == Turn.on.value:
        draft_properties["dflash_mode"] = True
    draft_scheduler_config = SchedulerConfig()
    # pylint: disable-next=c-extension-no-member  # pybind11 module: attribute exists at runtime
    return py_openvino_genai.draft_model(models_path=draft_model_path, device=device,
                                          scheduler_config=draft_scheduler_config, **draft_properties)


# Pipeline factory over the pybind11 openvino_genai API; kwargs mirror its constructors.
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def init_continuous_batching_engine(controller_config: ControllerConfig,
                                    handler_config: TokenHandlerConfig,
                                    model_path: str, device: str, parser_type: ParserType | None,
                                    stop_signal: threading.Event,
                                    # pylint: disable-next=c-extension-no-member  # pybind11 module
                                    scheduler_config=py_openvino_genai.SchedulerConfig(),
                                    generate_opts=GenerateOpts(),
                                    pipeline_properties: dict[str, Any] | None = None,
                                    tokenizer_properties: dict[str, Any] | None = None,
                                    vision_encoder_properties: dict[str, Any] | None = None) -> FastAPI:
    """Initialize a ContinuousBatchingPipeline-based engine.

    Creates a FastAPI app backed by OpenVINO's ContinuousBatchingPipeline,
    which supports dynamic batching of concurrent requests.

    Args:
        controller_config: Controller configuration (model name, max prompt length, etc.)
        handler_config: Token handler configuration for streaming and tool handling
        model_path: Path to the model directory or GGUF file
        device: Target device string (e.g., 'CPU', 'GPU', 'NPU')
        parser_type: Model family of the parser to instantiate
        stop_signal: Event to signal shutdown
        scheduler_config: Scheduler configuration (optional, defaults provided)
        generate_opts: Generation options (temperature, max tokens, etc.)
        pipeline_properties: OpenVINO pipeline properties dict (optional)
        tokenizer_properties: Tokenizer-specific properties (optional)
        vision_encoder_properties: Vision encoder properties for VLM models (optional)

    Returns:
        FastAPI: Initialized FastAPI application with ContinuousBatchingController
    """
    # --- Track initial memory usage ---
    start_mem = get_current_memory()
    log.debug("consumed memory: %.2f MB", start_mem)

    # --- Ensure all property dicts are initialized ---
    if not pipeline_properties:
        pipeline_properties = {}
    if not tokenizer_properties:
        tokenizer_properties = {}
    if not vision_encoder_properties:
        vision_encoder_properties = {}

    # --- Instantiate the ContinuousBatchingPipeline ---
    try:
        # pylint: disable-next=c-extension-no-member  # pybind11 module: attribute exists at runtime
        pipe = py_openvino_genai.ContinuousBatchingPipeline(models_path=model_path,
                                                            scheduler_config=scheduler_config,
                                                            device=device,
                                                            properties=pipeline_properties,
                                                            tokenizer_properties=tokenizer_properties,
                                                            vision_encoder_properties=vision_encoder_properties)
        log.info("model loaded successfully, pipe %s", type(pipe))

        # --- Log memory delta after model load ---
        loaded_pipe_mem = get_current_memory()
        delta = loaded_pipe_mem - start_mem

        log.debug("consumed memory: %.2f MB, delta: %.2f MB", loaded_pipe_mem, delta)
    except Exception as e:  # pylint: disable=broad-exception-caught  # any load failure must exit with a clear log
        log.error("instantiate pipeline error: %s", e, exc_info=e)
        sys.exit(1)

    # --- Wire controller to FastAPI app ---
    parser = new_model_parser(parser_type, pipe.get_tokenizer())
    return new_app(ContinuousBatchingController(config=controller_config,
                                                parser=parser, pipe=pipe,
                                                generate_opts=generate_opts,
                                                handler_config=handler_config,
                                                stop_signal=stop_signal))


# Pipeline factory over the pybind11 openvino_genai API; kwargs mirror its constructors.
# pylint: disable=too-many-arguments,too-many-positional-arguments
def init_sequential_engine(controller_config: ControllerConfig,
                           handler_config: TokenHandlerConfig,
                           model_path: str,
                           device: str, vlm: bool, parser_type: ParserType | None,
                           stop_signal: threading.Event,
                           scheduler_config: py_openvino_genai.SchedulerConfig | None = None,
                           generate_opts=GenerateOpts(),
                           pipeline_properties: dict[str, Any] | None = None) -> FastAPI:
    """Initialize a sequential LLMPipeline/VLMPipeline-based engine.

    Creates a FastAPI app backed by OpenVINO's LLMPipeline or VLMPipeline
    (for vision-language models), which processes requests sequentially.

    Args:
        controller_config: Controller configuration (model name, max prompt length, etc.)
        handler_config: Token handler configuration for streaming and tool handling
        model_path: Path to the model directory or GGUF file
        device: Target device string (e.g., 'CPU', 'GPU', 'NPU')
        vlm: If True, use VLMPipeline; otherwise use LLMPipeline
        parser_type: Model family of the parser to instantiate
        stop_signal: Event to signal shutdown
        scheduler_config: Optional scheduler configuration
        generate_opts: Generation options (temperature, max tokens, etc.)
        pipeline_properties: OpenVINO pipeline properties dict (optional)

    Returns:
        FastAPI: Initialized FastAPI application with VlmController
    """
    # --- Prepare pipeline properties ---
    if not pipeline_properties:
        pipeline_properties = {}
    if scheduler_config:
        pipeline_properties["scheduler_config"] = scheduler_config

        # --- Track initial memory usage ---
    start_mem = get_current_memory()
    log.debug("consumed memory: %.2f MB", start_mem)

    # --- Instantiate LLMPipeline or VLMPipeline based on vlm flag ---
    pipe = (
        # pylint: disable-next=c-extension-no-member  # pybind11 module: attribute exists at runtime
        py_openvino_genai.VLMPipeline(models_path=model_path, device=device, **pipeline_properties) if vlm else
        # pylint: disable-next=c-extension-no-member  # pybind11 module: attribute exists at runtime
        py_openvino_genai.LLMPipeline(models_path=model_path, device=device, **pipeline_properties)
    )

    log.info("model loaded successfully, pipe %s", type(pipe))

    # --- Log memory delta after model load ---
    loaded_pipe_mem = get_current_memory()
    delta = loaded_pipe_mem - start_mem

    log.debug("consumed memory: %.2f MB, delta: %.2f MB", loaded_pipe_mem, delta)

    # --- Wire VlmController to FastAPI app ---
    parser = new_model_parser(parser_type, pipe.get_tokenizer())
    return new_app(VlmController(config=controller_config,
                                 parser=parser, pipe=pipe,
                                 generate_opts=generate_opts,
                                 handler_config=handler_config,
                                 stop_signal=stop_signal))


def add_openvino_args(args_parser: ArgumentParser):
    """Register CLI arguments of the OpenVINO mode."""
    args_parser.add_argument("--host", default="127.0.0.1", help="%(default)s")
    args_parser.add_argument("--port", type=int, default=8888, help="%(default)s")
    args_parser.add_argument("--models_dir", type=str, default=DEFAULT_MODELS_DIR, required=False, help="%(default)s")
    args_parser.add_argument("--models_cache_dir", type=str, default=DEFAULT_MODELS_CACHE_DIR, help="%(default)s")
    args_parser.add_argument("--model", type=str, default=None, help="%(default)s")
    args_parser.add_argument("--draft_model", type=str, help="%(default)s")

    args_parser.add_argument("--device", type=str, required=False,
                             default=enum_value(DeviceType.GPU), choices=enum_values(DeviceType), help="%(default)s")
    args_parser.add_argument("--performance_hint", type=str, required=False,
                             default=enum_value(PerformanceHint.LATENCY), choices=enum_values(PerformanceHint),
                             help="%(default)s")
    args_parser.add_argument("--parser", type=str, required=False,
                             default=None, choices=enum_values(ParserType), help="%(default)s")
    args_parser.add_argument("--pipe", type=str, required=False,
                             default=None, choices=enum_values(Pipe), help="%(default)s")
    args_parser.add_argument("--attention_backend", type=str, required=False,
                             default=None, choices=enum_values(AttentionBackend), help="%(default)s")
    args_parser.add_argument("--max_prompt_len", type=int, required=False, default=None, help="%(default)s")
    # args_parser.add_argument("--max_generation_token_len", type=int, required=False, default=None, help="%(default)s")
    args_parser.add_argument("--kv_cache_precision", type=str, required=False,
                             default=None, choices=enum_values(KvCachePrecision), help="%(default)s")

    args_parser.add_argument("--chat_template_file", type=str, required=False, default=None, help="%(default)s")
    args_parser.add_argument("--fix_tool_type", type=str, required=False,
                             default=None, choices=enum_values(Turn), help="%(default)s")
    args_parser.add_argument("--detect_cycled_tool_call", type=str, required=False,
                             default=None, choices=enum_values(Turn), help="%(default)s")
    args_parser.add_argument("--detect_looped_inference", type=str, required=False,
                             default=None, choices=enum_values(Turn), help="%(default)s")
    args_parser.add_argument("--generate_config_file", type=str, required=False,
                             default=".config/generate_config.json",
                             help="%(default)s")
    args_parser.add_argument("--scheduler_config_file", type=str, required=False,
                             default=".config/scheduler_config.json",
                             help="%(default)s")
    args_parser.add_argument("--npu_compiler_type", type=str, required=False,
                             default=enum_value(NpuCompilerType.PREFER_PLUGIN), choices=enum_values(NpuCompilerType),
                             help="%(default)s")
    args_parser.add_argument("--npu_generate_hint", type=str, required=False,
                             default=enum_value(NpuGenerateHint.BEST_PERF), choices=enum_values(NpuGenerateHint),
                             help="%(default)s")
    args_parser.add_argument("--npu_prefill_hint", type=str, required=False,
                             default=enum_value(PrefillHint.DYNAMIC), choices=enum_values(PrefillHint),
                             help="%(default)s")
    args_parser.add_argument("--npuw_llm_prefill_attention_hint", type=str, required=False,
                             default=None, choices=enum_values(AttentionHint),
                             help="%(default)s")
    args_parser.add_argument("--npuw_llm_generate_attention_hint", type=str, required=False,
                             default=None, choices=enum_values(AttentionHint),
                             help="%(default)s")
    args_parser.add_argument("--npuw_llm_prefill_moe_hint", type=str, required=False,
                             default=None, choices=enum_values(MoEHint),
                             help="%(default)s")
    args_parser.add_argument("--npu_compilation_mode_params", type=str, required=False,
                             default=None, help="%(default)s")
    args_parser.add_argument("--npuw_llm_generate_moe_hint", type=str, required=False,
                             default=None, choices=enum_values(MoEHint),
                             help="%(default)s")
    args_parser.add_argument("--npuw_llm", type=str, required=False,
                             default=enum_value(YesNo.YES), choices=enum_values(YesNo),
                             help="%(default)s")
    args_parser.add_argument("--npuw_pa", type=str, required=False,
                             default=None, choices=enum_values(YesNo),
                             help="%(default)s")
    args_parser.add_argument("--npuw_llm_propagate_slice_up", type=str, required=False,
                             default=None, choices=enum_values(YesNo),
                             help="%(default)s")

    args_parser.add_argument("--npuw_llm_enable_prefix_caching", type=str, required=False,
                             default=None, choices=enum_values(YesNo),
                             help="%(default)s")
    args_parser.add_argument("--npuw_llm_prefill_chunk_size", type=int, required=False,
                             default=1024, help="%(default)s")
    args_parser.add_argument("--npuw_llm_enable_continuous_prefill", type=str, required=False,
                             default=None, choices=enum_values(YesNo),
                             help="%(default)s")
    args_parser.add_argument("--npuw_llm_enable_block_based_kv_cache", type=str, required=False,
                             default=None, choices=enum_values(YesNo),
                             help="%(default)s")
    args_parser.add_argument("--npuw_llm_generate_pyramid", type=str, required=False,
                             default=enum_value(YesNo.YES), choices=enum_values(YesNo), help="%(default)s")
    args_parser.add_argument("--npu_turbo", type=str, required=False,
                             default=enum_value(YesNo.YES), choices=enum_values(YesNo), help="%(default)s")
    args_parser.add_argument("--npuw_devices", type=str, required=False,
                             default=None, help="%(default)s")
    args_parser.add_argument("--gpu_enable_large_allocations", type=str, required=False,
                             default=enum_value(YesNo.YES), choices=enum_values(YesNo), help="%(default)s")
    args_parser.add_argument("--gpu_priorities", type=str, required=False,
                             default=enum_value(Level.HIGH), choices=enum_values(Level), help="%(default)s")
    args_parser.add_argument("--prompt_lookup", type=str, required=False,
                             default=enum_value(Turn.off), choices=enum_values(Turn), help="%(default)s")
    args_parser.add_argument("--max_media_mb", type=int, required=False, default=20,
                             help="%(default)s, size limit of a single image or audio attachment")
    args_parser.add_argument("--max_media_items", type=int, required=False, default=16,
                             help="%(default)s, media attachments allowed in one request")
    args_parser.add_argument("--allow_local_media_files", type=str, required=False,
                             default=enum_value(Turn.off), choices=enum_values(Turn),
                             help="read media referenced by a local file path, not only base64 and http(s)")
    args_parser.add_argument("--eagle3_mode", type=str, required=False,
                             default=None, choices=enum_values(Turn), help="%(default)s")
    args_parser.add_argument("--mtp_mode", type=str, required=False,
                             default=None, choices=enum_values(Turn), help="%(default)s")
    args_parser.add_argument("--dflash_mode", type=str, required=False,
                             default=None, choices=enum_values(Turn), help="%(default)s")
    args_parser.add_argument("--interrupt_completions", type=str, required=False,
                             default=enum_value(Turn.on), choices=enum_values(Turn),
                             help="interrupt the running /v1/completions request when a new one arrives, "
                                  "the new request is served first")
