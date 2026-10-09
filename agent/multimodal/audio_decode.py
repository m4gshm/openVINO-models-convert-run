"""Decoding audio media into OpenVINO GenAI audio tensors.

OpenVINO GenAI expects audio as ``float32`` mono PCM sampled at 16 kHz and normalized to
the ``[-1, 1]`` range. WAV/RIFF is decoded with the built-in decoder (``numpy`` only);
other containers require the optional ``soundfile`` or ``librosa`` packages.
"""

import io
import logging

import numpy as np
import openvino as ov

from agent.multimodal.media_error import MediaDecodeError

log = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000

RIFF_SIGNATURE = b"RIFF"
WAVE_TAG = b"WAVE"

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE

_SUPPORT_HINT = ("supported out of the box: WAV (PCM 8/16/24/32 bit and IEEE float 32); "
                 "install 'soundfile' or 'librosa' for mp3/flac/ogg/m4a")


def _riff_chunks(data: bytes) -> dict[bytes, bytes]:
    """Collect RIFF chunks by id, keeping the first occurrence of each id."""
    if not data.startswith(RIFF_SIGNATURE) or data[8:12] != WAVE_TAG:
        raise MediaDecodeError("not a WAV stream: missing RIFF/WAVE header")
    chunks: dict[bytes, bytes] = {}
    position = 12
    size = len(data)
    while position + 8 <= size:
        chunk_id = data[position:position + 4]
        chunk_size = int.from_bytes(data[position + 4:position + 8], "little")
        payload_start = position + 8
        payload_end = payload_start + chunk_size
        if payload_end > size:
            raise MediaDecodeError("WAV stream is truncated")
        if chunk_id not in chunks:
            chunks[chunk_id] = data[payload_start:payload_end]
        position = payload_end + (chunk_size % 2)
    return chunks


def _read_format(fmt: bytes) -> tuple[int, int, int, int]:
    """Parse a WAV ``fmt `` chunk into (audio_format, channels, sample_rate, bits_per_sample)."""
    if len(fmt) < 16:
        raise MediaDecodeError(f"WAV format chunk is too short: {len(fmt)} bytes")
    audio_format = int.from_bytes(fmt[0:2], "little")
    channels = int.from_bytes(fmt[2:4], "little")
    sample_rate = int.from_bytes(fmt[4:8], "little")
    bits = int.from_bytes(fmt[14:16], "little")
    if audio_format == WAVE_FORMAT_EXTENSIBLE:
        if len(fmt) < 40:
            raise MediaDecodeError("WAVE_FORMAT_EXTENSIBLE chunk is too short")
        audio_format = int.from_bytes(fmt[24:26], "little")
    if channels <= 0 or sample_rate <= 0:
        raise MediaDecodeError(f"invalid WAV channels={channels}, sample_rate={sample_rate}")
    return audio_format, channels, sample_rate, bits


def _samples_to_float(raw: bytes, audio_format: int, bits: int) -> np.ndarray:
    """Convert little-endian sample bytes into float32 samples in the ``[-1, 1]`` range."""
    if audio_format == WAVE_FORMAT_IEEE_FLOAT and bits == 32:
        return np.frombuffer(raw, dtype="<f4").astype(np.float32)
    if audio_format != WAVE_FORMAT_PCM:
        raise MediaDecodeError(f"unsupported WAV audio format {audio_format} ({_SUPPORT_HINT})")
    if bits == 8:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if bits == 16:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if bits == 32:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    if bits == 24:
        padded = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (padded[:, 0].astype(np.int64) | (padded[:, 1].astype(np.int64) << 8)
                  | (padded[:, 2].astype(np.int64) << 16))
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return (values / 8388608.0).astype(np.float32)
    raise MediaDecodeError(f"unsupported WAV sample width {bits} bits ({_SUPPORT_HINT})")


def _normalize(samples: np.ndarray) -> np.ndarray:
    """Scale samples into the ``[-1, 1]`` range when they overflow it."""
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1.0:
        return (samples / peak).astype(np.float32)
    return samples.astype(np.float32)


def to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    """Mix multi channel samples down to one channel."""
    if channels == 1:
        return samples
    usable = (samples.size // channels) * channels
    return samples[:usable].reshape(-1, channels).mean(axis=1)


def resample(samples: np.ndarray, source_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Resample mono samples with linear interpolation."""
    if not samples.size or source_rate == target_rate:
        return samples.astype(np.float32)
    target_size = max(1, int(round(samples.size * target_rate / source_rate)))
    positions = np.linspace(0, samples.size - 1, target_size)
    return np.interp(positions, np.arange(samples.size), samples).astype(np.float32)


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes into (float32 mono samples, sample rate)."""
    chunks = _riff_chunks(data)
    fmt = chunks.get(b"fmt ")
    raw = chunks.get(b"data")
    if fmt is None or raw is None:
        raise MediaDecodeError("WAV stream has no 'fmt ' or 'data' chunk")
    audio_format, channels, sample_rate, bits = _read_format(fmt)
    block_align = max(1, channels * bits // 8)
    usable = (len(raw) // block_align) * block_align
    if usable == 0:
        raise MediaDecodeError("WAV stream has no samples")
    samples = _samples_to_float(raw[:usable], audio_format, bits)
    return to_mono(samples, channels), sample_rate


def _decode_with_optional_library(data: bytes) -> tuple[np.ndarray, int] | None:
    """Decode any container using soundfile or librosa when they are installed."""
    try:
        import soundfile as sf  # pylint: disable=import-outside-toplevel
        samples, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return samples.mean(axis=1), int(sample_rate)
    except ImportError:
        pass
    except Exception as e:  # pylint: disable=broad-exception-caught
        log.debug("soundfile cannot decode audio: %s", e)
    try:
        import librosa  # pylint: disable=import-outside-toplevel
        samples, sample_rate = librosa.load(io.BytesIO(data), sr=None, mono=True)
        return np.asarray(samples, dtype=np.float32), int(sample_rate)
    except ImportError:
        pass
    except Exception as e:  # pylint: disable=broad-exception-caught
        log.debug("librosa cannot decode audio: %s", e)
    return None


def decode_audio(data: bytes, media_format: str | None = None,
                 target_rate: int = TARGET_SAMPLE_RATE) -> ov.Tensor:
    """Decode audio bytes into a 1-D ``float32`` mono ``ov.Tensor`` sampled at ``target_rate``."""
    if data.startswith(RIFF_SIGNATURE) and data[8:12] == WAVE_TAG:
        samples, sample_rate = decode_wav(data)
    else:
        decoded = _decode_with_optional_library(data)
        if decoded is None:
            raise MediaDecodeError(
                f"cannot decode audio format '{media_format or 'unknown'}' ({_SUPPORT_HINT})")
        samples, sample_rate = decoded

    mono = _normalize(resample(np.asarray(samples, dtype=np.float32).reshape(-1), sample_rate, target_rate))
    if not mono.size:
        raise MediaDecodeError("decoded audio has no samples")
    log.debug("audio tensor decoded, samples=%d, source_rate=%d, target_rate=%d, format=%s",
              mono.size, sample_rate, target_rate, media_format or "auto")
    return ov.Tensor(np.ascontiguousarray(mono, dtype=np.float32))
