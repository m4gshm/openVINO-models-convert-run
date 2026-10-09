"""Decoding image media into OpenVINO GenAI image tensors.

OpenVINO GenAI expects images as ``uint8`` tensors in ``[batch, height, width, channels]``
layout with RGB channels. PNG is decoded with the built-in decoder (``zlib`` + ``numpy`` only),
so the server works without extra packages; every other format (JPEG, WEBP, GIF, BMP, ...) goes
through Pillow, which also applies EXIF orientation.
"""

import io
import logging
import zlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
import openvino as ov

from agent.multimodal.media_error import MediaDecodeError

log = logging.getLogger(__name__)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
RGB_CHANNELS = 3

# PNG color types: value -> channels
PNG_COLOR_TYPES: dict[int, int] = {
    0: 1,  # grayscale
    2: 3,  # RGB
    3: 1,  # palette indexes
    4: 2,  # grayscale + alpha
    6: 4,  # RGBA
}

_PILLOW_HINT = ("cannot decode this image format: the built-in decoder handles PNG only, "
                "install pillow for JPEG/WEBP/GIF/BMP ('uv add pillow' or 'pip install pillow') "
                "or send the image as PNG")


@dataclass(frozen=True)
class PngHeader:
    """Fields of a PNG IHDR chunk."""

    width: int
    height: int
    bit_depth: int
    color_type: int
    channels: int


def _chunks_of(data: bytes) -> list[tuple[bytes, bytes]]:
    """Split a PNG stream into (type, payload) chunks."""
    if not data.startswith(PNG_SIGNATURE):
        raise MediaDecodeError("not a PNG stream: bad signature")
    chunks: list[tuple[bytes, bytes]] = []
    position = len(PNG_SIGNATURE)
    size = len(data)
    while position + 8 <= size:
        length = int.from_bytes(data[position:position + 4], "big")
        chunk_type = data[position + 4:position + 8]
        payload_start = position + 8
        payload_end = payload_start + length
        if payload_end > size:
            raise MediaDecodeError("not a PNG stream: truncated chunk")
        chunks.append((chunk_type, data[payload_start:payload_end]))
        position = payload_end + 4  # skip CRC
    return chunks


def _payload_of(chunks: list[tuple[bytes, bytes]], chunk_type: bytes) -> bytes | None:
    """Payload of the first chunk of the given type."""
    return next((payload for name, payload in chunks if name == chunk_type), None)


def _read_ihdr(ihdr: bytes) -> PngHeader:
    """Parse and validate the IHDR payload."""
    if len(ihdr) < 13:
        raise MediaDecodeError(f"PNG IHDR chunk is too short: {len(ihdr)} bytes")
    width = int.from_bytes(ihdr[0:4], "big")
    height = int.from_bytes(ihdr[4:8], "big")
    bit_depth, color_type, compression, filter_method, interlace = ihdr[8], ihdr[9], ihdr[10], ihdr[11], ihdr[12]
    if width <= 0 or height <= 0:
        raise MediaDecodeError(f"PNG has invalid size {width}x{height}")
    if compression != 0 or filter_method != 0:
        raise MediaDecodeError("unsupported PNG compression or filter method")
    if interlace != 0:
        raise MediaDecodeError("interlaced PNG is not supported")
    if bit_depth not in (8, 16):
        raise MediaDecodeError(f"unsupported PNG bit depth {bit_depth}, expected 8 or 16")
    channels = PNG_COLOR_TYPES.get(color_type, 0)
    if not channels:
        raise MediaDecodeError(f"unsupported PNG color type {color_type}")
    return PngHeader(width, height, bit_depth, color_type, channels)


def _paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _unfilter_row(line: list[int], prev: list[int], filter_type: int, bpp: int) -> list[int]:
    """Reconstruct one scanline according to its PNG filter type."""
    size = len(line)
    if filter_type == 0:
        return line
    if filter_type == 1:  # Sub: left sample of the reconstructed row
        current = list(line)
        for i in range(bpp, size):
            current[i] = (current[i] + current[i - bpp]) & 0xFF
        return current
    if filter_type == 2:  # Up: sample of the reconstructed row above
        return [(line[i] + prev[i]) & 0xFF for i in range(size)]
    if filter_type == 3:  # Average of left and above
        current = list(line)
        for i in range(size):
            left = current[i - bpp] if i >= bpp else 0
            current[i] = (current[i] + ((left + prev[i]) >> 1)) & 0xFF
        return current
    if filter_type == 4:  # Paeth predictor
        current = list(line)
        for i in range(size):
            left = current[i - bpp] if i >= bpp else 0
            up_left = prev[i - bpp] if i >= bpp else 0
            current[i] = (current[i] + _paeth_predictor(left, prev[i], up_left)) & 0xFF
        return current
    raise MediaDecodeError(f"unknown PNG filter type {filter_type}")


def _scanlines(raw: bytes, height: int, row_bytes: int, bpp: int) -> np.ndarray:
    """Unfilter every PNG scanline into a flat uint8 array."""
    previous = [0] * row_bytes
    rows: list[int] = []
    for y in range(height):
        offset = y * (row_bytes + 1)
        line = [int(value) for value in raw[offset + 1: offset + 1 + row_bytes]]
        current = _unfilter_row(line, previous, raw[offset], bpp)
        rows.extend(current)
        previous = current
    return np.asarray(rows, dtype=np.uint8)


def _to_rgb(samples: np.ndarray, color_type: int, palette: np.ndarray | None) -> np.ndarray:
    """Convert per-channel samples into RGB uint8 pixels."""
    if color_type == 2:
        return samples
    if color_type == 0:
        return np.repeat(samples, RGB_CHANNELS, axis=2)
    if color_type == 4:
        return np.repeat(samples[:, :, 0:1], RGB_CHANNELS, axis=2)
    if color_type == 6:
        return samples[:, :, 0:RGB_CHANNELS]
    if color_type == 3:
        if palette is None:
            raise MediaDecodeError("palette PNG has no PLTE chunk")
        indexes = samples[:, :, 0].astype(np.int64)
        return palette[indexes][:, :, 0:RGB_CHANNELS]
    raise MediaDecodeError(f"unsupported PNG color type {color_type}")


def decode_png(data: bytes) -> np.ndarray:
    """Decode a non-interlaced 8/16-bit PNG into an ``[H, W, 3]`` uint8 RGB array."""
    chunks = _chunks_of(data)
    ihdr = _payload_of(chunks, b"IHDR")
    if ihdr is None:
        raise MediaDecodeError("PNG has no IHDR chunk")
    header = _read_ihdr(ihdr)
    palette_payload = _payload_of(chunks, b"PLTE")
    palette = (np.frombuffer(palette_payload, dtype=np.uint8).reshape(-1, 3)
               if palette_payload is not None else None)

    raw = zlib.decompress(b"".join(payload for name, payload in chunks if name == b"IDAT"))
    bytes_per_sample = header.bit_depth // 8
    row_bytes = header.width * header.channels * bytes_per_sample
    expected = row_bytes * header.height
    if len(raw) < expected + header.height:
        raise MediaDecodeError(f"PNG data is truncated: {len(raw)} < {expected + header.height} bytes")

    bpp = header.channels * bytes_per_sample
    samples = _scanlines(raw, header.height, row_bytes, bpp)
    shape = (header.height, header.width, header.channels)
    if header.bit_depth == 16:
        samples = samples.reshape(shape + (2,))[:, :, :, 0]
    else:
        samples = samples.reshape(shape)
    return _to_rgb(samples, header.color_type, palette)


def _decode_with_pillow(data: bytes) -> np.ndarray:
    """Decode any Pillow supported image into RGB pixels, honouring EXIF orientation."""
    try:
        from PIL import Image, ImageOps  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise MediaDecodeError(_PILLOW_HINT) from e
    try:
        with Image.open(io.BytesIO(data)) as image:
            oriented = ImageOps.exif_transpose(image)
            return np.asarray(oriented.convert("RGB"))
    except Exception as e:  # pylint: disable=broad-exception-caught
        raise MediaDecodeError(f"cannot decode image: {e}") from e


def decode_image(data: bytes, media_format: str | None = None) -> ov.Tensor:
    """Decode image bytes into an ``ov.Tensor`` of shape ``[1, H, W, 3]`` (uint8 RGB)."""
    decoder: Callable[[bytes], np.ndarray] = decode_png if _is_png(data, media_format) \
        else _decode_with_pillow
    pixels = np.ascontiguousarray(decoder(data))
    if pixels.ndim != 3 or pixels.shape[2] != RGB_CHANNELS:
        raise MediaDecodeError(f"decoded image must have [H, W, 3] shape, got {pixels.shape}")
    image = pixels.astype(np.uint8)[None, ...]
    log.debug("image tensor decoded, shape=%s, format=%s", tuple(image.shape), media_format or "auto")
    return ov.Tensor(image)


def _is_png(data: bytes, media_format: str | None) -> bool:
    """Whether the payload should go to the built-in PNG decoder."""
    return media_format == "png" or (media_format is None and data.startswith(PNG_SIGNATURE))
