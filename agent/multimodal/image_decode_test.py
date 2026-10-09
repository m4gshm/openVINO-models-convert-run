import importlib.util
import io
import struct
import unittest
import zlib

import numpy as np

from agent.multimodal.image_decode import PNG_SIGNATURE, decode_image, decode_png
from agent.multimodal.media_error import MediaDecodeError

PILLOW_AVAILABLE = importlib.util.find_spec("PIL") is not None

# Minimal PNG encoder: builds decoder test inputs without external dependencies.

COLOR_TYPE_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + tag + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))


def _paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)


def _filter_row(row: list[int], prev: list[int], filter_type: int, bpp: int) -> list[int]:
    """Apply a PNG scanline filter to one row of sample bytes (encoding uses raw neighbors)."""
    size = len(row)
    if filter_type == 0:
        return list(row)
    if filter_type == 1:
        return [(row[i] - (row[i - bpp] if i >= bpp else 0)) & 0xFF for i in range(size)]
    if filter_type == 2:
        return [(row[i] - prev[i]) & 0xFF for i in range(size)]
    if filter_type == 3:
        return [(row[i] - (((row[i - bpp] if i >= bpp else 0) + prev[i]) >> 1)) & 0xFF for i in range(size)]
    return [(row[i] - _paeth_predictor(row[i - bpp] if i >= bpp else 0, prev[i],
                                       prev[i - bpp] if i >= bpp else 0)) & 0xFF for i in range(size)]


def _row_bytes(pixels: np.ndarray, bit_depth: int) -> list[list[int]]:
    """Flatten image rows into per-sample byte lists (big endian for 16 bit)."""
    height = pixels.shape[0]
    if bit_depth == 16:
        flat = pixels.astype(">u2").view(np.uint8).reshape(height, -1)
    else:
        flat = pixels.astype(np.uint8).reshape(height, -1)
    return [[int(value) for value in row] for row in flat]


def new_png(pixels: np.ndarray, color_type: int = 2, bit_depth: int = 8,
            filter_types: list[int] | None = None, palette: np.ndarray | None = None) -> bytes:
    """Encode an array into PNG bytes with the given color type, bit depth and row filters."""
    height, width = pixels.shape[0], pixels.shape[1]
    bpp = COLOR_TYPE_CHANNELS[color_type] * (bit_depth // 8)
    rows = _row_bytes(pixels, bit_depth)
    filters = filter_types if filter_types is not None else [0] * height
    if len(filters) < height:
        raise ValueError("filter_types must cover every row")
    raw = b""
    previous = [0] * len(rows[0])
    for y in range(height):
        filtered = _filter_row(rows[y], previous, filters[y], bpp)
        raw += bytes([filters[y]]) + bytes(filtered)
        previous = rows[y]
    ihdr = struct.pack(">II", width, height) + bytes([bit_depth, color_type, 0, 0, 0])
    return (PNG_SIGNATURE + _chunk(b"IHDR", ihdr)
            + (_chunk(b"PLTE", palette.tobytes()) if palette is not None else b"")
            + _chunk(b"IDAT", zlib.compress(raw, 1))
            + _chunk(b"IEND", b""))


def rgb_pixels(height: int, width: int) -> np.ndarray:
    """Deterministic RGB test pattern."""
    return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)


class DecodePngCase(unittest.TestCase):
    def test_decode_rgb(self):
        expected = rgb_pixels(4, 5)
        decoded = decode_png(new_png(expected, color_type=2))
        self.assertEqual((4, 5, 3), decoded.shape)
        self.assertTrue(np.array_equal(expected, decoded))

    def test_decode_rows_with_every_filter_type(self):
        expected = rgb_pixels(4, 6)
        decoded = decode_png(new_png(expected, color_type=2, filter_types=[0, 1, 2, 4]))
        self.assertTrue(np.array_equal(expected, decoded))

    def test_decode_average_filtered_rows(self):
        expected = rgb_pixels(3, 4)
        decoded = decode_png(new_png(expected, color_type=2, filter_types=[3, 3, 1]))
        self.assertTrue(np.array_equal(expected, decoded))

    def test_decode_grayscale(self):
        gray = np.arange(12, dtype=np.uint8).reshape(3, 4, 1)
        decoded = decode_png(new_png(gray, color_type=0))
        self.assertEqual((3, 4, 3), decoded.shape)
        self.assertTrue(np.array_equal(np.repeat(gray, 3, axis=2), decoded))

    def test_decode_rgba_drops_alpha(self):
        rgb = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        rgba = np.concatenate([rgb, np.full((2, 4, 1), 128, dtype=np.uint8)], axis=2)
        self.assertTrue(np.array_equal(rgb, decode_png(new_png(rgba, color_type=6))))

    def test_decode_gray_alpha(self):
        gray = np.arange(6, dtype=np.uint8).reshape(2, 3, 1)
        ga = np.concatenate([gray, np.full((2, 3, 1), 255, dtype=np.uint8)], axis=2)
        self.assertTrue(np.array_equal(np.repeat(gray, 3, axis=2), decode_png(new_png(ga, color_type=4))))

    def test_decode_palette(self):
        palette = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
        indexes = np.array([[[0], [2]], [[1], [0]]], dtype=np.uint8)
        decoded = decode_png(new_png(indexes, color_type=3, palette=palette))
        self.assertTrue(np.array_equal(palette[indexes[:, :, 0]], decoded))

    def test_decode_16_bit(self):
        rgb16 = np.array([[[1000, 2000, 3000], [4000, 5000, 6000]]], dtype=np.uint16)
        decoded = decode_png(new_png(rgb16, color_type=2, bit_depth=16))
        self.assertEqual((1, 2, 3), decoded.shape)
        self.assertEqual([3, 7, 11], [int(value) for value in decoded[0, 0]])

    def test_reject_interlaced(self):
        png = bytearray(new_png(rgb_pixels(1, 2), color_type=2))
        png[28] = 1  # interlace byte of the IHDR payload
        with self.assertRaises(Exception):
            decode_png(bytes(png))

    def test_reject_not_png(self):
        with self.assertRaises(Exception):
            decode_png(b"not a png at all")


class DecodeImageCase(unittest.TestCase):
    def test_tensor_layout_is_batch_height_width_rgb(self):
        expected = rgb_pixels(2, 3)
        tensor = decode_image(new_png(expected, color_type=2), "png")
        self.assertEqual((1, 2, 3, 3), tuple(tensor.shape))
        self.assertEqual("u8", tensor.get_element_type().to_string())
        self.assertTrue(np.array_equal(expected[None, ...], np.asarray(tensor.data)))

    def test_detects_png_when_format_is_not_given(self):
        tensor = decode_image(new_png(rgb_pixels(1, 2), color_type=2))
        self.assertEqual((1, 1, 2, 3), tuple(tensor.shape))


@unittest.skipUnless(PILLOW_AVAILABLE, "pillow is not installed")
class PillowDecodeCase(unittest.TestCase):
    """JPEG assertions compare regions, not pixels: lossy coding shifts exact values."""

    @staticmethod
    def jpeg_of(pixels: np.ndarray, **save_options) -> bytes:
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="JPEG", quality=95, subsampling=0, **save_options)
        return buffer.getvalue()

    def test_jpeg_is_decoded(self):
        source = np.zeros((8, 16, 3), dtype=np.uint8)
        source[:, :8] = (255, 0, 0)
        source[:, 8:] = (0, 255, 0)

        tensor = decode_image(self.jpeg_of(source), "jpeg")
        decoded = np.asarray(tensor.data)[0].astype(np.int16)

        self.assertEqual((1, 8, 16, 3), tuple(tensor.shape))
        self.assertEqual((8, 16, 3), decoded.shape)
        self.assertGreater(int(decoded[:, :8, 0].mean()), int(decoded[:, 8:, 0].mean()))
        self.assertGreater(int(decoded[:, 8:, 1].mean()), int(decoded[:, :8, 1].mean()))

    def test_exif_orientation_is_applied(self):
        from PIL import Image

        source = np.zeros((16, 32, 3), dtype=np.uint8)
        source[:, :16] = (255, 0, 0)
        source[:, 16:] = (0, 255, 0)
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90 degrees clockwise

        decoded = np.asarray(decode_image(self.jpeg_of(source, exif=exif.tobytes()), "jpeg").data)[0]

        # dimensions swapped: the left (red) half becomes the top rows after a clockwise rotation
        self.assertEqual((32, 16, 3), decoded.shape)
        self.assertGreater(int(decoded[:16, :, 0].mean()), int(decoded[16:, :, 0].mean()))
        self.assertGreater(int(decoded[16:, :, 1].mean()), int(decoded[:16, :, 1].mean()))

    def test_png_still_uses_the_builtin_decoder(self):
        expected = rgb_pixels(3, 4)
        self.assertTrue(np.array_equal(expected, decode_png(new_png(expected, color_type=2))))


@unittest.skipIf(PILLOW_AVAILABLE, "pillow is installed")
class PillowMissingCase(unittest.TestCase):
    def test_error_names_the_package_to_install(self):
        with self.assertRaises(MediaDecodeError) as context:
            decode_image(b"\xff\xd8\xff\xdb" + b"\x00" * 64, "jpeg")
        self.assertIn("pillow", str(context.exception))


if __name__ == '__main__':
    unittest.main()
