import struct
import unittest

import numpy as np

from agent.multimodal.audio_decode import (TARGET_SAMPLE_RATE, WAVE_FORMAT_IEEE_FLOAT, WAVE_FORMAT_PCM,
                                           decode_audio, decode_wav, resample, to_mono)
from agent.multimodal.media_error import MediaDecodeError

PCM_16 = 16
PCM_24 = 24
PCM_32 = 32
PCM_8 = 8


def new_wav(pcm: bytes, sample_rate: int, channels: int, bits: int, audio_format: int = WAVE_FORMAT_PCM) -> bytes:
    """Build WAV container bytes for the given payload."""
    block_align = channels * bits // 8
    byte_rate = sample_rate * block_align
    fmt = struct.pack("<HHIIHH", audio_format, channels, sample_rate, byte_rate, block_align, bits)
    padded = pcm + (b"\x00" if len(pcm) % 2 else b"")
    riff_size = 4 + 8 + len(fmt) + 8 + len(padded)
    return (b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
            + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"data" + struct.pack("<I", len(pcm)) + padded)


def pcm16(values: np.ndarray) -> bytes:
    """Encode float samples into signed 16 bit little endian bytes."""
    return np.round(np.clip(values, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def pcm24(values: np.ndarray) -> bytes:
    """Encode float samples into 24 bit little endian bytes."""
    scaled = np.round(np.clip(values, -1.0, 1.0) * 8388607)
    scaled = np.where(scaled < 0, scaled + 0x1000000, scaled).astype(np.int64)
    return b"".join(bytes([int(v) & 0xFF, (int(v) >> 8) & 0xFF, (int(v) >> 16) & 0xFF]) for v in scaled)


def sine(samples: int, cycles: float = 4) -> np.ndarray:
    """One period-complete sine wave in the [-1, 1] range."""
    return np.sin(np.linspace(0, cycles * 2 * np.pi, samples, endpoint=False)).astype(np.float32)


class DecodeWavCase(unittest.TestCase):
    def test_pcm16_mono(self):
        expected = sine(64)
        samples, sample_rate = decode_wav(new_wav(pcm16(expected), 8000, 1, PCM_16))
        self.assertEqual(8000, sample_rate)
        self.assertEqual(64, samples.size)
        self.assertTrue(np.allclose(expected, samples, atol=2 / 32767))

    def test_pcm8_mono_is_unsigned(self):
        encoded = np.round((np.array([0.0, 1.0, -1.0], dtype=np.float32) * 127) + 128).astype(np.uint8)
        samples, _ = decode_wav(new_wav(encoded.tobytes(), 16000, 1, PCM_8))
        self.assertTrue(np.allclose([0.0, 0.992, -0.992], samples, atol=0.01))

    def test_pcm24_mono(self):
        expected = sine(32)
        samples, _ = decode_wav(new_wav(pcm24(expected), 16000, 1, PCM_24))
        self.assertTrue(np.allclose(expected, samples, atol=1e-5))

    def test_pcm32_mono(self):
        expected = sine(16)
        payload = np.round(expected.astype(np.float64) * 2147483647).astype("<i4").tobytes()
        samples, _ = decode_wav(new_wav(payload, 16000, 1, PCM_32))
        self.assertTrue(np.allclose(expected, samples, atol=1e-6))

    def test_ieee_float32(self):
        expected = sine(16)
        payload = expected.astype("<f4").tobytes()
        samples, _ = decode_wav(new_wav(payload, 16000, 1, 32, WAVE_FORMAT_IEEE_FLOAT))
        self.assertTrue(np.allclose(expected, samples, atol=1e-6))

    def test_stereo_downmixes_to_mono(self):
        left = np.ones(8, dtype=np.float32)
        right = -np.ones(8, dtype=np.float32)
        interleaved = np.stack([left, right], axis=1).reshape(-1)
        samples, _ = decode_wav(new_wav(pcm16(interleaved), 16000, 2, PCM_16))
        self.assertEqual(8, samples.size)
        self.assertTrue(np.allclose(np.zeros(8), samples, atol=1e-3))

    def test_truncated_data_chunk_is_ignored(self):
        wav = new_wav(pcm16(sine(8)), 16000, 1, PCM_16)
        samples, _ = decode_wav(wav + b"\x00" * 3)
        self.assertEqual(8, samples.size)

    def test_rejects_non_wav(self):
        with self.assertRaises(MediaDecodeError):
            decode_wav(b"NOT RIFF DATA")

    def test_rejects_unsupported_sample_width(self):
        wav = new_wav(b"\x00" * 16, 16000, 1, 12)
        with self.assertRaises(MediaDecodeError):
            decode_wav(wav)


class HelpersCase(unittest.TestCase):
    def test_to_mono_averages_channels(self):
        samples = np.array([1.0, -1.0, 0.5, 0.5], dtype=np.float32)
        self.assertTrue(np.allclose([0.0, 0.5], to_mono(samples, 2)))

    def test_resample_doubles_rate(self):
        samples = np.array([0.0, 1.0], dtype=np.float32)
        resampled = resample(samples, 8000, 16000)
        self.assertEqual(4, resampled.size)

    def test_resample_keeps_target_rate(self):
        samples = sine(32)
        self.assertTrue(np.array_equal(samples, resample(samples, TARGET_SAMPLE_RATE)))


class DecodeAudioTensorCase(unittest.TestCase):
    def test_tensor_is_float32_at_target_rate(self):
        expected = sine(100)
        tensor = decode_audio(new_wav(pcm16(expected), 8000, 1, PCM_16), "wav")
        shape = tuple(tensor.shape)
        self.assertEqual(1, len(shape))
        self.assertEqual("f32", tensor.get_element_type().to_string())
        self.assertEqual(200, shape[0])
        self.assertLessEqual(float(np.max(np.abs(np.asarray(tensor.data)))), 1.0)

    def test_undecodable_payload_reports_supported_formats(self):
        with self.assertRaises(MediaDecodeError) as context:
            decode_audio(b"this is not an audio stream at all", None)
        self.assertIn("WAV", str(context.exception))

    def test_overflowing_samples_are_normalized(self):
        payload = np.array([2.0, -4.0, 1.0], dtype=np.float32).astype("<f4").tobytes()
        tensor = decode_audio(new_wav(payload, 16000, 1, 32, WAVE_FORMAT_IEEE_FLOAT), "wav")
        samples = np.asarray(tensor.data)
        self.assertAlmostEqual(1.0, float(np.max(np.abs(samples))), places=5)
        self.assertAlmostEqual(-1.0, float(samples.min()), places=5)

    def test_unknown_media_format_reports_hint(self):
        with self.assertRaises(MediaDecodeError):
            decode_audio(b"\xff\xfb\x90\x00" + b"\x00" * 64, "mp3")


if __name__ == '__main__':
    unittest.main()
