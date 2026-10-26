"""Tests of the server launcher helpers."""

import unittest

from fim_bench.__main__ import DEFAULT_SERVER_ARGS, build_server_args, pick_free_port


class BuildServerArgsCase(unittest.TestCase):
    """Tests of merging the safe NPU defaults with explicit run args."""

    def test_defaults_are_appended(self):
        merged = build_server_args([])
        self.assertEqual(DEFAULT_SERVER_ARGS, merged)

    def test_explicit_flag_drops_the_default_pair(self):
        merged = build_server_args(["--npu_prefill_hint", "DYNAMIC"])
        self.assertNotIn("STATIC", merged)
        self.assertIn("DYNAMIC", merged)
        # the untouched defaults survive
        self.assertIn("--npu_turbo", merged)
        self.assertIn("YES", merged)

    def test_explicit_max_prompt_len(self):
        merged = build_server_args(["--max_prompt_len", "4096"])
        self.assertNotIn("8192", merged)
        self.assertEqual("4096", merged[merged.index("--max_prompt_len") + 1])

    def test_explicit_args_come_last_and_win(self):
        merged = build_server_args(["--max_prompt_len", "4096"])
        self.assertLess(merged.index("--max_prompt_len"), merged.index("4096"))
        self.assertEqual(merged[-2:], ["--max_prompt_len", "4096"])


class PickFreePortCase(unittest.TestCase):
    """Tests of the free port picker."""

    def test_returns_a_busable_port(self):
        import socket
        port = pick_free_port()
        self.assertTrue(1024 <= port <= 65535)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))  # must be bindable right away

    def test_two_calls_differ(self):
        self.assertNotEqual(pick_free_port(), pick_free_port())


if __name__ == "__main__":
    unittest.main()
