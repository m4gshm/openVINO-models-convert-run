import subprocess
import unittest
from unittest.mock import patch

from agent.typecheck import typecheck


def completed(returncode: int) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="mypy output", stderr="")


@patch("agent.typecheck.subprocess.run", return_value=completed(0))
class TypecheckCase(unittest.TestCase):
    def test_clean_run_passes(self, run):
        typecheck()
        self.assertEqual(1, run.call_count)
        self.assertFalse(run.call_args.kwargs["check"])
        self.assertTrue(run.call_args.kwargs["capture_output"])

    def test_type_errors_abort_the_process(self, run):
        run.return_value = completed(1)
        with self.assertRaises(SystemExit):
            typecheck()

    def test_missing_mypy_is_skipped(self, run):
        run.side_effect = FileNotFoundError("mypy")
        typecheck()
        self.assertEqual(1, run.call_count)

    def test_timeout_is_skipped(self, run):
        run.side_effect = subprocess.TimeoutExpired(cmd="mypy", timeout=1)
        typecheck()
        self.assertEqual(1, run.call_count)

    def test_frozen_executable_does_not_spawn_mypy(self, run):
        with patch("agent.typecheck.sys.frozen", True, create=True):
            typecheck()
        self.assertEqual(0, run.call_count)


if __name__ == '__main__':
    unittest.main()
