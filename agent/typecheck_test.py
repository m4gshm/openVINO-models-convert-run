"""Tests of the startup mypy gate: pass, abort and every skip path."""
import subprocess
import unittest
from unittest.mock import patch

from agent.typecheck import typecheck


def completed(returncode: int) -> subprocess.CompletedProcess:
    """Build a fake subprocess result with the given exit code."""
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="mypy output", stderr="")


@patch("agent.typecheck.subprocess.run", return_value=completed(0))
class TypecheckCase(unittest.TestCase):
    """Runs typecheck() with a patched subprocess.run."""
    def test_clean_run_passes(self, run):
        """A clean mypy run neither aborts nor forces check mode."""
        typecheck()
        self.assertEqual(1, run.call_count)
        self.assertFalse(run.call_args.kwargs["check"])
        self.assertTrue(run.call_args.kwargs["capture_output"])

    def test_type_errors_abort_the_process(self, run):
        """A failing mypy run raises SystemExit."""
        run.return_value = completed(1)
        with self.assertRaises(SystemExit):
            typecheck()

    def test_missing_mypy_is_skipped(self, run):
        """Absent mypy binary is skipped instead of crashing."""
        run.side_effect = FileNotFoundError("mypy")
        typecheck()
        self.assertEqual(1, run.call_count)

    def test_timeout_is_skipped(self, run):
        """A hung mypy run is skipped after the timeout."""
        run.side_effect = subprocess.TimeoutExpired(cmd="mypy", timeout=1)
        typecheck()
        self.assertEqual(1, run.call_count)

    def test_frozen_executable_does_not_spawn_mypy(self, run):
        """Frozen builds never spawn mypy."""
        with patch("agent.typecheck.sys.frozen", True, create=True):
            typecheck()
        self.assertEqual(0, run.call_count)


if __name__ == '__main__':
    unittest.main()
