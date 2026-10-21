"""Tests of the subprocess-free platform detection."""
import os
import platform
import subprocess

from agent.common.platform_info import system


def test_system_matches_platform_without_subprocess(monkeypatch):
    """The helper reports the OS without starting any process."""
    def _fail(*args, **kwargs):
        raise AssertionError("system() must not spawn a subprocess")

    monkeypatch.setattr(subprocess, "run", _fail)
    monkeypatch.setattr(subprocess, "check_output", _fail)
    expected = "Windows" if os.name == "nt" else platform.uname().system
    assert system() == expected
