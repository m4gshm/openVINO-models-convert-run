"""Pytest bootstrap of the agent test suite.

Importing ``openvino`` (directly or through ``openvino_genai``) initializes the
OpenVINO telemetry, which asks ``platform.system()`` for the OS name. On
Windows that call spawns ``cmd /c ver``, and the process spawn intermittently
kills the interpreter with ``Windows fatal exception: access violation`` in the
middle of the test collection.

``platform.system()`` is therefore replaced for the whole test session by the
subprocess-free variant of ``agent.common.platform_info``. It returns the very
same value, it just does not start a process to find it out.
"""
import platform

from agent.common.platform_info import system as _system_without_subprocess

platform.system = _system_without_subprocess
