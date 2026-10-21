"""Platform identification without spawning a subprocess.

``platform.system()`` runs ``cmd /c ver`` on Windows, so every call starts a
new process. The OpenVINO telemetry checker calls it while ``openvino`` is
imported, and that process spawn intermittently kills the interpreter with
``Windows fatal exception: access violation`` during test collection.

The helpers here answer the same question from ``os.name``/``sys.platform``
only, so they never touch the process table.
"""
import os
import platform


def system() -> str:
    """Return the OS name (``Windows``, ``Darwin``, ``Linux``, ...) without a subprocess.

    Windows is detected from ``os.name``; on POSIX ``platform.uname()`` is used,
    which reads the values the kernel exports instead of starting a process.
    """
    if os.name == "nt":
        return "Windows"
    return platform.uname().system
