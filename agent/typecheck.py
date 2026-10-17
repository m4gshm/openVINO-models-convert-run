"""Startup type check: runs mypy against the agent package before the model loads.

Deliberately forgiving about its own absence: a production dist (PyInstaller) has no
mypy inside, and the check must never become the reason a server cannot start.
"""

import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

TYPECHECK_TIMEOUT_SECONDS = 120


def typecheck(timeout_seconds: int = TYPECHECK_TIMEOUT_SECONDS) -> None:
    """Run mypy on the agent package and abort the process when it reports errors.

    Skips silently (with a log line) when: running from a frozen executable, mypy is
    not installed, or the project config is not reachable.
    """
    if getattr(sys, "frozen", False):
        log.info("frozen executable: startup type check skipped")
        return
    repo_root = Path(__file__).resolve().parent.parent
    if not (repo_root / "pyproject.toml").is_file():
        log.info("no pyproject.toml above %s: startup type check skipped", repo_root)
        return
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "mypy"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            cwd=repo_root,
        )
    except FileNotFoundError:
        log.warning("mypy is not installed: startup type check skipped (pip install mypy)")
        return
    except subprocess.TimeoutExpired:
        log.warning("startup type check timed out after %ss: skipped", timeout_seconds)
        return
    if completed.returncode != 0:
        print(completed.stdout, file=sys.stderr)
        raise SystemExit("startup type check failed: fix the errors above or run 'task typecheck'")
