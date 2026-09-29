"""Environment fingerprint: what the machine was when the command ran.

The goal asks for reproducibility records to carry the Python version, the
package versions, and the platform -- because "it works on my machine" is the
only answer an ops layer can give if it cannot say which machine.

Cheap by design: this is attached to *every* audit record, so it cannot afford to
import torch or walk the filesystem.  It reads ``sys`` and one ``pip list``-free
source (``importlib.metadata``) and caches the result per process.
"""

from __future__ import annotations

import platform
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def fingerprint() -> dict[str, Any]:
    """The environment this process is running in."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "",
        "implementation": platform.python_implementation(),
        "os": platform.system(),
        "os_release": platform.release(),
    }


@lru_cache(maxsize=256)
def package_version(dist: str) -> str | None:
    """The installed version of ``dist``, or ``None`` if it is absent.

    ``None`` is the interesting value: it is how a playbook says "the dependency
    is missing" rather than "the command failed".
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - py<3.8
        return None
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def selected_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    """Versions for the tools the playbooks actually invoke.

    Only what is used, not everything installed: an audit record that lists 400
    packages is a log nobody reads.  ``None`` is recorded rather than omitted, so
    "absent" and "not checked" stay distinguishable.
    """
    return {name: package_version(name) for name in names}


#: The tools ``sylphae.ops_env.playbooks`` may invoke.  Listed here rather than
#: discovered at runtime so the fingerprint is stable across runs of the same
#: playbook, even on a machine where some are missing.
PLAYBOOK_TOOLS = (
    "pip",
    "pytest",
    "ruff",
    "black",
)


def playbook_fingerprint() -> dict[str, Any]:
    """Fingerprint plus the versions of the playbook tools."""
    out = dict(fingerprint())
    out["tools"] = selected_versions(PLAYBOOK_TOOLS)
    return out


def interpreter_path() -> Path:
    """The interpreter that would run a discovered service.

    ``sys.executable`` is the ground truth for "which python am I", and the venv
    playbook compares a discovered interpreter against this to decide whether a
    virtual environment is *this* environment or a different one.
    """
    return Path(sys.executable)
