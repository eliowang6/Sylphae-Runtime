"""PEP 562 lazy loading for the ops environment package.

The goal requires the light CLI paths to stay light (§3 rule 9), and this
package's playbooks pull in ``subprocess``-heavy helpers that a probe command
never needs.  Importing ``sylphae.ops_env`` costs nothing; importing
``sylphae.ops_env.playbooks`` is what pays.
"""

from __future__ import annotations

from typing import Any

__all__ = ["safety", "executor", "playbooks", "trash", "fingerprint"]


def __getattr__(name: str) -> Any:
    if name in ("safety", "executor", "playbooks", "trash", "fingerprint"):
        import importlib

        return importlib.import_module(f"sylphae.ops_env.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
