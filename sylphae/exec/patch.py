"""Apply a structured op to a file inside a tree.

Implements ``apply_to_tree`` / ``ApplyResult`` from INTERFACES.md section 5.

Two properties matter and both are tested:

* **``dry_run=True`` writes nothing.** Not "writes and rolls back" — nothing is created,
  truncated or replaced. The caller gets the candidate source and decides.
* **``dry_run=False`` is atomic.** The new source is written to a temporary file in the
  *same directory* and then ``os.replace``-d over the target, so a crash or a full disk
  leaves either the old file or the new file, never a half-written one.

``sylphae.ops`` is imported lazily, inside the function. Another workstream owns that module
and it may not exist yet; this module must import cleanly regardless, and a missing
``sylphae.ops`` is reported as ``ok=False`` with a reason rather than an ``ImportError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sylphae.exec.textio import read_source, write_source

__all__ = ["ApplyResult", "apply_to_tree"]

# The byte-faithful I/O contract lives in ``sylphae.exec.textio`` and is shared with
# ``sylphae.coprocessor.executor``, which edits the same kind of file. It used to be defined
# here as well, which is how one of the two writers ended up without it.


@dataclass
class ApplyResult:
    """Outcome of one :func:`apply_to_tree` call.

    ``ok=False`` always carries a human-readable ``error`` and ``new_source=None``.
    ``ok=True`` always carries the full resulting source, whether or not it was written.
    """

    ok: bool
    new_source: str | None
    error: str | None


def _fail(message: str) -> ApplyResult:
    return ApplyResult(ok=False, new_source=None, error=message)


def _resolve_inside(root: Path, target_path: str) -> Path | None:
    """Resolve ``target_path`` under ``root``, or ``None`` if it escapes the tree.

    A patch spec that writes outside the repository it was generated for is never the
    caller's intent, so it is refused rather than clamped.
    """
    root = Path(root).resolve()
    raw = Path(target_path)
    candidate = raw if raw.is_absolute() else root / raw
    try:
        resolved = candidate.resolve()
    except OSError as exc:  # pragma: no cover - platform dependent
        return None
    if resolved != root and root not in resolved.parents:
        return None
    return resolved


def _load_apply_op() -> Any:
    """Import ``sylphae.ops.apply_op`` lazily. Raises ImportError with a useful message."""
    from sylphae.ops.apply import apply_op  # noqa: PLC0415 - deliberately lazy

    return apply_op


def apply_to_tree(
    root: Path, target_path: str, spec: Any, dry_run: bool = True
) -> ApplyResult:
    """Apply ``spec`` to ``root/target_path``.

    Parameters
    ----------
    root:
        Repository root. ``target_path`` is interpreted relative to it.
    target_path:
        Repo-relative posix path of the file to edit. Paths that resolve outside ``root``
        are refused.
    spec:
        An ``sylphae.ops.spec.OpSpec``. Passed through to ``sylphae.ops.apply.apply_op``
        untouched.
    dry_run:
        When true (the default) nothing is written to disk and the caller receives the
        prospective source in ``ApplyResult.new_source``.
    """
    if spec is None:
        return _fail("spec is None")

    target = _resolve_inside(root, target_path)
    if target is None:
        return _fail(f"target path escapes the tree root: {target_path!r}")
    if not target.is_file():
        return _fail(f"target file does not exist: {target_path!r}")

    try:
        old_source = read_source(target)
    except OSError as exc:
        return _fail(f"cannot read {target_path!r}: {exc}")

    try:
        apply_op = _load_apply_op()
    except ImportError as exc:
        return _fail(f"sylphae.ops is unavailable, cannot apply: {exc}")

    try:
        new_source = apply_op(old_source, spec)
    except Exception as exc:  # noqa: BLE001 - OpApplyError plus anything ops raises
        return _fail(f"{type(exc).__name__}: {exc}")

    if not isinstance(new_source, str):
        return _fail(
            f"apply_op returned {type(new_source).__name__}, expected str"
        )

    if dry_run:
        return ApplyResult(ok=True, new_source=new_source, error=None)

    try:
        write_source(target, new_source)
    except OSError as exc:
        return _fail(f"cannot write {target_path!r}: {exc}")
    return ApplyResult(ok=True, new_source=new_source, error=None)
