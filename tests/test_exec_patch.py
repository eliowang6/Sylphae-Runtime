"""Tests for ``sylphae/exec/patch.py`` -- INTERFACES.md section 5.

``sylphae.ops`` is owned by another workstream and may not exist yet, so the mechanics of
``apply_to_tree`` are tested against a local stub ``apply_op`` injected through
``sylphae.exec.patch._load_apply_op``. A separate integration test runs against the real
``sylphae.ops`` when it is present.

Run from the repository root:

    python -m pytest tests/test_exec_patch.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sylphae.exec import patch as patch_mod  # noqa: E402
from sylphae.exec.patch import ApplyResult, apply_to_tree  # noqa: E402


class StubOpApplyError(Exception):
    """Stands in for ``sylphae.ops.spec.OpApplyError``."""


class StubSpec:
    """Minimal stand-in for ``sylphae.ops.spec.OpSpec``."""

    def __init__(self, kind: str, path: str, span: tuple[int, int], props: dict):
        self.kind = kind
        self.path = path
        self.span = span
        self.props = props


def _stub_apply_op(old_src: str, spec) -> str:
    """Replace the single line in ``spec.span`` with ``spec.props['text']``.

    Mimics a well-behaved applier: the replacement inherits the terminator of the line it
    replaces, so the untouched part of the file is reproduced byte for byte.
    """
    if spec.kind == "explode":
        raise StubOpApplyError("stub refuses this spec")
    lines = old_src.splitlines(keepends=True)
    start, end = spec.span
    if not (1 <= start <= end <= len(lines)):
        raise StubOpApplyError(f"span {spec.span} out of range for {len(lines)} lines")
    replacement = spec.props["text"]
    if replacement and not replacement.endswith(("\n", "\r")):
        replacement += "\r\n" if lines[start - 1].endswith("\r\n") else "\n"
    return "".join(lines[: start - 1]) + replacement + "".join(lines[end:])


def _write_lf(path: Path, text: str) -> None:
    """Write text with LF endings regardless of platform, so fixtures are deterministic."""
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


@pytest.fixture()
def stub_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(patch_mod, "_load_apply_op", lambda: _stub_apply_op)


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    _write_lf(tmp_path / "pkg" / "__init__.py", "")
    _write_lf(tmp_path / "pkg" / "core.py", "def add(a, b):\n    return a + b\n")
    return tmp_path


def _spec(path: str = "pkg/core.py") -> StubSpec:
    return StubSpec("replace_literal", path, (2, 2), {"text": "    return a - b"})


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)).replace(os.sep, "/"): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# --------------------------------------------------------------------------------------
# dry_run
# --------------------------------------------------------------------------------------


def test_dry_run_writes_nothing(tree: Path, stub_ops: None) -> None:
    before = _snapshot(tree)
    result = apply_to_tree(tree, "pkg/core.py", _spec(), dry_run=True)
    after = _snapshot(tree)

    assert result.ok is True
    assert result.error is None
    assert result.new_source == "def add(a, b):\n    return a - b\n"
    assert after == before, "dry_run must not create, modify or delete a single byte"


def test_dry_run_default_is_dry(tree: Path, stub_ops: None) -> None:
    before = _snapshot(tree)
    apply_to_tree(tree, "pkg/core.py", _spec())
    assert _snapshot(tree) == before


def test_dry_run_does_not_create_a_missing_file(tree: Path, stub_ops: None) -> None:
    result = apply_to_tree(tree, "pkg/new_file.py", _spec("pkg/new_file.py"), dry_run=True)
    assert result.ok is False
    assert result.error is not None
    assert not (tree / "pkg" / "new_file.py").exists()


# --------------------------------------------------------------------------------------
# real writes
# --------------------------------------------------------------------------------------


def test_apply_writes_the_new_source(tree: Path, stub_ops: None) -> None:
    result = apply_to_tree(tree, "pkg/core.py", _spec(), dry_run=False)
    assert result.ok is True
    assert (tree / "pkg" / "core.py").read_text(encoding="utf-8") == result.new_source
    assert (tree / "pkg" / "core.py").read_text(encoding="utf-8").endswith("return a - b\n")


def test_apply_leaves_no_temporary_files(tree: Path, stub_ops: None) -> None:
    apply_to_tree(tree, "pkg/core.py", _spec(), dry_run=False)
    leftovers = [p.name for p in (tree / "pkg").iterdir() if p.name.startswith(".sylphae-")]
    assert leftovers == []
    assert sorted(p.name for p in (tree / "pkg").iterdir()) == ["__init__.py", "core.py"]


def test_apply_preserves_crlf_line_endings(tree: Path, stub_ops: None) -> None:
    target = tree / "pkg" / "core.py"
    target.write_bytes(b"def add(a, b):\r\n    return a + b\r\n")
    result = apply_to_tree(tree, "pkg/core.py", _spec(), dry_run=False)
    assert result.ok is True
    raw = target.read_bytes()
    assert b"\r\n" in raw, "the untouched line must keep its CRLF ending"
    assert b"return a - b\r\n" in raw


def test_apply_reports_failure_without_writing(tree: Path, stub_ops: None) -> None:
    """Negative case: a spec that does not fit must leave the file exactly as it was."""
    before = _snapshot(tree)
    result = apply_to_tree(tree, "pkg/core.py", StubSpec("explode", "pkg/core.py", (1, 1), {}))

    assert result.ok is False
    assert result.new_source is None
    assert "StubOpApplyError" in result.error
    assert _snapshot(tree) == before


def test_apply_rejects_out_of_range_span(tree: Path, stub_ops: None) -> None:
    before = _snapshot(tree)
    result = apply_to_tree(
        tree, "pkg/core.py", StubSpec("x", "pkg/core.py", (1, 999), {"text": "x"}), dry_run=False
    )
    assert result.ok is False
    assert _snapshot(tree) == before


# --------------------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------------------


def test_missing_target_is_reported_not_raised(tree: Path, stub_ops: None) -> None:
    result = apply_to_tree(tree, "pkg/nope.py", _spec("pkg/nope.py"))
    assert result.ok is False
    assert "does not exist" in result.error


def test_target_outside_root_is_refused(tree: Path, tmp_path_factory, stub_ops: None) -> None:
    outside = tmp_path_factory.mktemp("outside")
    (outside / "secret.py").write_text("secret = 1\n", encoding="utf-8")

    for attempt in ("../secret.py", str(outside / "secret.py")):
        result = apply_to_tree(tree, attempt, _spec(attempt), dry_run=False)
        assert result.ok is False, attempt
        assert "escapes the tree root" in result.error
    assert (outside / "secret.py").read_text(encoding="utf-8") == "secret = 1\n"


def test_none_spec_is_refused(tree: Path, stub_ops: None) -> None:
    result = apply_to_tree(tree, "pkg/core.py", None)
    assert result.ok is False
    assert result.error == "spec is None"


def test_missing_sylphae_ops_is_reported_not_raised(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative case: the other workstream's module being absent must not be an ImportError."""
    monkeypatch.setitem(sys.modules, "sylphae.ops", None)
    monkeypatch.setitem(sys.modules, "sylphae.ops.apply", None)
    result = apply_to_tree(tree, "pkg/core.py", _spec(), dry_run=False)

    assert result.ok is False
    assert result.new_source is None
    assert "sylphae.ops is unavailable" in result.error


def test_apply_result_shape() -> None:
    result = ApplyResult(ok=True, new_source="x", error=None)
    assert (result.ok, result.new_source, result.error) == (True, "x", None)


# --------------------------------------------------------------------------------------
# integration with the real sylphae.ops (skipped until that workstream lands)
# --------------------------------------------------------------------------------------


def test_apply_to_tree_with_real_sylphae_ops(tree: Path) -> None:
    """End-to-end against the real ``sylphae.ops`` applier, once that workstream has landed."""
    ops_spec = pytest.importorskip("sylphae.ops.spec", reason="sylphae.ops not implemented yet")
    pytest.importorskip("sylphae.ops.apply", reason="sylphae.ops not implemented yet")

    spec = ops_spec.OpSpec(
        kind="replace_literal",
        path="pkg/core.py",
        span=(2, 2),
        props={"old_text": "    return a + b\n", "new_text": "    return a - b\n"},
    )
    before = _snapshot(tree)
    dry = apply_to_tree(tree, "pkg/core.py", spec, dry_run=True)
    assert dry.ok is True, dry.error
    assert dry.new_source == "def add(a, b):\n    return a - b\n"
    assert _snapshot(tree) == before

    result = apply_to_tree(tree, "pkg/core.py", spec, dry_run=False)
    assert result.ok is True, result.error
    assert (tree / "pkg" / "core.py").read_text(encoding="utf-8") == "def add(a, b):\n    return a - b\n"

    # A spec built from a different revision must be rejected, not applied to the wrong line.
    stale = ops_spec.OpSpec(
        kind="replace_literal",
        path="pkg/core.py",
        span=(2, 2),
        props={"old_text": "    return a + b\n", "new_text": "    return a * b\n"},
    )
    rejected = apply_to_tree(tree, "pkg/core.py", stale, dry_run=False)
    assert rejected.ok is False
    assert rejected.new_source is None
    assert "OpApplyError" in rejected.error
    assert (tree / "pkg" / "core.py").read_text(encoding="utf-8").endswith("return a - b\n")
