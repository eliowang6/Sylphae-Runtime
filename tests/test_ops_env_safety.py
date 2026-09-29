"""The safety layer is the part of C3 OPS that must be unable to fail.

These tests exist because the goal makes a destructive command a hard failure
rather than a bug: an ops layer that can delete work is a data-loss feature, and
the only way to keep that property is to make it untestable-in-reverse -- every
shape that should be refused is refused, and the refusals are themselves recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sylphae.ops_env.safety import (
    AuditLog,
    CommandSpec,
    Refusal,
    classify_destructive,
    make_audit,
    validate_command,
)


def _spec(program: str, args: tuple[str, ...] = (), **kw):
    return CommandSpec(program=program, args=tuple(args), **kw)


@pytest.mark.parametrize(
    ("program", "args"),
    [
        ("rm", ("-rf", "tmp")),
        ("rm", ("-fr", "tmp")),
        ("rm", ("-r", "-f", "tmp")),
        ("rm", ("file.txt",)),
        ("cmd", ("/c", "del", "secret.txt")),
        ("cmd", ("/c", "rd", "/s", "/q", "dir")),
        ("format", ("C:",)),
        ("format", ("/fs:ntfs", "D:")),
        ("mkfs.ext4", ("/dev/sda1",)),
        ("dd", ("if=/dev/zero", "of=/dev/sda")),
        ("git", ("commit", "--amend")),
        ("git", ("push", "--force", "origin", "main")),
        ("git", ("push", "-f", "origin", "main")),
        ("git", ("reset", "--hard", "HEAD~1")),
        ("git", ("clean", "-fd")),
        ("shred", ("-u", "file")),
        ("truncate", ("-s", "0", "log.txt")),
        ("taskkill", ("/F", "/PID", "1234")),
        ("kill", ("-9", "1234")),
        ("kill", ("1234",)),
        ("chmod", ("-R", "777", "/")),
    ],
)
def test_destructive_shapes_are_refused(program: str, args: tuple[str, ...]):
    refusal = validate_command(_spec(program, args))
    assert refusal is not None, f"{program} {args} was not refused"


@pytest.mark.parametrize(
    ("program", "args"),
    [
        ("python", ("-m", "venv", ".venv")),
        ("python", ("-m", "pip", "install", "-r", "requirements.txt")),
        ("pip", ("install", "--dry-run", "requests")),
        ("pip", ("check",)),
        ("pytest", ("-q", "tests/test_a.py")),
        ("pytest", ("--co", "-q", "--tb", "short")),
        ("ruff", ("check", "--fix", "src/a.py")),
        ("ruff", ("format", "--no-cache", "src/a.py")),
        ("rg", ("-i", "-n", "needle", ".")),
        ("git", ("status", "--porcelain")),
        ("git", ("diff", "--name-only")),
        ("netstat", ("-ano",)),
        ("where", ("python",)),
    ],
)
def test_benign_shapes_are_allowed(program: str, args: tuple[str, ...]):
    refusal = validate_command(_spec(program, args))
    assert refusal is None, f"{program} {args} refused for {refusal.reason}: {refusal.detail}"


def test_an_unlisted_program_is_refused():
    """The allowlist is closed.  A harmless but unknown program is refused, because
    an ops layer whose behaviour is predictable from the code cannot include
    programs the code has never seen."""
    refusal = validate_command(_spec("curl", ("http://example.com/",)))
    assert refusal is not None
    assert refusal.reason == "program-not-allowlisted"


def test_an_unlisted_flag_is_refused():
    """Flags are a closed set for the same reason; a new flag is a diff, not a
    runtime discovery."""
    refusal = validate_command(_spec("ruff", ("check", "--some-new-flag", "a.py")))
    assert refusal is not None
    assert refusal.reason == "flag-not-allowlisted"


def test_a_glob_is_refused_for_programs_that_do_not_widen():
    """``pytest`` may take a glob (it narrows a run); ``pip`` may not, because a
    glob in a pip argument names an unpredictable set of packages."""
    assert validate_command(_spec("pytest", ("-q", "tests/*.py"))) is None
    refusal = validate_command(_spec("pip", ("install", "pkg*.whl")))
    assert refusal is not None
    assert refusal.reason == "glob-not-permitted"


def test_a_parent_path_escape_is_refused():
    """``..`` is refused in any position: it is how a path argument reaches outside
    the directory the caller declared writable."""
    refusal = validate_command(_spec("ruff", ("check", "../outside.py")))
    assert refusal is not None
    assert refusal.reason == "forbidden-substring"


def test_a_path_outside_the_writable_roots_is_refused():
    """``ruff check`` may rewrite, so its positional path must stay inside the
    declared boundary.  This is the check that keeps a format verb inside the
    repository rather than pointing it at an absolute path elsewhere."""
    spec = _spec("ruff", ("check", "--fix", "C:/elsewhere/a.py"))
    spec.writable_roots = (Path("D:/repo"),)
    refusal = validate_command(spec)
    assert refusal is not None
    assert refusal.reason == "path-outside-writable-roots"


def test_a_relative_path_is_not_misread_as_outside():
    """Relative paths are resolved against the executor's cwd at run time, so the
    validator must not treat them as absolute escapes."""
    spec = _spec("ruff", ("check", "--fix", "src/a.py"))
    spec.writable_roots = (Path("D:/repo"),)
    assert validate_command(spec) is None


def test_classify_destructive_is_a_pure_function():
    """The caller can ask "would this be refused?" and get the run-time answer,
    which is what lets a playbook report an intended command it will not run."""
    assert classify_destructive("rm", ("-rf", "x")) is not None
    assert classify_destructive("git", ("status",)) is None


def test_a_refusal_is_truthy():
    """``refusal or result`` is the idiom in the executor; it only reads correctly
    when a refusal is truthy and a missing one is falsy."""
    assert Refusal("reason") is not None
    assert bool(Refusal("reason")) is True
    assert bool(None) is False


def test_an_audit_record_hashes_but_does_not_store_the_output(tmp_path: Path):
    """The goal asks for an output *summary hash*, not the output.  Stdout can
    carry a secret, and a log that keeps it is a second copy of that secret."""
    record = make_audit(
        _spec("pytest", ("-q",)),
        exit_code=0,
        duration_ms=12.5,
        stdout="a" * 10000,
        stderr="",
        refusal=None,
        timed_out=False,
    )
    assert record.stdout_bytes == 10000
    assert len(record.stdout_sha256) == 64
    assert "a" * 10000 not in json.dumps(record.to_json())


def test_truncation_is_marked():
    """A truncated transcript must be distinguishable from a short one, otherwise
    a cut traceback reads as a complete traceback."""
    long_text = "x" * 20000
    out, n = make_audit.__globals__["_truncate"](long_text, 100)
    assert n == 20000
    assert out.endswith("[truncated: 19900 more chars]")
    short, _ = make_audit.__globals__["_truncate"]("short", 100)
    assert short == "short"


def test_the_audit_log_records_refusals_as_well_as_runs(tmp_path: Path):
    """An attempted-but-blocked command is an audit event.  Without this, the trail
    shows every executed command and silently omits every dangerous one."""
    log = AuditLog(tmp_path / "audit.jsonl")
    record = make_audit(
        _spec("rm", ("-rf", "x")),
        exit_code=None,
        duration_ms=0.0,
        stdout="",
        stderr="",
        refusal=Refusal("destructive-pattern", "-rf"),
        timed_out=False,
    )
    log.append(record)
    entries = log.read()
    assert len(entries) == 1
    assert entries[0]["refused"] is True
    assert entries[0]["refusal_reason"] == "destructive-pattern"
    assert entries[0]["exit_code"] is None


def test_parsing_a_command_string_preserves_the_program_and_args():
    spec = CommandSpec.parse("pytest -q tests/test_a.py", cwd=Path("."))
    assert spec.program == "pytest"
    assert spec.args == ("-q", "tests/test_a.py")
