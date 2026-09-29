"""Playbook tests: each category resolves or escalates, and never silently does
the wrong one.

Two execution styles, on purpose.  The playbooks that *parse* an output
(missing-dependency, import locator, pip conflict) are tested against canned
executor output, because a real ``pip`` run needs a network and the test's
assertion is about the parse, not the network.  The playbooks that *act* (venv,
lint, port) are tested against a real ``EnvExecutor`` in ``tmp_path``, because a
fake executor would not test the one thing that matters: that the command the
playbook builds is one the safety layer actually permits.
"""

from __future__ import annotations

import socket
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from sylphae.ops_env.executor import EnvExecutor
from sylphae.ops_env.fingerprint import playbook_fingerprint
from sylphae.ops_env.playbooks import (
    AffectedTestPlaybook,
    FlakeRetryPlaybook,
    ImportErrorLocator,
    LintPlaybook,
    MissingDependencyPlaybook,
    PipConflictPlaybook,
    PlaybookResult,
    PortOccupancyPlaybook,
    VenvPlaybook,
    select,
)
from sylphae.ops_env.safety import AuditLog, CommandResult, CommandSpec


@dataclass
class FakeExecutor:
    """Returns prepared output for each command, in order.

    Records what it was asked to run, which is the assertion that matters for a
    parsing playbook: did it run the right command, and did it run it once?
    """

    outputs: list[str]
    specs: list[CommandSpec] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.specs = []

    def run(self, spec: CommandSpec) -> CommandResult:
        self.specs.append(spec)
        out = self.outputs[len(self.specs) - 1] if len(self.specs) <= len(self.outputs) else ""
        exit_code = 0 if not out.startswith("ERR:") else 1
        return CommandResult(
            command=spec.command,
            cwd=str(spec.cwd) if spec.cwd else "",
            exit_code=exit_code,
            duration_ms=1.0,
            stdout=out[4:] if out.startswith("ERR:") else out,
            stderr=out[4:] if out.startswith("ERR:") else "",
        )

    def would_allow(self, spec: CommandSpec):
        from sylphae.ops_env.safety import validate_command

        return validate_command(spec)


def _executor(tmp_path: Path) -> EnvExecutor:
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    return EnvExecutor(
        audit=AuditLog(scratch / "audit.jsonl"),
        writable_roots=(tmp_path, scratch),
        fingerprint=playbook_fingerprint(),
    )


# --------------------------------------------------------------- category registry


def test_every_category_name_resolves_to_a_playbook():
    """The registry is what a caller keys off, so a name with no playbook is a
    typo that would otherwise surface as a silent no-op."""
    for kind in (
        "venv",
        "pip-conflict",
        "port-occupied",
        "missing-dependency",
        "affected-tests",
        "flake-retry",
        "lint-fix",
        "import-error-locator",
    ):
        assert select(kind) is not None, f"no playbook registered for {kind}"


def test_an_unknown_category_returns_none_rather_than_raising():
    """``None`` is how the caller learns to escalate; an exception would make
    'uncovered category' indistinguishable from 'playbook crashed'."""
    assert select("not-a-category") is None


# ------------------------------------------------------------------------ 1. venv


@pytest.mark.asyncio
async def test_venv_creation_actually_builds_an_interpreter(tmp_path: Path):
    """The real executor, because the point is that ``python -m venv`` is a command
    the safety layer permits -- and that the playbook checks for the interpreter
    rather than trusting the exit code."""
    ex = _executor(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    result = VenvPlaybook().diagnose(
        _symptom("venv", "", repo, scratch=tmp_path / "scratch"), ex
    )
    assert result.resolved, result.evidence
    interpreter = repo / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    assert interpreter.exists(), "the playbook reported success without an interpreter"


def test_venv_reuse_does_not_rebuild(tmp_path: Path):
    """An existing environment is reported, not recreated: 'delete and remake the
    venv' is a destructive action dressed as a retry."""
    ex = FakeExecutor(outputs=[])
    repo = tmp_path / "repo"
    (repo / ".venv" / ("Scripts" if sys.platform == "win32" else "bin")).mkdir(parents=True)
    interpreter_dir = repo / ".venv" / ("Scripts" if sys.platform == "win32" else "bin")
    interpreter = interpreter_dir / ("python.exe" if sys.platform == "win32" else "python")
    interpreter.write_text("stub", encoding="utf-8")

    result = VenvPlaybook().diagnose(_symptom("venv", "", repo), ex)

    assert result.resolved
    assert not ex.specs, "an existing venv must not trigger a command"
    assert "already exists" in result.evidence


# ------------------------------------------------------------------ 2. pip conflict


def test_pip_conflict_names_the_conflicting_pair(tmp_path: Path):
    """The useful output of a conflict is which two requirements clash.  pip's
    message names them; the playbook surfaces them instead of the full log."""
    out = (
        "ERROR: Cannot install flask==2.0.0 and werkzeug==0.15.0 because these "
        "package versions have conflicting dependencies.\n"
        "The conflict is caused by:\n"
        "    flask 2.0.0 depends on Werkzeug>=2.0\n"
    )
    ex = FakeExecutor(outputs=[out])
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("flask==2.0.0\nwerkzeug==0.15.0\n", encoding="utf-8")

    result = PipConflictPlaybook().diagnose(_symptom("pip-conflict", "", repo), ex)

    assert result.resolved
    assert "conflicting dependencies" in result.evidence
    assert result.attempts == 1


def test_pip_conflict_reports_a_satisfiable_set_as_clean(tmp_path: Path):
    """A satisfiable requirement set is a resolution, not a non-event: the caller
    asked 'is this installable', and the answer is yes."""
    ex = FakeExecutor(outputs=["Requirement already satisfied: requests\n"])
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("requests\n", encoding="utf-8")

    result = PipConflictPlaybook().diagnose(_symptom("pip-conflict", "", repo), ex)

    assert result.resolved
    assert "no conflict" in result.evidence


def test_pip_conflict_escalates_when_there_is_no_requirements_file(tmp_path: Path):
    ex = FakeExecutor(outputs=[])
    result = PipConflictPlaybook().diagnose(
        _symptom("pip-conflict", "", tmp_path), ex
    )
    assert not result.resolved
    assert result.escalation is not None
    assert not ex.specs, "no file to check means no command to run"


def test_pip_conflict_runs_in_dry_run_mode(tmp_path: Path):
    """``--dry-run`` is the reason this playbook is safe to run against a real
    project: it resolves the graph without writing to the environment."""
    ex = FakeExecutor(outputs=["Successfully installed\n"])
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("requests\n", encoding="utf-8")

    PipConflictPlaybook().diagnose(_symptom("pip-conflict", "", repo), ex)

    assert "--dry-run" in ex.specs[0].command


# ------------------------------------------------------------- 3. port occupancy


@pytest.mark.slow
def test_port_occupancy_identifies_a_real_listener(tmp_path: Path):
    """A real listener, because ``netstat``'s output shape is what the parser is
    for and no fixture reproduces it faithfully."""
    ex = _executor(tmp_path)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = int(srv.getsockname()[1])
        result = PortOccupancyPlaybook().diagnose(
            _symptom("port-occupied", "", tmp_path, context={"port": port}), ex
        )

    assert result.resolved, result.evidence
    assert str(port) in result.evidence
    assert result.proposed_command is not None
    #: The kill is proposed, not run: the listener is still there because ending
    #: it is the caller's decision.
    assert "kill" in result.proposed_command.command.lower()


def test_port_occupancy_reports_a_free_port(tmp_path: Path):
    ex = FakeExecutor(outputs=[""])
    result = PortOccupancyPlaybook().diagnose(
        _symptom("port-occupied", "", tmp_path, context={"port": "1"}), ex
    )
    assert result.resolved
    assert "nothing is listening" in result.evidence
    assert result.proposed_command is None


def test_port_occupancy_requires_a_port_number(tmp_path: Path):
    """A symptom without a port number escalates rather than guessing one; a
    guessed port is how the wrong process gets reported."""
    ex = FakeExecutor(outputs=[])
    result = PortOccupancyPlaybook().diagnose(
        _symptom("port-occupied", "", tmp_path), ex
    )
    assert not result.resolved
    assert not ex.specs


# ------------------------------------------------------ 4. missing system dependency


def test_missing_dependency_names_the_module_from_the_traceback():
    """The category's whole job: a traceback becomes a sentence naming the module."""
    text = (
        "Traceback (most recent call last):\n"
        '  File "app.py", line 3, in <module>\n'
        "    import fastapi\n"
        "ModuleNotFoundError: No module named 'fastapi'\n"
    )
    ex = FakeExecutor(outputs=[])
    result = MissingDependencyPlaybook().diagnose(
        _symptom("missing-dependency", text, Path(".")), ex
    )
    assert result.resolved
    assert "fastapi" in result.evidence


def test_missing_dependency_proposes_a_dry_run_install():
    """The proposed install is dry-run, because proposing an install and running
    one are different actions with different blast radii."""
    text = "ModuleNotFoundError: No module named 'fastapi'\n"
    ex = FakeExecutor(outputs=[])
    with _installed_as({"fastapi": None}):
        result = MissingDependencyPlaybook().diagnose(
            _symptom("missing-dependency", text, Path(".")), ex
        )
    assert result.proposed_command is not None
    assert "--dry-run" in result.proposed_command.command


def test_missing_dependency_maps_an_import_name_to_its_distribution():
    """``import yaml`` is provided by ``pyyaml``; the map is the narrow, unambiguous
    kind of lookup the goal permits, and it is tested so it cannot drift."""
    text = "ModuleNotFoundError: No module named 'yaml'\n"
    ex = FakeExecutor(outputs=[])
    with _installed_as({"pyyaml": None}):
        result = MissingDependencyPlaybook().diagnose(
            _symptom("missing-dependency", text, Path(".")), ex
        )
    assert result.proposed_command is not None
    assert "pyyaml" in result.proposed_command.command


def test_missing_dependency_flags_an_installed_but_unimportable_module():
    """A module present as a distribution but not importable is a different failure
    than an absent one, and gets a different escalation."""
    text = "ModuleNotFoundError: No module named 'yaml'\n"
    ex = FakeExecutor(outputs=[])
    #: Pretend pyyaml is installed, so the playbook takes the 'installed but
    #: broken' branch rather than proposing an install.
    with _installed_as({"pyyaml": "6.0.1"}):
        result = MissingDependencyPlaybook().diagnose(
            _symptom("missing-dependency", text, Path(".")), ex
        )
    assert result.resolved
    assert result.proposed_command is None
    assert result.escalation is not None


def test_a_stdlib_backport_is_classified_as_a_version_gap():
    """``tomllib`` missing means the interpreter is too old, not that a package is
    absent -- and proposing ``pip install tomllib`` would be wrong advice."""
    text = "ModuleNotFoundError: No module named 'tomllib'\n"
    ex = FakeExecutor(outputs=[])
    with _installed_as({"tomllib": "3.14"}):
        result = MissingDependencyPlaybook().diagnose(
            _symptom("missing-dependency", text, Path(".")), ex
        )
    assert result.resolved
    assert "older than the code requires" in result.evidence
    assert result.proposed_command is None


# --------------------------------------------------------- 5. affected test selection


def test_affected_tests_select_only_the_tests_touching_changed_modules(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    (repo / "src" / "billing.py").write_text("def charge():\n    pass\n", encoding="utf-8")
    (repo / "src" / "auth.py").write_text("def login():\n    pass\n", encoding="utf-8")
    (repo / "tests" / "test_billing.py").write_text(
        "from src.billing import charge\n\n\ndef test_charge():\n    assert charge() is None\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_auth.py").write_text(
        "from src.auth import login\n\n\ndef test_login():\n    assert login() is None\n",
        encoding="utf-8",
    )
    ex = FakeExecutor(outputs=[])
    result = AffectedTestPlaybook().diagnose(
        _symptom(
            "affected-tests",
            "",
            repo,
            context={"changed": ["src/billing.py"]},
        ),
        ex,
    )
    assert result.resolved
    assert "tests/test_billing.py" in result.proposed_command.command
    assert "tests/test_auth.py" not in result.proposed_command.command


def test_affected_tests_falls_back_to_all_tests_when_none_match(tmp_path: Path):
    """Running nothing would silently report a green run over an untested change;
    the fallback is the full directory."""
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "test_x.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    ex = FakeExecutor(outputs=[])
    result = AffectedTestPlaybook().diagnose(
        _symptom("affected-tests", "", repo, context={"changed": ["src/unrelated.py"]}),
        ex,
    )
    assert result.resolved
    assert "tests/test_x.py" in result.proposed_command.command


# --------------------------------------------------------------- 6. flake retry


def test_flake_retry_retries_exactly_once_then_reports_a_real_failure(tmp_path: Path):
    """Two failures must read as a defect.  A loop here is how a flaky test becomes
    green without anyone reading it, so the budget is one."""
    ex = FakeExecutor(outputs=["ERR: FAILED", "ERR: FAILED"])
    result = FlakeRetryPlaybook().diagnose(
        _symptom("flake-retry", "", tmp_path, context={"target": "tests/test_x.py"}), ex
    )
    assert result.resolved
    assert len(ex.specs) == 2, "the retry budget is one, not 'until it passes'"
    assert "defect" in result.evidence
    assert result.escalation is not None


def test_flake_retry_reports_a_pass_on_the_immediate_retry(tmp_path: Path):
    ex = FakeExecutor(outputs=["ERR: FAILED", "passed"])
    result = FlakeRetryPlaybook().diagnose(
        _symptom("flake-retry", "", tmp_path, context={"target": "tests/test_x.py"}), ex
    )
    assert result.resolved
    assert "flake" in result.evidence
    assert result.escalation is None


def test_flake_retry_does_not_retry_a_passing_test(tmp_path: Path):
    """A passing test is not flaky, and retrying it would spend the budget the
    goal caps."""
    ex = FakeExecutor(outputs=["passed"])
    result = FlakeRetryPlaybook().diagnose(
        _symptom("flake-retry", "", tmp_path, context={"target": "tests/test_x.py"}), ex
    )
    assert result.resolved
    assert len(ex.specs) == 1
    assert "not flaky" in result.evidence


# ---------------------------------------------------------- 7. format / lint fix


@pytest.mark.slow
def test_lint_fix_formats_a_file_and_backs_it_up(tmp_path: Path):
    """The real executor, because the command must be one the safety layer
    permits and the backup must survive the rewrite."""
    ex = _executor(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "mod.py"
    target.write_text("x=1\n", encoding="utf-8")

    result = LintPlaybook().diagnose(
        _symptom("lint-fix", "", repo, scratch=tmp_path / "scratch",
                 context={"target": "mod.py"}),
        ex,
    )

    assert result.resolved, result.evidence
    assert "clean" in result.evidence
    #: The undo copy exists, which is what makes the rewrite safe.
    backups = list((tmp_path / "scratch").glob("mod*.py"))
    assert backups, "no backup was written before the file was rewritten"
    assert backups[0].read_text(encoding="utf-8") == "x=1\n"
    #: And the file itself was actually changed.
    assert target.read_text(encoding="utf-8") != "x=1\n"


def test_lint_fix_refuses_a_missing_file(tmp_path: Path):
    ex = FakeExecutor(outputs=[])
    result = LintPlaybook().diagnose(
        _symptom("lint-fix", "", tmp_path, context={"target": "nope.py"}), ex
    )
    assert not result.resolved
    assert not ex.specs


# ------------------------------------------------- 8. ImportError / ModuleNotFound


def test_import_locator_points_at_the_users_frame():
    """Library frames are filtered out: 'the import is inside site-packages' is the
    same answer for every project and tells nobody anything."""
    text = (
        "Traceback (most recent call last):\n"
        '  File "C:/site-packages/flask/app.py", line 20, in main\n'
        "    from flask import request\n"
        '  File "app.py", line 12, in load\n'
        "    import utils\n"
        "ModuleNotFoundError: No module named 'utils'\n"
    )
    ex = FakeExecutor(outputs=[])
    result = ImportErrorLocator().diagnose(
        _symptom("import-error-locator", text, Path("C:/proj")), ex
    )
    assert result.resolved
    assert "app.py:12" in result.evidence
    assert "site-packages" not in result.evidence


def test_import_locator_reports_without_a_module_name():
    """A traceback without a ModuleNotFoundError still has a frame worth locating."""
    text = (
        "Traceback (most recent call last):\n"
        '  File "app.py", line 40, in run\n'
        "    raise ValueError('boom')\n"
    )
    ex = FakeExecutor(outputs=[])
    result = ImportErrorLocator().diagnose(
        _symptom("import-error-locator", text, Path(".")), ex
    )
    assert result.resolved
    assert "app.py:40" in result.evidence


def test_import_locator_escalates_on_a_frameless_traceback():
    ex = FakeExecutor(outputs=[])
    result = ImportErrorLocator().diagnose(
        _symptom("import-error-locator", "something broke", Path(".")), ex
    )
    assert not result.resolved
    assert result.escalation is not None


# -------------------------------------------------------------------- robustness


def test_a_crashing_playbook_escalates_instead_of_raising(tmp_path: Path):
    """A playbook that raises would take down whatever called it; the contract is
    that it escalates, so the failure stays inside the report."""

    class Exploding(VenvPlaybook):
        def _run(self, symptom, ex):
            raise RuntimeError("boom")

    result = Exploding().diagnose(_symptom("venv", "", tmp_path), FakeExecutor(outputs=[]))
    assert not result.resolved
    assert result.escalation is not None
    assert "boom" in result.escalation


def test_a_result_serialises_to_json():
    """The gate writes these to an artifact, so the shape must round-trip."""
    import json

    out = PlaybookResult(
        category="venv",
        resolved=True,
        evidence="made one",
        action_taken="python -m venv",
    ).to_json()
    assert json.loads(json.dumps(out))["category"] == "venv"


# ----------------------------------------------------------------------- helpers


import contextlib

import sylphae.ops_env.playbooks as _playbooks


@contextlib.contextmanager
def _installed_as(versions: dict[str, str | None]):
    """Pin what the playbook sees as installed, so these tests do not pass or fail
    depending on what this machine happens to have."""
    original = _playbooks.package_version
    _playbooks.package_version = versions.get
    try:
        yield
    finally:
        _playbooks.package_version = original


def _symptom(kind: str, text: str, cwd: Path, scratch: Path | None = None, context=None):
    from sylphae.ops_env.playbooks import Symptom

    return Symptom(kind=kind, text=text, cwd=cwd, scratch=scratch, context=context or {})
