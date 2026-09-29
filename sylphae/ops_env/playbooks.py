"""The eight ops categories the goal names, as playbooks.

A playbook is a ``diagnose`` function: given a symptom, it either resolves it
with an allowlisted command, or escalates with the evidence attached.  It never
raises, and it never runs a command that ``EnvExecutor`` would refuse without
saying so first.

The boundary between "resolve" and "escalate" is drawn by one rule: an action
that could change the user's machine outside the declared scratch directory is
proposed, not taken.  Killing the process on a port is the clearest case --
identifying it is safe and automatic, ending it is a decision.  The playbook
reports the PID and the command that would end it, and counts the identification
as the resolution; the kill is a separate call with explicit authorisation.

The eight categories (goal M3): venv creation, ``pip install`` conflicts, port
occupancy, missing system dependencies, affected-test selection, flake retry
(<=1), format/lint auto-fix, and ``ImportError``/``ModuleNotFoundError``
localization.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sylphae.ops_env.executor import EnvExecutor
from sylphae.ops_env.fingerprint import package_version
from sylphae.ops_env.safety import CommandSpec, Refusal

#: How many times a playbook may retry the same problem before handing it back.
#: The goal sets this at 3; the flake category sets it at 1 for the *test* retry
#: because that is what "flake" means -- a second failure is a real failure.
_MAX_ATTEMPTS = 3

_MODULE_NOT_FOUND_RE = re.compile(
    r"ModuleNotFoundError:\s*No module named ['\"]?([A-Za-z0-9_\.]+)"
)
_IMPORT_ERROR_RE = re.compile(
    r"ImportError:\s*(?:cannot import name ['\"]([A-Za-z0-9_]+)['\"] from ['\"]([A-Za-z0-9_\.]+)['\"]|(.+))"
)
_TRACEBACK_FRAME_RE = re.compile(r'^\s*File "([^"]+)", line (\d+), in (.+)$', re.M)
_PORT_BINDING_RE = re.compile(r"(?:Address already in use|Errno 10048|bind\(\).*10048)", re.I)


@dataclass
class Symptom:
    """What the caller noticed.  ``kind`` selects the playbook."""

    kind: str
    text: str
    cwd: Path
    scratch: Path | None = None
    #: Facts the caller already has, so the playbook does not rediscover them.
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlaybookResult:
    """What the playbook concluded.

    ``resolved`` means the category's job is done, not that the underlying
    problem is gone: for "port occupied", resolved means "the occupant is
    identified and named"; the occupant is still there until someone authorises
    the kill.  ``escalation`` says why a human or a model is needed next.
    """

    category: str
    resolved: bool
    evidence: str
    action_taken: str = ""
    escalation: str | None = None
    attempts: int = 0
    #: A command the caller could run, validated but not executed.  Present when
    #: resolving needs a decision the playbook will not make.
    proposed_command: CommandSpec | None = None
    #: Every command this playbook ran, in order, for the report.
    commands: list[str] = field(default_factory=list)
    refusal: Refusal | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "resolved": self.resolved,
            "evidence": self.evidence,
            "action_taken": self.action_taken,
            "escalation": self.escalation,
            "attempts": self.attempts,
            "proposed_command": self.proposed_command.command
            if self.proposed_command
            else None,
            "commands": self.commands,
            "refusal": self.refusal.reason if self.refusal else None,
        }


class Playbook:
    """Base class: a name and a diagnose method that cannot raise."""

    category: str = ""

    def diagnose(self, symptom: Symptom, ex: EnvExecutor) -> PlaybookResult:
        try:
            return self._run(symptom, ex)
        except Exception as exc:  # noqa: BLE001 - a playbook escalates, never raises
            return PlaybookResult(
                category=self.category,
                resolved=False,
                evidence=f"playbook crashed: {exc!r}",
                escalation=f"the {self.category} playbook raised {exc!r}",
            )

    def _run(self, symptom: Symptom, ex: EnvExecutor) -> PlaybookResult:
        raise NotImplementedError

    def _exec(self, ex: EnvExecutor, spec: CommandSpec, result: PlaybookResult) -> str:
        """Run ``spec`` through ``ex`` and fold the outcome into ``result``."""
        result.commands.append(spec.command)
        outcome = ex.run(spec)
        if outcome.refusal is not None:
            result.refusal = outcome.refusal
            result.escalation = (
                f"{outcome.refusal.reason}: {outcome.refusal.detail}"
            )
            return ""
        return outcome.stdout + ("\n" + outcome.stderr if outcome.stderr.strip() else "")


# --------------------------------------------------------------------------- 1. venv


class VenvPlaybook(Playbook):
    """Create a virtual environment, or report that one already exists.

    Resolves when there is a usable interpreter at the expected path.  Refuses to
    recreate over an existing one -- "the venv is broken, delete it" is a
    destructive action, not an ops one, and would need the trash module's undo
    semantics plus an explicit caller request.
    """

    category = "venv"

    def _run(self, symptom: Symptom, ex: EnvExecutor) -> PlaybookResult:
        cwd = Path(symptom.context.get("cwd") or symptom.cwd)
        venv_dir = Path(symptom.context.get("venv_dir") or cwd / ".venv")
        #: Resolved against the command's cwd rather than the process's: the
        #: interpreter is checked after the command returns, and a relative
        #: ``venv_dir`` means one frame of reference for the launch and another
        #: for the check.
        if not venv_dir.is_absolute():
            venv_dir = cwd / venv_dir
        result = PlaybookResult(
            category=self.category, resolved=False, evidence="", attempts=0
        )
        interpreter = venv_dir / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        if interpreter.exists():
            result.resolved = True
            result.evidence = f"a virtualenv already exists at {venv_dir}"
            result.action_taken = "none; reused the existing environment"
            return result

        scratch = Path(symptom.scratch or cwd / ".sylphae-scratch")
        spec = CommandSpec(
            program=sys.executable,
            args=("-m", "venv", str(venv_dir)),
            cwd=cwd,
            writable_roots=(cwd, scratch),
        )
        out = self._exec(ex, spec, result)
        if result.refusal:
            return result
        result.attempts += 1
        if interpreter.exists():
            result.resolved = True
            result.evidence = f"created {venv_dir}"
            result.action_taken = spec.command
            return result

        result.evidence = (
            f"`python -m venv` did not produce {interpreter}; output follows.\n{out}"
        )
        result.escalation = (
            "venv creation reported success but no interpreter appeared; the "
            "ensurepip step may have failed (common on Debian without python3-venv)"
        )
        return result


# ------------------------------------------------------------------ 2. pip conflict


class PipConflictPlaybook(Playbook):
    """Find the conflicting pair in a dependency install, without forcing a fix.

    Resolves by *naming* the conflict: which two requirements cannot hold at
    once.  It deliberately does not resolve by choosing a version, because
    picking a dependency version is a decision about the user's software, not
    about their environment.
    """

    category = "pip-conflict"

    _CONFLICT_RE = re.compile(
        r"(?im)^.*(cannot install [^\n]*|because these package versions have conflicting dependencies[^\n]*|ERROR: [Cc]annot install[^\n]*)$"
    )
    _PAIR_RE = re.compile(r"(?im)^\s*([A-Za-z0-9_.\-]+[A-Za-z0-9_])[=<>!~^].*")

    def _run(self, symptom: Symptom, ex: EnvExecutor) -> PlaybookResult:
        cwd = Path(symptom.context.get("cwd") or symptom.cwd)
        req_file = Path(symptom.context.get("requirements") or cwd / "requirements.txt")
        result = PlaybookResult(
            category=self.category, resolved=False, evidence="", attempts=0
        )
        if not req_file.exists():
            result.evidence = f"no requirements file at {req_file}"
            result.escalation = "the caller must say which requirements file to check"
            return result

        scratch = Path(symptom.scratch or cwd / ".sylphae-scratch")
        #: ``--dry-run`` resolves the dependency graph without installing
        #: anything, which is what makes this safe to run on a real project.
        spec = CommandSpec(
            program=sys.executable,
            args=(
                "-m",
                "pip",
                "install",
                "--dry-run",
                "--no-input",
                "--disable-pip-version-check",
                "-r",
                str(req_file.name),
            ),
            cwd=cwd,
            timeout_s=120.0,
            writable_roots=(scratch,),
        )
        out = self._exec(ex, spec, result)
        result.attempts += 1
        if result.refusal:
            return result

        conflicts = [line.strip() for line in out.splitlines() if self._CONFLICT_RE.match(line)]
        if conflicts:
            result.resolved = True
            result.evidence = "\n".join(conflicts[:6])
            result.action_taken = "resolved the dependency graph in dry-run mode"
            return result

        if "Successfully" in out or "already satisfied" in out:
            result.resolved = True
            result.evidence = "no conflict: the requirement set is satisfiable"
            result.action_taken = "resolved the dependency graph in dry-run mode"
            return result

        result.evidence = out.strip() or "pip produced no output"
        result.escalation = (
            "pip neither reported a conflict nor succeeded; the output is not a "
            "shape this playbook recognises"
        )
        return result


# ------------------------------------------------------------- 3. port occupancy


class PortOccupancyPlaybook(Playbook):
    """Identify what holds a port, and propose ending it.

    Resolution is the identification.  Ending the process is returned as
    ``proposed_command`` -- validated, described, not run -- because the process
    on that port may be the user's database, and an ops layer that kills it
    automatically is a hazard wearing a feature's clothes.
    """

    category = "port-occupied"

    def _run(self, symptom: Symptom, ex: EnvExecutor) -> PlaybookResult:
        port = str(symptom.context.get("port") or symptom.context.get("PORT") or "")
        result = PlaybookResult(
            category=self.category, resolved=False, evidence="", attempts=0
        )
        if not port.isdigit():
            result.evidence = f"no port number in symptom context: {symptom.context!r}"
            result.escalation = "the caller must name the occupied port"
            return result

        scratch = Path(symptom.scratch or Path(symptom.cwd) / ".sylphae-scratch")
        if sys.platform == "win32":
            listing = self._exec(
                ex,
                CommandSpec(
                    program="netstat",
                    args=("-ano",),
                    cwd=Path(symptom.cwd),
                    writable_roots=(scratch,),
                ),
                result,
            )
        else:
            listing = self._exec(
                ex,
                CommandSpec(
                    program="ss",
                    args=("-ltnp",),
                    cwd=Path(symptom.cwd),
                    writable_roots=(scratch,),
                ),
                result,
            )
        result.attempts += 1
        if result.refusal:
            return result

        pid = self._find_pid(listing, port)
        if pid is None:
            result.resolved = True
            result.evidence = f"nothing is listening on port {port} now"
            result.action_taken = "listed all listeners and found none on that port"
            return result

        who = self._identify(ex, pid, result, Path(symptom.cwd), scratch)
        result.resolved = True
        result.evidence = f"port {port} is held by PID {pid}{who}"
        result.action_taken = "identified the occupying process; no action taken"
        result.proposed_command = CommandSpec(
            program="taskkill" if sys.platform == "win32" else "kill",
            args=("/F", "/T", "/PID", pid) if sys.platform == "win32" else (pid,),
            cwd=Path(symptom.cwd),
            writable_roots=(scratch,),
            authorisation="process-kill",
        )
        #: The refusal is the point of carrying it: ``taskkill`` is in the
        #: destructive list, so ``proposed_command`` is a *description* of what
        #: the caller could do, and running it would be refused unless the
        #: safety layer is told to accept the authorisation.  The playbook hands
        #: it back rather than reaching for that path itself.
        check = ex.would_allow(result.proposed_command)
        if check is not None:
            result.refusal = check
        return result

    @staticmethod
    def _find_pid(listing: str, port: str) -> str | None:
        want = f":{port}"
        for line in listing.splitlines():
            if want not in line:
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            if want not in fields[1]:
                continue
            #: ``LISTENING`` on Windows / ``LISTEN`` on posix; a connection in
            #: TIME_WAIT is not an occupant, it is a closing socket.
            upper = line.upper()
            if "LISTEN" not in upper:
                continue
            return fields[-1]
        return None

    @staticmethod
    def _identify(
        ex: EnvExecutor, pid: str, result: PlaybookResult, cwd: Path, scratch: Path
    ) -> str:
        if sys.platform != "win32":
            return ""
        out = ex.run(
            CommandSpec(
                program="tasklist",
                args=("/FI", f"PID eq {pid}", "/FO", "LIST"),
                cwd=cwd,
                writable_roots=(scratch,),
            )
        )
        result.commands.append(out.command)
        if out.ok:
            for line in out.stdout.splitlines():
                if ":" in line and line.split(":", 1)[0].strip() in ("Image Name", "进程名", "映像名称"):
                    name = line.split(":", 1)[1].strip()
                    return f" ({name})"
        return ""


# ------------------------------------------------------ 4. missing system dependency


class MissingDependencyPlaybook(Playbook):
    """Name the missing module and say whether pip can provide it.

    This is the category that turns a traceback into a sentence.  A
    ``ModuleNotFoundError`` is already the answer; the playbook's job is to
    connect it to the environment: is the module installable, is it a system
    library that pip cannot supply, and is it already present under a different
    name.
    """

    category = "missing-dependency"

    #: Modules that are part of the stdlib in some versions and not others; a
    #: ``ModuleNotFoundError`` for one of these means the interpreter is too old,
    #: not that a package is missing.
    _STDLIB_MOVED = frozenset(
        {
            "dataclasses",
            "typing_extensions",
            "importlib.metadata",
            "importlib.metadata.version",
            "contextvars",
            "graphlib",
            "zoneinfo",
            "tomllib",
        }
    )

    def _run(self, symptom: Symptom, ex: EnvExecutor) -> PlaybookResult:
        text = symptom.text or ""
        result = PlaybookResult(
            category=self.category, resolved=False, evidence="", attempts=0
        )
        match = _MODULE_NOT_FOUND_RE.search(text)
        if match is None:
            #: An ImportError that is not "no module named" is a different shape:
            #: a name exists in the package but not where the import expects it.
            #: That is a code bug, not an environment one.
            imp = _IMPORT_ERROR_RE.search(text)
            if imp is None:
                result.evidence = "no ModuleNotFoundError or ImportError in the symptom text"
                result.escalation = (
                    "this playbook only resolves missing-module errors; other "
                    "tracebacks belong to the model"
                )
                return result
            name, package, rest = imp.groups()
            result.resolved = True
            result.evidence = (
                f"the import of {name or 'a name'} from {package or 'a package'} "
                f"fails: {rest or 'the symbol is not exported there'}"
            )
            result.action_taken = "located the failing import in the traceback"
            result.escalation = (
                "an ImportError inside an importable package is a code defect, "
                "not an environment one; the fix is an edit"
            )
            return result

        module = match.group(1)
        top = module.split(".")[0]
        if top in self._STDLIB_MOVED:
            result.resolved = True
            result.evidence = (
                f"{module} is missing, which for this module usually means the "
                f"interpreter is older than the code requires"
            )
            result.action_taken = "classified the missing module as a version gap"
            result.escalation = "upgrade the interpreter, or vendor the backport"
            return result

        version = package_version(_pip_name(top))
        result.resolved = True
        result.evidence = f"missing module: {module}"
        result.action_taken = "named the missing module from the traceback"
        if version is not None:
            result.evidence += f"; {top} is installed as {version}"
            result.escalation = (
                f"{top} is installed but {module} is not importable -- the "
                "installed distribution may be a stub, or the import path may be wrong"
            )
        else:
            result.proposed_command = CommandSpec(
                program=sys.executable,
                args=(
                    "-m",
                    "pip",
                    "install",
                    "--dry-run",
                    "--no-input",
                    "--disable-pip-version-check",
                    _pip_name(top),
                ),
                cwd=Path(symptom.cwd),
            )
        return result


_PIP_NAME_OVERRIDES = {
    "yaml": "pyyaml",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4",
    "OpenSSL": "pyopenssl",
    "Crypto": "pycryptodome",
    "dateutil": "python-dateutil",
    "magic": "python-magic",
}


def _pip_name(top: str) -> str:
    """The distribution name for an import name, where they differ.

    Narrow and unambiguous by construction: these are the pairs where the import
    name and the distribution name are simply different strings and no guesswork
    is involved.  Anything not here is passed through unchanged, which is the
    correct default since most names do match.
    """
    return _PIP_NAME_OVERRIDES.get(top, top)


# --------------------------------------------------------- 5. affected test selection


class AffectedTestPlaybook(Playbook):
    """Choose which tests to run from which files changed.

    The honest version of "run only affected tests" needs a dependency graph.
    A static import graph is built from the repository's own import statements,
    because that is available without executing the code -- running the code is
    what the tests are for.
    """

    category = "affected-tests"

    def _run(self, symptom: Symptom, ex: EnvExecutor) -> PlaybookResult:
        cwd = Path(symptom.context.get("cwd") or symptom.cwd)
        changed = symptom.context.get("changed") or symptom.context.get("changed_files")
        result = PlaybookResult(
            category=self.category, resolved=False, evidence="", attempts=0
        )
        if not changed:
            result.evidence = "no changed files were named"
            result.escalation = "the caller must say which files changed"
            return result
        changed_paths = [Path(c) for c in changed]
        tests_dir = Path(symptom.context.get("tests_dir") or cwd / "tests")
        if not tests_dir.exists():
            result.resolved = True
            result.evidence = f"no tests directory at {tests_dir}; nothing to select"
            result.action_taken = "reported the absence of a tests directory"
            return result

        #: The relation runs from the *changed* module to the tests that name it,
        #: not the other way round: a changed file whose own body imports nothing
        #: is still imported by something, and that something is what must run.
        changed_tokens: set[str] = set()
        for path in changed_paths:
            if path.suffix != ".py":
                continue
            parts = path.with_suffix("").parts
            if not parts:
                continue
            #: ``src/billing.py`` -> {billing, src.billing}.  The bare top-level
            #: package is skipped when it is ``src``/``tests``, because those
            #: names match half the repository and select nothing useful.
            changed_tokens.add(parts[-1])
            if len(parts) > 1 and parts[0] not in ("src", "tests", "test"):
                changed_tokens.add(".".join(parts))

        selected: list[str] = []
        for test in sorted(tests_dir.rglob("test_*.py")):
            rel = test.relative_to(cwd).as_posix()
            body = test.read_text(encoding="utf-8", errors="replace")
            #: Matched against import lines only, and never the whole file: the
            #: word ``billing`` in a docstring is not a dependency on billing.
            import_lines = "\n".join(
                line
                for line in body.splitlines()
                if re.match(r"^\s*(?:from|import)\s+", line)
            )
            if any(tok in import_lines for tok in changed_tokens):
                selected.append(rel)
        if not selected:
            selected = sorted(
                p.relative_to(cwd).as_posix()
                for p in tests_dir.rglob("test_*.py")
            )
            result.evidence = (
                "no test file imports a changed module; falling back to the full "
                "tests directory rather than running nothing"
            )
        else:
            result.evidence = (
                f"{len(selected)} test file(s) import one of the changed modules "
                f"({sorted(changed_tokens)[:6]})"
            )
        result.resolved = True
        result.action_taken = "built a static import graph over the changed files"
        result.proposed_command = CommandSpec(
            program="pytest",
            args=("--co", "-q", *selected[:12]),
            cwd=cwd,
            writable_roots=(Path(symptom.scratch or cwd / ".sylphae-scratch"),),
        )
        return result


# --------------------------------------------------------------- 6. flake retry


class FlakeRetryPlaybook(Playbook):
    """Re-run a failed test exactly once.

    "Flake" means the failure is not reproducible, which is only knowable by
    running it again.  A second failure is a real failure, so the retry budget is
    one, and the result reports which outcome it was rather than reporting the
    last one alone.
    """

    category = "flake-retry"

    def _run(self, symptom: Symptom, ex: EnvExecutor) -> PlaybookResult:
        cwd = Path(symptom.context.get("cwd") or symptom.cwd)
        target = symptom.context.get("target") or symptom.context.get("test")
        result = PlaybookResult(
            category=self.category, resolved=False, evidence="", attempts=0
        )
        if not target:
            result.evidence = "no failing test target was named"
            result.escalation = "the caller must say which test failed"
            return result

        scratch = Path(symptom.scratch or cwd / ".sylphae-scratch")
        args = ("-q", "--tb=short", str(target))
        first = ex.run(
            CommandSpec(
                program="pytest",
                args=args,
                cwd=cwd,
                timeout_s=120.0,
                writable_roots=(scratch,),
            )
        )
        result.commands.append(first.command)
        result.attempts += 1
        if first.ok:
            result.resolved = True
            result.evidence = "the target passed on the first run; it was not flaky"
            result.action_taken = "ran the target once"
            return result

        #: One retry.  Not two, and not the caller's choice: the goal sets the
        #: flake budget at <=1, and a retry loop is how a flaky test becomes a
        #: green test without anyone reading it.
        retry = ex.run(
            CommandSpec(
                program="pytest",
                args=args,
                cwd=cwd,
                timeout_s=120.0,
                writable_roots=(scratch,),
            )
        )
        result.commands.append(retry.command)
        result.attempts += 1
        result.resolved = True
        result.action_taken = f"ran the target twice ({first.command})"
        if retry.ok:
            result.evidence = (
                "failed then passed on the immediate retry: consistent with a "
                "flake, not with a defect"
            )
        else:
            result.evidence = (
                "failed on both runs; this is a defect, and the retry budget is spent"
            )
            result.escalation = (
                "the failure reproduces; hand the traceback to the model rather "
                "than retrying again"
            )
        return result


# ---------------------------------------------------------- 7. format / lint fix


class LintPlaybook(Playbook):
    """Auto-format a file and verify the result is clean.

    Two properties, both checked rather than assumed.  The command must be
    ``--fix``, which rewrites the file -- so the scratch directory is what keeps
    that rewrite undoable, and the playbook copies the original there before
    touching it.  And "fixed" is verified by re-running in check mode, because a
    formatter that reports success and leaves violations is how a lint step
    silently stops working.
    """

    category = "lint-fix"

    def _run(self, symptom: Symptom, ex: EnvExecutor) -> PlaybookResult:
        cwd = Path(symptom.context.get("cwd") or symptom.cwd)
        target = symptom.context.get("target") or symptom.context.get("file")
        result = PlaybookResult(
            category=self.category, resolved=False, evidence="", attempts=0
        )
        if not target:
            result.evidence = "no file was named"
            result.escalation = "the caller must say which file to format"
            return result

        scratch = Path(symptom.scratch or cwd / ".sylphae-scratch")
        scratch.mkdir(parents=True, exist_ok=True)
        target_path = Path(target)
        if not target_path.is_absolute():
            target_path = cwd / target_path
        if not target_path.exists():
            result.evidence = f"no such file: {target_path}"
            result.escalation = "the named file does not exist"
            return result

        #: Undo before rewrite.  The backup is a *copy*, not a move: the
        #: formatter has to see the original where it expects it, and a move would
        #: hand it a path that no longer exists.
        import shutil

        scratch.mkdir(parents=True, exist_ok=True)
        backup_path = scratch / target_path.name
        counter = 1
        while backup_path.exists():
            backup_path = scratch / f"{target_path.stem}.{counter}{target_path.suffix}"
            counter += 1
        shutil.copy2(str(target_path), str(backup_path))

        fix = ex.run(
            CommandSpec(
                program="ruff",
                args=("format", "--no-cache", str(target_path.relative_to(cwd).as_posix())),
                cwd=cwd,
                writable_roots=(cwd, scratch),
            )
        )
        result.commands.append(fix.command)
        result.attempts += 1
        if fix.refusal:
            result.refusal = fix.refusal
            result.evidence = f"the format command was refused: {fix.refusal.detail}"
            return result

        check = ex.run(
            CommandSpec(
                program="ruff",
                args=(
                    "check",
                    "--no-cache",
                    str(target_path.relative_to(cwd).as_posix()),
                ),
                cwd=cwd,
                writable_roots=(cwd, scratch),
            )
        )
        result.commands.append(check.command)
        result.attempts += 1
        result.resolved = True
        result.action_taken = fix.command
        if check.ok:
            result.evidence = (
                f"formatted {target_path.relative_to(cwd).as_posix()}; "
                f"`ruff check` is now clean (backup at {backup_path})"
            )
        else:
            result.evidence = (
                f"formatted the file, but `ruff check` still reports violations; "
                f"these are not auto-fixable\n{check.stdout[:600]}"
            )
            result.escalation = (
                "remaining violations need an edit, not a formatter"
            )
        return result


# ------------------------------------------------- 8. ImportError / ModuleNotFound


class ImportErrorLocator(Playbook):
    """Point at the line that imported the missing module.

    A traceback names the missing module and a stack; this playbook reads the
    stack to find the *user's* frame, which is the one worth reporting.  Library
    frames are filtered out, because "the missing import is inside site-packages"
    is the same answer for every project.
    """

    category = "import-error-locator"

    _SITE_MARKERS = ("site-packages", "dist-packages", "lib/python", "venv", ".venv")

    def _run(self, symptom: Symptom, ex: EnvExecutor) -> PlaybookResult:
        text = symptom.text or ""
        cwd = Path(symptom.context.get("cwd") or symptom.cwd)
        result = PlaybookResult(
            category=self.category, resolved=False, evidence="", attempts=0
        )
        module = None
        match = _MODULE_NOT_FOUND_RE.search(text)
        if match:
            module = match.group(1)
        imp = _IMPORT_ERROR_RE.search(text)
        if imp and imp.group(1):
            module = f"{imp.group(2)}.{imp.group(1)}"

        frames = _TRACEBACK_FRAME_RE.findall(text)
        if not frames:
            result.evidence = "the traceback contains no file frames"
            result.escalation = "nothing to locate without a stack"
            return result

        user_frames = [
            (f, ln, fn)
            for f, ln, fn in frames
            if not any(marker in f.replace("\\", "/") for marker in self._SITE_MARKERS)
        ]
        pool = user_frames or frames
        frame_file, frame_line, frame_fn = pool[0]
        try:
            rel = Path(frame_file).relative_to(cwd).as_posix()
        except ValueError:
            rel = frame_file

        result.resolved = True
        result.action_taken = "parsed the traceback and located the importing frame"
        if module:
            result.evidence = (
                f"missing module {module!r} is imported at {rel}:{frame_line} in {frame_fn}()"
            )
        else:
            result.evidence = f"the failing frame is {rel}:{frame_line} in {frame_fn}()"

        #: The locator reports; it does not edit.  The fix for a missing import is
        #: either an install (category 4 handles that) or an edit, and this
        #: playbook's boundary is locating.
        result.escalation = (
            f"the import at {rel}:{frame_line} needs the module "
            f"{module or 'that is missing'} present, or the import corrected"
        )
        return result


#: Registered by category name.  ``select`` is what a caller uses, so the mapping
#: is the only place a new category has to be declared twice.
PLAYBOOKS: dict[str, type[Playbook]] = {
    "venv": VenvPlaybook,
    "pip-conflict": PipConflictPlaybook,
    "port-occupied": PortOccupancyPlaybook,
    "missing-dependency": MissingDependencyPlaybook,
    "affected-tests": AffectedTestPlaybook,
    "flake-retry": FlakeRetryPlaybook,
    "lint-fix": LintPlaybook,
    "import-error-locator": ImportErrorLocator,
}


def select(kind: str) -> Playbook | None:
    """The playbook for ``kind``, or ``None`` when no playbook covers it.

    ``None`` rather than a raise: the caller's job is to escalate to the model,
    and raising would make "no playbook" indistinguishable from "playbook broke".
    """
    cls = PLAYBOOKS.get(kind)
    return cls() if cls else None
