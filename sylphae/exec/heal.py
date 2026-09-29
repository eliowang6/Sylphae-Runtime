"""The autonomous verify-and-heal loop.

This is the reliability keystone from DESIGN.md section 7.5.  A step that does not verify is a
step that reports success on faith, which is how ``applied=True syntax_ok=True`` ended up on
files that would not build.  The loop replaces that with a cycle:

::

    edit -> compile -> run affected tests -> classify failure -> repair -> re-verify

and it stops at the first failure it cannot repair mechanically.  Two hard rules keep it safe:

1.  **Tests are the ground truth, and the loop never edits them.**  A test that fails is a
    statement about the edit, never an obstacle to be removed.  The loop repairs *source*,
    never a test file, and never a fixture.
2.  **Mechanical repairs only.**  An assertion failure goes back to the planner with its
    evidence attached.  The loop would rather escalate a hundred times than silently change
    behaviour once.

The loop is bounded twice over: ``max_attempts`` caps iterations, and the *repeat guard* stops
it the moment a repair fails to move the failure.  A repair that does not clear its named
defect is not a repair for that defect, so a second identical attempt is wasted compute that
could also thrash the file.  Each attempt restores the original source on failure, so a bounded
loop leaves the tree no worse than it found it.
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field
from pathlib import Path

from sylphae.exec.diagnose import ClassifiedFailure, classify_failure
from sylphae.exec.repair import RepairAttempt, attempt_repair
from sylphae.exec.textio import read_source, write_source
from sylphae.exec.verify import VerifyReport, affected_tests, verify

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "HealAttempt",
    "HealOutcome",
    "heal",
    "index_repo",
    "verify_and_heal",
]

DEFAULT_MAX_ATTEMPTS = 3

#: Repos above this many Python files are not symbol-indexed; the loop falls back to
#: escalating undefined names rather than walking a tree large enough for the index itself
#: to be the slow part.
_INDEX_FILE_CAP = 4000


@dataclass
class HealAttempt:
    """One pass of the loop, recorded so the outcome is auditable."""

    n: int
    failure_class: str
    repair: str
    cleared: bool
    #: What the verifier said after this attempt, empty when the attempt was declined.
    result: str = ""


@dataclass
class HealOutcome:
    """The loop's verdict, with the trail the planner needs when it escalates.

    ``repaired=True`` means the verifier observed a clean tree after the loop's own edit, not
    that the loop believes its repair was right.  ``attempts`` is the audit trail; when the
    loop escalates, ``attempts[-1]`` is the failure the planner inherits.
    """

    repaired: bool
    attempts: list[HealAttempt] = field(default_factory=list)
    #: The classification that caused the escalation, for the planner's error message.
    escalation: str = ""
    final_report: VerifyReport | None = None
    #: Number of verify subprocess invocations, for latency accounting.
    verifications: int = 0
    wall_ms: float = 0.0


# --------------------------------------------------------------------------------------
# Repo indexing -- the ground truth typo resolution needs
# --------------------------------------------------------------------------------------


def index_repo(root: Path, cap: int = _INDEX_FILE_CAP) -> tuple[set[str], set[str]]:
    """Collect top-level symbol names and dotted module paths under ``root``.

    This is what lets the classifier tell a *typo* (``doble`` -> ``double``) from a *new name*
    the edit intends to introduce.  Without it every undefined name escalates, which is the
    safe behaviour but wastes the planner's time on the most common real failure.

    Only top-level ``def``/``class`` names are collected: a nested name is not something an
    unqualified reference can reach anyway.  Files that do not parse are skipped -- an
    unparseable file is itself a failure the loop will report through the normal path.
    """
    symbols: set[str] = set()
    modules: set[str] = set()
    seen = 0
    try:
        for path in sorted(root.rglob("*.py")):
            if any(part.startswith(".") for part in path.parts):
                continue
            seen += 1
            if seen > cap:
                break
            rel = path.relative_to(root).with_suffix("")
            parts = rel.parts
            if parts and parts[-1] == "__init__":
                parts = parts[:-1]
            if parts:
                modules.add(".".join(parts))
            try:
                tree = ast.parse(read_source(path))
            except (SyntaxError, OSError, UnicodeDecodeError, UnicodeEncodeError, ValueError):
                # ``read_source`` keeps bytes that are not valid UTF-8 as surrogate escapes so
                # the byte survives a round trip, and ``ast.parse`` refuses those surrogates.
                # The index is best-effort: a file it cannot read is simply not indexed.
                continue
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    symbols.add(node.name)
    except OSError:
        pass
    return symbols, modules


# --------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------


def heal(
    root: Path,
    target_path: str,
    report: VerifyReport,
    *,
    test_paths: list[str] | None = None,
    repo_symbols: set[str] | None = None,
    repo_modules: set[str] | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> HealOutcome:
    """Repair the mechanical failures in ``target_path``, re-verifying after each attempt.

    ``report`` is the failing verification the loop inherits.  It is re-run after every repair
    rather than trusted, because a repair can fix one defect and reveal another -- and the
    second one might be semantic.

    Returns a :class:`HealOutcome` whose ``repaired`` is ``True`` only when the verifier
    observed a clean tree at the end.  The caller writes nothing; this function owns the file
    because it must be able to restore it.
    """
    t_start = time.perf_counter()
    tests = test_paths if test_paths is not None else affected_tests(root, [target_path], [])
    symbols = repo_symbols if repo_symbols is not None else set()
    modules = repo_modules if repo_modules is not None else set()

    outcome = HealOutcome(repaired=False, verifications=0)
    current = report
    last_signature = ""

    for n in range(1, max_attempts + 1):
        failure = classify_failure(
            current.errors, repo_symbols=symbols, repo_modules=modules
        )
        if failure is None:
            # Nothing to classify means nothing failed: a previous attempt cleared it.
            outcome.repaired = True
            outcome.final_report = current
            break

        signature = failure.summary()
        # Repeat guard: the same defect surviving a repair means the repair was not the right
        # one, and repeating it would churn the file without progress.
        if signature == last_signature:
            outcome.escalation = f"{failure.summary()} (repair did not clear the defect)"
            outcome.final_report = current
            break
        last_signature = signature

        if not failure.auto_repairable:
            outcome.escalation = f"{failure.summary()} -- {failure.family.value}, escalated to the planner"
            outcome.final_report = current
            break

        attempt = _attempt(root, target_path, failure, current, tests, symbols, modules, n)
        outcome.attempts.append(attempt.attempt)
        outcome.verifications = attempt.verifications
        if not attempt.attempt.cleared:
            outcome.escalation = failure.summary()
            outcome.final_report = attempt.report
            break
        current = attempt.report
        if current.clean:
            outcome.repaired = True
            outcome.final_report = current
            break

    outcome.wall_ms = (time.perf_counter() - t_start) * 1000.0
    return outcome


@dataclass
class _AttemptResult:
    attempt: HealAttempt
    report: VerifyReport
    verifications: int


def _attempt(
    root: Path,
    target_path: str,
    failure: ClassifiedFailure,
    current: VerifyReport,
    tests: list[str],
    symbols: set[str],
    modules: set[str],
    n: int,
) -> _AttemptResult:
    """Apply one repair, write it, and re-verify.  Restores the original when it fails.

    The original is read fresh on every attempt rather than cached across the loop, because a
    previous attempt that did not clear the failure left a modified file behind.  Starting
    from the current on-disk state keeps successive repairs from compounding.
    """
    resolved = root / target_path
    try:
        original = read_source(resolved)
    except OSError as exc:
        return _AttemptResult(
            attempt=HealAttempt(
                n=n,
                failure_class=failure.failure_class.value,
                repair=f"read failed: {exc}",
                cleared=False,
            ),
            report=VerifyReport(
                compiles=False, tests_run=0, tests_passed=0, errors=[f"read: {exc}"]
            ),
            verifications=0,
        )

    repair: RepairAttempt = attempt_repair(failure, original)
    if not repair.applied:
        # Nothing was written, so the report we came in with still describes the tree.  A
        # fabricated "compiles=True" here would turn a declined repair into a green step.
        return _AttemptResult(
            attempt=HealAttempt(
                n=n,
                failure_class=failure.failure_class.value,
                repair=f"declined: {repair.error}",
                cleared=False,
            ),
            report=current,
            verifications=0,
        )

    write_source(resolved, repair.new_source or "")
    rep = verify(root, [target_path], tests)
    if rep.clean:
        return _AttemptResult(
            attempt=HealAttempt(
                n=n,
                failure_class=failure.failure_class.value,
                repair=repair.description,
                cleared=True,
                result=f"clean: {rep.tests_run} tests",
            ),
            report=rep,
            verifications=1,
        )

    # The repair did not clear the failure, so restore the file.  Leaving a verified-bad edit
    # in place would make the loop's exit indistinguishable from a success it never earned.
    write_source(resolved, original)
    return _AttemptResult(
        attempt=HealAttempt(
            n=n,
            failure_class=failure.failure_class.value,
            repair=f"{repair.description} (reverted)",
            cleared=False,
            result="; ".join(rep.errors),
        ),
        report=rep,
        verifications=1,
    )


def verify_and_heal(
    root: Path,
    changed: list[str],
    all_tests: list[str],
    *,
    repo_symbols: set[str] | None = None,
    repo_modules: set[str] | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> tuple[VerifyReport, HealOutcome]:
    """Verify ``changed`` against their affected tests, then heal any mechanical failure.

    This is the entry point the executor should call in place of the bare ``verify`` that used
    to pass an empty test list.  Test selection goes through :func:`affected_tests` so the loop
    runs the tests the edit can actually touch rather than the whole suite.
    """
    tests = affected_tests(root, changed, all_tests)
    report = verify(root, changed, tests)
    if report.clean:
        return report, HealOutcome(
            repaired=True, verifications=1, final_report=report
        )
    outcome = heal(
        root,
        changed[0] if changed else "",
        report,
        test_paths=tests,
        repo_symbols=repo_symbols,
        repo_modules=repo_modules,
        max_attempts=max_attempts,
    )
    final = outcome.final_report or report
    return final, outcome
