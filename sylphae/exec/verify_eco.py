"""Cross-ecosystem verification: give a patch something that can falsify it.

``sylphae.exec.verify`` is pytest-only, and 12.35 measured why that limit was costing more
than it looked. On ``runs/iter34``, 14 of 25 scenarios had ``tests_run = None`` -- every
nextjs / vue / go / rust / vision scenario -- and were scored ``clean`` on syntax alone.
One of them, ``nextjs_btn_darken_enlarge``, was recorded as a pass carrying the single
character ``"'"`` for the task "darken and enlarge the create-project button". Syntax
parsing is the only gate a quote cannot pass, and it is not a gate about intent.

This module is the other side of that finding. It runs the test suite a repository
actually has, in whatever language that is, and reports outcomes in the same
``{test_id: status}`` shape ``pytest_outcomes`` returns, so a before/after diff is
language-agnostic and the existing ``diff_outcomes`` regression logic applies unchanged.

**Why the fixtures are small.** MEASURED 2026-10-02: ``work/real_projects/`` holds 17 files
across five directories, not vendored projects --

    go_authservice     4 files   go.mod, config/settings.go, handlers/auth.go
    nextjs_dashboard   6 files   package.json, 2 components, globals.css, cn.ts
    rust_taskrunner    2 files   Cargo.toml, src/config.rs
    vue_storefront     3 files   package.json, ProductCard.vue

and none of them ships a test. ``rust_taskrunner`` has no ``main.rs`` or ``lib.rs``, so
``cargo test`` cannot even link. So the gap is not "install the dependencies" -- there is
nothing to install and nothing to run. The gap is that these are hand-written samples with
no assertions, and a sample with no assertions cannot tell a correct patch from a wrong one.

**What this module does NOT do.** It does not generate tests. A verifier that writes the
test it then uses to grade the patch is grading itself: the patch and its own reward would
come from the same author, and a wrong patch that also breaks a wrong test would pass. Test
*suites* are authored deliberately and reviewed (``tests/fixtures/eco`` holds them, see
``DESIGN.md`` section 6.4); this module only runs them and parses the result faithfully.

**The contract is the same as ``verify``'s.** ``tests_run == 0`` implies ``errors`` is
non-empty, so "nothing failed" and "I could not ask" stay distinguishable. Every reason
string carries a stable prefix:

=============================  ========================================================
``eco-no-runner:``             no test command is declared for this ecosystem
``eco-toolchain-missing:``     the declared runner is not on PATH (node/go/cargo/...)
``eco-timeout:``               the runner exceeded ``timeout_s`` and was killed
``eco-build-error:``            the project does not compile/link, so no test can run
``eco-no-tests:``              the runner succeeded but collected/executed nothing
``eco-parse-error:``           the runner's output was not in a shape we can read
``eco-test-failures:``          tests ran and some failed
=============================  ========================================================

``eco-build-error`` is listed separately from ``eco-test-failures`` on purpose. A Go
package that does not compile has not "failed its tests"; the question was never asked,
and a runner that reports that as 0/0 is indistinguishable from a project with no tests.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Ecosystem",
    "EcoVerifyReport",
    "declare_runner",
    "discover_runner",
    "run_eco_tests",
    "eco_outcomes",
]


#: Characters of runner output echoed into ``EcoVerifyReport.errors``.
_TAIL_CHARS = 700


@dataclass(frozen=True)
class Ecosystem:
    """One language's way of running its own tests.

    ``argv`` is the command; ``cwd_rel`` is where it runs relative to the repository root.
    ``build`` is a separate command that must succeed *before* the test command, because
    for compiled languages a type error means the tests never ran -- and reporting that as
    "0 failed" is the exact conflation 12.35 documented.
    """

    name: str
    #: Test directories/files, relative to root. Empty means "run the whole project".
    test_globs: tuple[str, ...] = ()
    argv: tuple[str, ...] = ()
    build: tuple[str, ...] = ()
    cwd_rel: str = "."


# --- Outcome parsing -------------------------------------------------------------------
# Written per ecosystem rather than by a generic heuristic: `go test` prints `--- PASS`
# and `ok`, `cargo test` prints `test name ... ok`, and node's runners print TAP-ish or
# vitest's own shapes. A single regex across four formats would either miss cases or invent
# them, and an invented status is worse than an absent one -- it becomes a fabricated test
# result.

_GO_TEST_LINE = re.compile(r"^\s*--- (PASS|FAIL|SKIP):\s+(\S+)")
_GO_PKG_OK = re.compile(r"^ok\s+(\S+)")
_CARGO_LINE = re.compile(r"^test\s+(\S+)\s+\.\.\.\s+(ok|FAILED|ignored)")
_NODE_TAP_OK = re.compile(r"^(not ok|ok)\s+(\d+)(?:\s+-\s+(.*))?$")

# Outcome vocabulary, shared with sylphae.exec.verify.
#
# MEASURED, and this is the second time a mismatch between these two modules cost a whole
# ecosystem its numbers. ``verify._GREEN`` is ``{"PASSED", "XFAIL"}`` -- pytest's uppercase
# vocabulary. These were lowercase, so every non-Python scenario reported
# ``tests_run=11, tests_passed=0, regressions=0``: 11 tests ran, none counted as passed, and
# none counted as regressed, which is a combination no real suite produces. The rate came
# out at 0.190 and looked like a finding about patch quality.
#
# A regression is computed as "green before, not green after", so a vocabulary that never
# intersects _GREEN silently reports zero regressions no matter how much the patch broke.
# The statuses are now spelled the same way, and ``test_the_two_verifiers_agree_on_status
# _vocabulary`` fails if they ever diverge again.
_PASS, _FAIL, _SKIP = "PASSED", "FAILED", "SKIPPED"


ECOSYSTEMS: dict[str, Ecosystem] = {
    "go": Ecosystem(
        name="go",
        test_globs=("**/*_test.go",),
        argv=("go", "test", "./...", "-count=1", "-v"),
    ),
    "rust": Ecosystem(
        name="rust",
        test_globs=("**/tests/**/*.rs", "**/*_test.rs", "**/tests.rs"),
        argv=("cargo", "test", "--", "--nocapture"),
    ),
    "node": Ecosystem(
        name="node",
        test_globs=("**/*.test.js", "**/*.test.cjs", "**/*.test.mjs",
                    "**/*.test.ts", "**/*.spec.js", "**/*.spec.ts"),
        # ``node --test`` rather than vitest, and the reason is measured rather than
        # preferred. The nextjs and vue fixtures have no lockfile, no node_modules and no
        # framework; the scenarios edit Tailwind class names and CSS custom properties, none
        # of which needs a bundler or a DOM. vitest would mean a devDependency that cannot be
        # installed offline, and the first casualty would be the ability to measure anything
        # -- 12.35's failure mode exactly: no runner, no ground truth, a score made of
        # syntax checks.
        #
        # ``--test-reporter=tap`` is the default shape and is what _NODE_TAP_OK reads. Node 22
        # (verified v22.19.0) ships the runner, so this needs no install and no network.
        argv=("node", "--test"),
    ),
}


def discover_runner(ecosystem: str, root: Path) -> tuple[Ecosystem | None, str | None]:
    """Return the ecosystem descriptor, or why it cannot be used here.

    Two failure modes are kept apart because they need different fixes: the runner is not
    installed (environment problem, fixable) versus no test files exist yet (fixture
    problem, not fixable by installing anything).
    """
    spec = ECOSYSTEMS.get(ecosystem)
    if spec is None:
        return None, f"eco-no-runner: no test command is declared for {ecosystem!r}"

    tool = spec.argv[0]
    if shutil.which(tool) is None:
        return None, (f"eco-toolchain-missing: {tool!r} is not on PATH; "
                      f"the {ecosystem} suites cannot be run on this machine")

    if spec.test_globs and not any(
            p for pat in spec.test_globs for p in root.glob(pat)):
        return None, (f"eco-no-tests: {ecosystem} declares a test command but no file "
                      f"matches {', '.join(spec.test_globs)}; there is nothing to run")
    return spec, None


@dataclass
class EcoVerifyReport:
    """Outcome of one :func:`run_eco_tests` call.

    Mirrors :class:`sylphae.exec.verify.VerifyReport` so callers branch the same way.
    The invariant ``tests_run == 0`` implies non-empty ``errors`` holds on every path.
    """

    ecosystem: str
    #: ``{test_id: "passed"|"failed"|"skipped"}``. Empty when nothing could be asked.
    outcomes: dict[str, str] = field(default_factory=dict)
    tests_run: int = 0
    tests_passed: int = 0
    #: Real reasons, set by the runner. The public ``errors`` is a property over this plus
    #: the invariant fallback; see it for why that is computed on read.
    _errors: list[str] = field(default_factory=list)
    #: Wall clock for the runner itself.
    duration_ms: float | None = None
    exit_code: int | None = None

    @property
    def errors(self) -> list[str]:
        """Reasons this run could not be trusted, computed on read rather than stored.

        This was a field with a ``__post_init__`` fallback, and MEASURED what that costs: on
        a run that reported ``tests_run=5, tests_passed=5`` the stored errors still contained
        ``eco-no-tests``, because ``__post_init__`` ran when ``tests_run`` was still 0 and
        the value was patched afterwards. The report said both "5 passed" and "no outcome was
        recorded", and a caller reading the errors concluded the run had produced nothing --
        which is how ten benchmark scenarios ended up with ``tests_passed=0`` on trees whose
        suites were green.

        The invariant still holds, but it is evaluated at the moment someone asks: a report
        with results carries only real reasons, and one without is required to explain
        itself. A fallback that can outlive the condition it was written for is a fallback
        that eventually lies.
        """
        if self._errors:
            return list(self._errors)
        if self.tests_run == 0:
            return [
                f"eco-no-tests: no outcome was recorded for {self.ecosystem!r}; a report "
                f"with no tests has to say whether the runner was missing, the suite was "
                f"absent, or the output was unreadable"
            ]
        return []

    @errors.setter
    def errors(self, value: list[str]) -> None:
        self._errors = list(value)

    @property
    def ground_truth(self) -> bool:
        """Whether this run produced evidence that could have falsified a patch.

        Same question as ``ScenarioResult.ground_truth``, asked of a runner instead of a
        scenario, so a caller cannot treat "the command exited 0" as verification.
        """
        return self.tests_run > 0

    @property
    def regressions(self) -> list[str] | None:
        """Tests that failed, or ``None`` when nothing was asked.

        ``None`` is not ``[]``. An absent suite and an empty one are different answers, and
        12.35's ``test_regressions is None`` branch is exactly where that distinction was
        being lost.
        """
        if not self.ground_truth:
            return None
        return sorted(t for t, s in self.outcomes.items() if s == _FAIL)


def _parse_ecosystem_output(text: str, eco: Ecosystem) -> dict[str, str]:
    """Turn runner output into ``{test_id: status}``.

    Returns only statuses read literally off a line. A line that matches nothing yields
    nothing, so an unrecognised format produces an empty dict and therefore
    ``tests_run == 0`` plus ``eco-parse-error`` -- a reported inability, not a guessed
    result.

    The three formats do not agree on shape, so each branch reads the groups it declared
    rather than assuming a common arity: ``go test -v`` prints both a per-test
    ``--- PASS: TestName`` line and a per-package ``ok package`` line, and only the first
    names an individual test. Counting the package line as a test would inflate the
    denominator with an entry that has no assertion behind it.
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if eco.name == "go":
            m = _GO_TEST_LINE.match(line)
            if m:
                verdict, ident = m.group(1), m.group(2)
                out[f"go::{ident}"] = {"PASS": _PASS, "FAIL": _FAIL,
                                       "SKIP": _SKIP}[verdict]
                continue
            m = _GO_PKG_OK.match(line)
            if m:
                # A package with no test files is reported by go as `? pkg [no test files]`,
                # which matches nothing here -- correctly, since it asserted nothing.
                out[f"go::pkg::{m.group(1)}"] = _PASS
        elif eco.name == "rust":
            m = _CARGO_LINE.match(line)
            if m:
                ident, verdict = m.group(1), m.group(2)
                out[f"rust::{ident}"] = {"ok": _PASS, "FAILED": _FAIL,
                                         "ignored": _SKIP}[verdict]
        elif eco.name == "node":
            m = _NODE_TAP_OK.match(line)
            if m:
                verdict, num, title = m.group(1), m.group(2), m.group(3)
                out[f"node::{num}:{title or ''}"] = (_FAIL if verdict == "not ok"
                                                      else _PASS)
    return out


def _tail(text: str, limit: int = _TAIL_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else "..." + text[-limit:]


def run_eco_tests(root: Path, ecosystem: str, timeout_s: float = 300.0,
                  env: dict[str, str] | None = None) -> EcoVerifyReport:
    """Run ``ecosystem``'s suite in ``root`` and report what it said.

    ``root`` must already be a disposable copy. This runs arbitrary project code, so it is
    never pointed at ``work/repos`` or ``work/real_projects`` by the benchmark -- see
    ``real_world_bench.run_scenario``, which clones before verifying.
    """
    eco, reason = discover_runner(ecosystem, root)
    if eco is None:
        return EcoVerifyReport(ecosystem=ecosystem, errors=[reason or "eco-no-runner:"])

    # A compiled language has to build before it can test. Reported separately so a type
    # error is never read as "the tests passed".
    if eco.build:
        proc = _spawn(list(eco.build), root, timeout_s, env)
        if proc.returncode != 0:
            return EcoVerifyReport(
                ecosystem=ecosystem, exit_code=proc.returncode,
                _errors=[f"eco-build-error: {' '.join(eco.build)} exited "
                         f"{proc.returncode}; no test could run. "
                         f"{_tail(proc.combined)}"])

    import time

    t0 = time.perf_counter()
    proc = _spawn(list(eco.argv), root, timeout_s, env)
    duration = (time.perf_counter() - t0) * 1000.0

    if proc.timed_out:
        return EcoVerifyReport(
            ecosystem=ecosystem, duration_ms=duration, _errors=[
                f"eco-timeout: {' '.join(eco.argv)} exceeded {timeout_s:.0f}s and was "
                f"killed. {_tail(proc.combined)}"])

    outcomes = _parse_ecosystem_output(proc.combined, eco)
    report = EcoVerifyReport(
        ecosystem=ecosystem, outcomes=outcomes, duration_ms=duration,
        exit_code=proc.returncode)

    if not outcomes:
        # Exit 0 with nothing recognised is the dangerous case: it reads as success. A
        # non-zero exit with no parseable lines is a build or collection failure, which is
        # a different answer from "no tests" and gets its own prefix.
        prefix = "eco-build-error" if proc.returncode != 0 else "eco-parse-error"
        detail = ("the runner exited non-zero and printed no recognisable test lines"
                  if proc.returncode != 0 else
                  "the runner exited 0 but printed no recognisable test lines; treating "
                  "this as 'could not read the result' rather than 'passed'")
        report.errors = [f"{prefix}: {detail}. {_tail(proc.combined)}"]
        return report

    report.tests_run = len(outcomes)
    report.tests_passed = sum(1 for s in outcomes.values() if s == _PASS)
    if report.regressions:
        report.errors = [f"eco-test-failures: {len(report.regressions)} of "
                         f"{report.tests_run} failed: {', '.join(report.regressions[:5])}"]
    return report


@dataclass
class _Proc:
    returncode: int
    combined: str
    timed_out: bool = False


def _spawn(argv: list[str], cwd: Path, timeout_s: float,
           env: dict[str, str] | None) -> _Proc:
    """Run a command, capturing both streams, never inheriting stdin.

    Both streams matter: ``go test`` writes its per-test lines to stdout and build errors
    to stderr, and reading only one of them turns a compile failure into "no tests".
    """
    child_env = dict(os.environ)
    # Colour codes would break the `--- PASS` regexes, and the runners' own non-interactive
    # switches keep a missing TTY from disabling their output.
    child_env.update({"NO_COLOR": "1", "TERM": "dumb", "CI": "1",
                      "GOFLAGS": "-mod=mod", "CARGO_TERM_COLOR": "never"})
    if env:
        child_env.update(env)
    try:
        proc = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout_s, stdin=subprocess.DEVNULL,
            env=child_env, shell=False)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return _Proc(returncode=-1, combined=f"{out}\n{err}", timed_out=True)
    except OSError as exc:
        return _Proc(returncode=-1, combined=f"could not launch {argv[0]!r}: {exc}")
    return _Proc(returncode=proc.returncode,
                 combined=f"{proc.stdout}\n{proc.stderr}")


def eco_outcomes(root: Path, ecosystem: str, timeout_s: float = 300.0,
                 ) -> dict[str, str] | None:
    """Convenience wrapper returning outcomes, or ``None`` when nothing could be asked."""
    report = run_eco_tests(root, ecosystem, timeout_s=timeout_s)
    return report.outcomes if report.ground_truth else None


def declare_runner(root: Path, ecosystem: str) -> dict:
    """Report what would happen, without running anything.

    Used by the benchmark to label a scenario ``unverifiable`` with a specific reason
    instead of the blanket ``no-baseline-target`` string, so a missing toolchain and a
    missing suite are distinguishable in the artifact.
    """
    eco, reason = discover_runner(ecosystem, root)
    if eco is None:
        return {"ecosystem": ecosystem, "runnable": False, "reason": reason,
                "argv": None}
    return {"ecosystem": ecosystem, "runnable": True, "reason": None,
            "argv": list(eco.argv),
            "test_globs": [p for p in root.glob("**/*")
                           if any(Path(p).match(g) for g in eco.test_globs)]}
