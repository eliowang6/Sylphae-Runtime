"""Verification: does the edit compile, and do the affected tests still pass?

Implements ``verify`` / ``affected_tests`` / ``VerifyReport`` from INTERFACES.md section 5.

The governing rule is **never claim success you did not observe**. There is one invariant
and it is enforced on every return path:

    ``tests_run == 0``  implies  ``errors`` is non-empty.

So a caller can always tell "0 tests failed" (``tests_run > 0``, no failure error) apart
from "I could not run the tests" (``tests_run == 0``, ``errors`` names the reason). Every
reason string carries a stable prefix so the caller can branch on it without substring
guessing:

======================  ==========================================================
``compile:``            a changed file failed ``compile()``
``no-tests:``           no test paths were supplied, so nothing was verified
``pytest-unavailable:`` pytest is not importable, or could not be launched
``pytest-timeout:``     pytest exceeded ``timeout_s`` and was killed
``pytest-no-tests:``    pytest ran but collected/executed nothing
``pytest-error:``       collection error, usage error, or unparseable output
``test-failures:``      tests ran and some failed
======================  ==========================================================
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path

from sylphae.repo.index import (
    ImportRef,
    collect_imports,
    iter_python_files,
    module_name_of,
    parse_source,
    resolve_import_candidates,
    tokenize_text,
)

__all__ = ["VerifyReport", "affected_tests", "verify"]

_ENCODING = "utf-8"
_ERRORS = "surrogateescape"

#: Characters of pytest output echoed into ``VerifyReport.errors``.
_TAIL_CHARS = 700


@dataclass
class VerifyReport:
    """Result of one :func:`verify` call.

    ``compiles`` is about ``compile()`` only; it says nothing about the tests.
    ``tests_run`` counts tests pytest actually executed (passed + failed), excluding
    skipped and collection errors. ``errors`` is the list of reasons this verification is
    not clean; it is non-empty whenever ``tests_run == 0``.
    """

    compiles: bool
    tests_run: int
    tests_passed: int
    errors: list[str] = field(default_factory=list)

    @property
    def observed(self) -> bool:
        """True when at least one test actually ran. Derived, not part of the contract."""
        return self.tests_run > 0

    @property
    def clean(self) -> bool:
        """True only when the code compiled *and* tests ran *and* none failed."""
        return self.compiles and self.tests_run > 0 and not self.errors


# --------------------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------------------


def _to_rel(root: Path, path: str | os.PathLike) -> str:
    """Normalise a path to a repo-relative posix string."""
    text = str(path).replace("\\", "/")
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            text = candidate.resolve().relative_to(Path(root).resolve()).as_posix()
        except (ValueError, OSError):
            return text
    while text.startswith("./"):
        text = text[2:]
    return text


def _tail(text: str, limit: int = _TAIL_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return "... " + text[-limit:]


# --------------------------------------------------------------------------------------
# Compile stage
# --------------------------------------------------------------------------------------


def _compile_changed(root: Path, changed_paths: list[str]) -> tuple[bool, list[str]]:
    """Validate syntax of changed files. Cheap enough to always run first."""
    errors: list[str] = []
    for rel in changed_paths:
        target = Path(root) / rel
        try:
            raw = target.read_bytes()
        except OSError as exc:
            errors.append(f"compile: cannot read {rel}: {exc}")
            continue

        # ``errors="surrogateescape"`` represents an undecodable byte as a lone surrogate, and
        # ``compile`` refuses a ``str`` that contains one.  That made a file which is valid
        # Python under its own coding declaration report a *syntax* error, naming the wrong
        # subsystem.  ``compile`` accepts ``bytes`` and honours PEP 263 itself, so the bytes
        # are what get compiled and the decoded string is only for the text-based checks.
        # MEASURED 2026-09-23: a latin-1 0xe9 in a comment produced
        # "compile: notes.py: UnicodeEncodeError: ... surrogates not allowed".
        source = raw.decode(_ENCODING, errors=_ERRORS)

        ext = Path(rel).suffix.lower()
        if ext in (".py", ".pyw"):
            try:
                compile(raw, rel, "exec")
            except SyntaxError as exc:
                errors.append(f"compile: {rel}:{exc.lineno}: {exc.msg}")
            except ValueError as exc:  # e.g. source containing null bytes
                errors.append(f"compile: {rel}: {type(exc).__name__}: {exc}")
        elif ext == ".json":
            import json
            try:
                json.loads(source)
            except Exception as exc:
                errors.append(f"compile: {rel}: {exc}")
        elif ext in (".css", ".scss", ".ts", ".tsx", ".js", ".jsx", ".vue", ".html"):
            if source.count("{") != source.count("}"):
                errors.append(f"compile: {rel}: unbalanced braces {{ and }}")
    return (not errors), errors


# --------------------------------------------------------------------------------------
# Pytest stage
# --------------------------------------------------------------------------------------

_COUNT_RE = re.compile(
    r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed|deselected|warning|warnings)"
)
_DURATION_RE = re.compile(r"\bin\s+\d+(?:\.\d+)?s\b")


def _parse_pytest_counts(output: str) -> dict[str, int] | None:
    """Extract the terminal summary counts from pytest's output.

    Scans backwards for the summary line, which is the only line that carries a duration
    and the count words together. Returns ``{}`` for an explicit "no tests ran", and
    ``None`` when no summary line could be found at all -- the caller must treat ``None``
    as "unknown", never as zero failures.
    """
    lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if _DURATION_RE.search(line):
            found = _COUNT_RE.findall(line)
            if found:
                counts: dict[str, int] = {}
                for number, kind in found:
                    key = "error" if kind.startswith("error") else kind.rstrip("s")
                    counts[key] = counts.get(key, 0) + int(number)
                return counts
            if "no tests ran" in line:
                return {}
    for line in reversed(lines):
        if "no tests ran" in line:
            return {}
    return None


def _run_pytest(
    root: Path, test_paths: list[str], timeout_s: int
) -> tuple[int, int, list[str]]:
    """Run pytest on ``test_paths``. Returns ``(tests_run, tests_passed, errors)``.

    On every failure path ``tests_run`` is 0 and ``errors`` explains why.
    """
    if find_spec("pytest") is None:
        return 0, 0, [f"pytest-unavailable: pytest is not importable by {sys.executable}"]

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        "-p",
        "no:cacheprovider",
        *test_paths,
    ]
    env = dict(os.environ)
    # Bytecode caches and ambient pytest state would both be side effects of *our*
    # invocation leaking into the caller's tree / shell -- and `verify` is itself called
    # from inside pytest by our own tests, so a stale PYTEST_CURRENT_TEST would otherwise
    # be inherited by the child run. Drop the ambient ones; keep the repository's own
    # pytest.ini / pyproject configuration, which is genuine project config and whose
    # failure we want to see rather than hide.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_CURRENT_TEST", None)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            encoding=_ENCODING,
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return 0, 0, [
            f"pytest-timeout: pytest did not finish within {timeout_s}s; "
            f"killed without result, nothing was verified"
        ]
    except OSError as exc:
        return 0, 0, [f"pytest-unavailable: could not launch pytest: {exc}"]

    output = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    counts = _parse_pytest_counts(output)

    if counts is None:
        return 0, 0, [
            f"pytest-error: could not read a pytest summary line "
            f"(exit {proc.returncode}); nothing was verified: {_tail(output)}"
        ]

    passed = counts.get("passed", 0)
    failed = counts.get("failed", 0)
    errored = counts.get("error", 0)
    tests_run = passed + failed
    errors: list[str] = []

    if failed:
        errors.append(
            f"test-failures: {failed} of {tests_run} executed tests failed: {_tail(output)}"
        )
    if errored:
        errors.append(f"pytest-error: {errored} collection/usage error(s): {_tail(output)}")
    if tests_run == 0 and not failed and not errored:
        errors.append(
            f"pytest-no-tests: pytest executed no tests "
            f"(exit {proc.returncode}, counts {counts or '{}'}): {_tail(output)}"
        )
    return tests_run, passed, errors


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def verify(
    root: Path,
    changed_paths: list[str],
    test_paths: list[str],
    timeout_s: int = 120,
) -> VerifyReport:
    """Compile the changed files, then run pytest over ``test_paths``.

    ``compiles`` reflects the in-process ``compile()`` of ``changed_paths`` only.
    ``tests_run == 0`` always comes with a non-empty ``errors``; see the module docstring.
    """
    root = Path(root)
    changed = [_to_rel(root, p) for p in (changed_paths or [])]
    tests = [_to_rel(root, p) for p in (test_paths or [])]

    compiles, errors = _compile_changed(root, changed)

    if not tests:
        errors.append("no-tests: no test paths were supplied, so nothing was verified")
        return VerifyReport(compiles=compiles, tests_run=0, tests_passed=0, errors=errors)

    tests_run, tests_passed, test_errors = _run_pytest(root, tests, timeout_s)
    errors.extend(test_errors)

    report = VerifyReport(
        compiles=compiles, tests_run=tests_run, tests_passed=tests_passed, errors=errors
    )
    # The invariant, asserted rather than assumed: a zero-run result must always say why.
    assert report.tests_run > 0 or report.errors, "verify() returned an unexplained zero run"
    return report


def affected_tests(root: Path, changed_paths: list[str], all_tests: list[str]) -> list[str]:
    """Test files that exercise the changed source files.

    Three tiers, in order. The first that yields anything wins, and the last tier is
    *every* test -- because returning an empty list would silently mean "verified", which
    is the one answer this function is not allowed to give when tests exist.

    1. **Import graph.** Files that transitively import the changed file, intersected with
       ``all_tests``. This is the precise answer.
    2. **Path-token overlap.** ``src/pkg/serializer.py`` -> ``tests/test_serializer.py``.
       Covers the common layout where the test never imports the module by name.
    3. **Everything.** Undetermined mapping, so verify broadly rather than nothing.

    A changed path that is itself a test file maps to itself.
    """
    root = Path(root)
    tests = [_to_rel(root, t) for t in (all_tests or [])]
    if not tests:
        return []
    changed = [_to_rel(root, c) for c in (changed_paths or [])]
    if not changed:
        return []

    tests_set = set(tests)
    order = {path: idx for idx, path in enumerate(tests)}

    # -- module map + import edges over the whole tree ---------------------------------
    module_suffix: dict[str, list[str]] = {}
    imports_of: dict[str, list[ImportRef]] = {}
    for rel in iter_python_files(root):
        mod = module_name_of(rel)
        if mod:
            parts = mod.split(".")
            for cut in range(len(parts)):
                module_suffix.setdefault(".".join(parts[cut:]), []).append(rel)
        try:
            source = (root / rel).read_text(encoding=_ENCODING, errors=_ERRORS)
            tree = parse_source(source, rel)
        except (OSError, SyntaxError, ValueError):
            imports_of[rel] = []
            continue
        imports_of[rel] = collect_imports(tree)

    reverse: dict[str, set[str]] = {}
    for rel, refs in imports_of.items():
        for ref in refs:
            for target in resolve_import_candidates(rel, ref):
                for importer_target in module_suffix.get(target, ()):
                    if importer_target != rel:
                        reverse.setdefault(importer_target, set()).add(rel)

    def importers_closure(start: str) -> set[str]:
        seen: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            for importer in reverse.get(node, ()):
                if importer not in seen:
                    seen.add(importer)
                    stack.append(importer)
        return seen

    # -- resolve ------------------------------------------------------------------------
    test_tokens = {t: set(tokenize_text(t)) for t in tests}
    # Tokens shared by *every* test path ("py", "test", "tests") carry no discriminating
    # power. Without removing them the overlap test below would match everything and tier
    # 2 would silently degenerate into tier 3.
    universal = set.intersection(*test_tokens.values()) if test_tokens else set()

    resolved: set[str] = set()
    for path in changed:
        if path in tests_set:
            resolved.add(path)
            continue
        hits = {t for t in tests if t in importers_closure(path)}
        if hits:
            resolved |= hits
            continue
        changed_tokens = set(tokenize_text(path)) - universal
        overlap = {t for t in tests if changed_tokens & test_tokens[t]} if changed_tokens else set()
        if overlap:
            resolved |= overlap
            continue
        resolved |= tests_set

    return sorted(resolved, key=lambda t: order.get(t, len(tests)))
