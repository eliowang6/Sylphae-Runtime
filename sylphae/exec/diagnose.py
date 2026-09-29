"""Failure classification: which failures are mechanically repairable in place?

The verification loop in DESIGN.md section 7.5 distinguishes two failure families, and the
distinction is what makes the autonomous loop safe:

============================  =================================================================
**mechanical**                 A defect whose repair needs no judgement about intent: a syntax error, an
                              indentation slip, a name that is misspelt, a symbol that is used but never
                              imported.  Repairing it cannot change the *meaning* of the edit, only
                              whether it parses and resolves.  These are the only failures the loop is
                              allowed to fix on its own.
**semantic**                   A defect of meaning: an assertion failed, a value came out wrong, a test
                              that ran cleanly and disagrees.  Repairing this requires deciding what the
                              edit was *supposed* to do, which is the planner LLM's job, not ours.  The
                              loop must escalate these with the evidence attached.
============================  =================================================================

The classifier is deliberately pessimistic.  ``auto_repairable`` is ``True`` only when the
failure is one the loop may attempt without consulting the planner; anything ambiguous is
classified semantic and escalated.  A wrong escalation costs the planner one read; a wrong
autonomous repair could silently change behaviour and report it as success -- the second
is far more expensive than the first.

Every classification records the evidence it was made from, because the whole point of the
escalation path is that the planner can audit *why* the loop gave up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

__all__ = ["FailureFamily", "FailureClass", "classify_failure"]


class FailureFamily(str, Enum):
    """The two families above, plus the plumbing failures that belong to neither."""

    MECHANICAL = "mechanical"
    SEMANTIC = "semantic"
    #: The verifier itself could not reach a verdict (timeout, no pytest, unreadable
    #: output).  Not a defect in the edit; the loop cannot act on it and must escalate so
    #: the planner knows nothing was verified.
    PLUMBING = "plumbing"


class FailureClass(str, Enum):
    """The specific failure kind.  All mechanical classes imply ``auto_repairable``."""

    # -- mechanical -------------------------------------------------------------------------
    #: ``compile()`` rejected the edited file.  Repairable: the loop can re-indent, close a
    #: bracket, or drop the malformed region -- though only the last is always safe, so the
    #: loop treats this as repairable-but-escalate-on-repeat.
    SYNTAX = "syntax_error"
    #: An ``IndentationError`` / ``TabError``, which Python reports as a ``SyntaxError``
    #: subclass but which has a uniquely mechanical repair (re-indent the region).
    INDENTATION = "indentation_error"
    #: ``NameError`` for a name the file does not define.  Repairable when the name is an
    #: obvious misspelling of something in scope, otherwise semantic -- a genuinely new
    #: symbol is a request the planner has to honour.
    UNDEFINED_NAME = "undefined_name"
    #: ``ImportError`` / ``ModuleNotFoundError`` for a symbol the edit uses.  Repairable when
    #: the module is present in the repo and merely un-imported.
    MISSING_IMPORT = "missing_import"
    #: Duplicate def/class name, redefinition of a symbol the edit introduced twice.
    DUPLICATE_DEFINITION = "duplicate_definition"
    # -- semantic ---------------------------------------------------------------------------
    #: A test ran and its assertion failed.  Never auto-repairable.
    ASSERTION = "assertion_failure"
    #: A test raised an exception that is not mechanical (e.g. a real ``ValueError`` from
    #: the code's own logic) -- it compiled and imported fine, then misbehaved.
    RUNTIME = "runtime_error"
    #: The edit applied and everything passes, but a test that used to pass no longer does,
    #: or a value the plan named was not produced.  Semantic by definition.
    REGRESSION = "regression"
    # -- plumbing ---------------------------------------------------------------------------
    VERIFIER = "verifier_failure"

    @property
    def family(self) -> FailureFamily:
        if self in _MECHANICAL:
            return FailureFamily.MECHANICAL
        if self is FailureClass.VERIFIER:
            return FailureFamily.PLUMBING
        return FailureFamily.SEMANTIC

    @property
    def auto_repairable(self) -> bool:
        return self in _AUTO_REPAIRABLE


_MECHANICAL = frozenset(
    {
        FailureClass.SYNTAX,
        FailureClass.INDENTATION,
        FailureClass.UNDEFINED_NAME,
        FailureClass.MISSING_IMPORT,
        FailureClass.DUPLICATE_DEFINITION,
    }
)
_AUTO_REPAIRABLE = frozenset(
    {
        # Syntax errors are only "mechanical" in the weak sense: the loop can reformat the
        # region, but it cannot know what the malformed code meant.  It may attempt one
        # repair and must escalate if that does not clear it -- see the loop's repeat guard.
        FailureClass.SYNTAX,
        FailureClass.INDENTATION,
        FailureClass.UNDEFINED_NAME,
        FailureClass.MISSING_IMPORT,
        FailureClass.DUPLICATE_DEFINITION,
    }
)


@dataclass
class ClassifiedFailure:
    """One classified failure, carrying everything the loop or the planner needs next.

    ``evidence`` is the exact substring of the verifier output the decision was made from.
    A caller who disagrees with the classification can check the evidence rather than
    trusting the label, which is the audit property the escalation path depends on.
    """

    failure_class: FailureClass
    family: FailureFamily
    message: str
    file: str = ""
    line: int | None = None
    evidence: str = ""
    #: The symbol or import the loop would need to touch, when the class names one.
    symbol: str = ""
    #: Names the classifier found in the repo that the failure might refer to.  Empty means
    #: "no candidate" -- the loop must not guess, and escalates instead.
    candidates: list[str] = field(default_factory=list)

    @property
    def auto_repairable(self) -> bool:
        return self.failure_class.auto_repairable

    def summary(self) -> str:
        loc = f"{self.file}:{self.line}" if self.line is not None else self.file
        return f"[{self.failure_class.value}] {loc} {self.message}".strip()


# --------------------------------------------------------------------------------------
# Regexes over pytest / compile output
# --------------------------------------------------------------------------------------

#: pytest ``--tb=short`` frames look like ``tests/test_x.py:12: in test_foo``.
_FRAME_RE = re.compile(r"^(?P<file>\S+?):(?P<line>\d+):\s*in\s+")

#: The exception line under ``--tb=short`` is ``E   ExceptionType: message``.
_EXC_LINE_RE = re.compile(r"^E\s+(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Warning))\s*:\s*(?P<msg>.*)$")

#: A bare ``E   NameError: name 'doble' is not defined`` -> the quoted identifier.
_NAME_ERROR_RE = re.compile(r"""name\s+['"](?P<name>[^'"]+)['"]\s+is\s+not\s+defined""")

#: ``E   ImportError: cannot import name 'foo' from 'pkg.mod'`` -> the quoted module.
_IMPORT_FROM_RE = re.compile(r"""from\s+['"](?P<module>[^'"]+)['"]""")

#: ``E   ModuleNotFoundError: No module named 'pkg'``.
_MODULE_ERROR_RE = re.compile(r"""No module named\s+['"](?P<module>[^'"]+)['"]""")

#: ``compile:`` errors from :mod:`sylphae.exec.verify` carry ``file:line: message``.
_COMPILE_ERROR_RE = re.compile(r"^compile:\s*(?P<file>\S+?):(?P<line>\d+):\s*(?P<msg>.*)$")


def _last_exception_block(output: str) -> tuple[str, str, list[tuple[str, int]]] | None:
    """Pull ``(exception_type, message, frames)`` out of a ``--tb=short`` tail.

    pytest prints the traceback frames in order and the ``E ...`` line last, so the final
    exception block is the one that describes the actual failure.  Returns ``None`` when no
    exception line is present -- the caller then has nothing mechanical to classify.
    """
    lines = (output or "").splitlines()
    exc_type = exc_msg = ""
    frames: list[tuple[str, int]] = []
    for line in lines:
        m = _FRAME_RE.match(line.strip())
        if m:
            frames.append((m.group("file"), int(m.group("line"))))
        m = _EXC_LINE_RE.match(line.strip())
        if m:
            exc_type = m.group("type")
            exc_msg = m.group("msg").strip()
    if not exc_type:
        return None
    return exc_type, exc_msg, frames


def _pick_location(frames: list[tuple[str, int]], fallback: str) -> tuple[str, int | None]:
    """Choose the frame that names a real source file.

    pytest frames include plugin internals (``_pytest/...``, ``pluggy/...``) whose line
    numbers describe pytest, not the edit.  The last frame inside the project is the one
    that failed; if there is none, the caller's fallback path stands.
    """
    for file, line in reversed(frames):
        if not re.match(r"^(?:_[a-z]|pluggy|site-packages|/usr/|.*\\lib\\)", file):
            return file, line
    if frames:
        return frames[-1]
    return fallback, None


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------


def classify_failure(
    errors: list[str],
    *,
    repo_symbols: set[str] | None = None,
    repo_modules: set[str] | None = None,
) -> ClassifiedFailure | None:
    """Classify one verifier failure from the ``errors`` list of a :class:`VerifyReport`.

    Returns ``None`` when there is nothing to classify (``errors`` is empty), which the loop
    treats as success, not as a plumbing failure.

    ``repo_symbols`` / ``repo_modules`` are the ground truth the classifier uses to decide
    whether a missing name is a typo (mechanical) or a genuinely new symbol (semantic).
    Both default to empty sets, in which case typo resolution is skipped and every
    undefined name escalates -- the safe default for a caller with no index.
    """
    if not errors:
        return None

    symbols = repo_symbols or set()
    modules = repo_modules or set()

    # A compile error is definitive: nothing ran, and the file is the story.
    for err in errors:
        if not err.startswith("compile:"):
            continue
        m = _COMPILE_ERROR_RE.match(err)
        if m:
            msg = m.group("msg")
            cls = (
                FailureClass.INDENTATION
                if "indentation" in msg.lower() or "tab" in msg.lower() or "expected an indented block" in msg
                else FailureClass.SYNTAX
            )
            return ClassifiedFailure(
                failure_class=cls,
                family=cls.family,
                message=msg,
                file=m.group("file"),
                line=int(m.group("line")),
                evidence=err,
            )
        # ``compile: file: <UnicodeDecodeError ...>`` -- malformed, not a syntax defect.
        return ClassifiedFailure(
            failure_class=FailureClass.SYNTAX,
            family=FailureFamily.MECHANICAL,
            message=err,
            evidence=err,
        )

    # The verifier could not reach a verdict.  Not a defect in the edit.
    for err in errors:
        if err.startswith(("pytest-unavailable:", "pytest-timeout:", "pytest-error:", "pytest-no-tests:", "no-tests:")):
            return ClassifiedFailure(
                failure_class=FailureClass.VERIFIER,
                family=FailureFamily.PLUMBING,
                message=err,
                evidence=err,
            )

    # Test failures: the traceback decides mechanical vs semantic.
    for err in errors:
        if not err.startswith("test-failures:"):
            continue
        block = _last_exception_block(err)
        if block is None:
            # Tests failed but pytest reported no exception line, which means an assertion
            # failed.  That is the definition of semantic.
            return ClassifiedFailure(
                failure_class=FailureClass.ASSERTION,
                family=FailureFamily.SEMANTIC,
                message="a test assertion failed",
                evidence=err,
            )
        exc_type, exc_msg, frames = block
        file, line = _pick_location(frames, "")

        # ImportError / ModuleNotFoundError: mechanical only when the module is ours.
        if "ImportError" in exc_type or "ModuleNotFoundError" in exc_type:
            mod_match = _MODULE_ERROR_RE.search(exc_msg)
            from_match = _IMPORT_FROM_RE.search(exc_msg)
            # ``cannot import name X from 'M'`` blames the *symbol* within a module that
            # does exist; ``No module named 'M'`` blames the module itself.
            target = from_match.group("module") if from_match else ""
            target = target or (mod_match.group("module") if mod_match else "")
            known = bool(target) and (target in modules or target.split(".")[0] in modules)
            if not target or known:
                return ClassifiedFailure(
                    failure_class=FailureClass.MISSING_IMPORT,
                    family=FailureFamily.MECHANICAL,
                    message=exc_msg,
                    file=file,
                    line=line,
                    evidence=err,
                    symbol=target,
                    candidates=[target] if target and known else [],
                )
            # An unknown module is a dependency request, not a repair.
            return ClassifiedFailure(
                failure_class=FailureClass.MISSING_IMPORT,
                family=FailureFamily.SEMANTIC,
                message=exc_msg,
                file=file,
                line=line,
                evidence=err,
                symbol=target,
            )

        # NameError: mechanical only when a near-spelling exists in the repo.
        if "NameError" in exc_type:
            name_match = _NAME_ERROR_RE.search(exc_msg)
            name = name_match.group("name") if name_match else ""
            near = _nearest_symbol(name, symbols) if name and symbols else ""
            if name and near:
                return ClassifiedFailure(
                    failure_class=FailureClass.UNDEFINED_NAME,
                    family=FailureFamily.MECHANICAL,
                    message=exc_msg,
                    file=file,
                    line=line,
                    evidence=err,
                    symbol=name,
                    candidates=[near],
                )
            return ClassifiedFailure(
                failure_class=FailureClass.UNDEFINED_NAME,
                family=FailureFamily.SEMANTIC,
                message=exc_msg,
                file=file,
                line=line,
                evidence=err,
                symbol=name,
            )

        # SyntaxError surfacing at import time: same treatment as the compile path above.
        if "SyntaxError" in exc_type:
            indented = "indentation" in exc_msg.lower() or "expected an indented block" in exc_msg
            cls = FailureClass.INDENTATION if indented else FailureClass.SYNTAX
            return ClassifiedFailure(
                failure_class=cls,
                family=FailureFamily.MECHANICAL,
                message=exc_msg,
                file=file,
                line=line,
                evidence=err,
            )

        # Duplicate definition -- two ops in one plan wrote the same symbol.
        if "cannot assign to function call" in exc_msg or "redefinition" in exc_msg.lower():
            return ClassifiedFailure(
                failure_class=FailureClass.DUPLICATE_DEFINITION,
                family=FailureFamily.MECHANICAL,
                message=exc_msg,
                file=file,
                line=line,
                evidence=err,
            )

        # Anything else that compiled and imported: the code's own behaviour is wrong.
        return ClassifiedFailure(
            failure_class=FailureClass.RUNTIME,
            family=FailureFamily.SEMANTIC,
            message=f"{exc_type}: {exc_msg}",
            file=file,
            line=line,
            evidence=err,
        )

    # Some other error string the classifier does not know.  Escalate rather than guess.
    return ClassifiedFailure(
        failure_class=FailureClass.RUNTIME,
        family=FailureFamily.SEMANTIC,
        message=errors[0],
        evidence=errors[0],
    )


# --------------------------------------------------------------------------------------
# Typo resolution
# --------------------------------------------------------------------------------------


def _edit_distance(a: str, b: str, cap: int = 3) -> int:
    """Levenshtein distance, bailing above ``cap`` where the pair is clearly unrelated."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def _nearest_symbol(name: str, symbols: set[str], max_distance: int = 2) -> str:
    """The closest repo symbol within ``max_distance`` edits, or ``""`` if none is close.

    The threshold is tight on purpose.  A distance of 1 covers the real cases (a dropped
    letter, a transposed pair); distance 2 admits ``resut`` -> ``result``.  Beyond that the
    "typo" is more likely a new name the edit wants to introduce, and silently rewriting it
    would change what the plan asked for.
    """
    if not name or not symbols:
        return ""
    # An exact match is not a typo of anything.  Without this, a symbol like ``doubled`` whose
    # name is distance 1 from another repo symbol (``double``) could be "corrected" into that
    # other symbol if the near name is reached first -- rewriting real code into a name the
    # author never wrote.  Distance 0 short-circuits before any near match is considered.
    if name in symbols:
        return name
    best, best_d = "", max_distance + 1
    for sym in symbols:
        if abs(len(sym) - len(name)) > max_distance:
            continue
        d = _edit_distance(name, sym, max_distance)
        if d < best_d:
            best, best_d = sym, d
            if d <= 1:
                return best
    return best if best_d <= max_distance else ""
