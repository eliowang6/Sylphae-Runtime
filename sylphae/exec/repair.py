"""Mechanical repairs the autonomous verify loop may perform on its own.

A repair is the *only* kind of edit the loop is allowed to make without asking the planner,
and it earns that privilege by satisfying two properties:

1.  **It cannot change meaning.**  Adding an import that the module already needs, correcting
    a misspelling of a name that exists, or indenting a line Python says should be indented do
    not alter what the code computes.  A repair that *could* change meaning is not a repair --
    it is an edit, and edits belong to the planner.
2.  **It is checkable.**  Every repair is followed by a re-run of the verifier.  A repair that
    does not clear the failure is undone and the failure escalates, so a bad repair can cost
    time but cannot ship silently wrong code.

The guards are strict on purpose.  When the expected pattern is not found exactly where the
diagnosis says it is, the repair returns ``applied=False`` rather than searching hopefully --
a miss here means an escalation, which is the safe direction.

Undo is the caller's responsibility: :func:`attempt_repair` returns the candidate source, and
the loop keeps the original so it can restore it if verification does not improve.  Repairs
never touch the filesystem; only :func:`write_source` does, and only the caller calls it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sylphae.exec.diagnose import ClassifiedFailure, FailureClass
from sylphae.exec.textio import line_ending, with_ending

__all__ = ["RepairAttempt", "attempt_repair"]


@dataclass
class RepairAttempt:
    """Outcome of one repair attempt on a single file.

    ``applied=False`` with ``error`` set means the repair recognised the failure but declined
    to act -- the pattern was not where the diagnosis said, or the repair would have touched
    more than one site.  The loop escalates in that case, same as if the failure were
    semantic.
    """

    applied: bool
    new_source: str | None = None
    description: str = ""
    error: str | None = None


# --------------------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------------------


def attempt_repair(failure: ClassifiedFailure, source: str) -> RepairAttempt:
    """Attempt the mechanical repair the failure calls for, on ``source``.

    Returns ``applied=False`` for any failure that is not auto-repairable, and for a
    repairable one whose pattern is not found.  Never raises on an unexpected failure shape:
    an exception here would take down the loop for a problem the loop was meant to report.
    """
    if not failure.auto_repairable:
        return RepairAttempt(applied=False, error="failure is not auto-repairable")

    try:
        if failure.failure_class is FailureClass.MISSING_IMPORT:
            return _add_import(failure, source)
        if failure.failure_class is FailureClass.UNDEFINED_NAME:
            return _fix_typo(failure, source)
        if failure.failure_class is FailureClass.INDENTATION:
            return _reindent(failure, source)
        if failure.failure_class is FailureClass.SYNTAX:
            return RepairAttempt(applied=False, error="syntax errors have no safe general repair")
        if failure.failure_class is FailureClass.DUPLICATE_DEFINITION:
            return RepairAttempt(applied=False, error="duplicate definitions need the planner to choose")
    except Exception as exc:  # noqa: BLE001 - a repair must never kill the loop
        return RepairAttempt(applied=False, error=f"repair raised {type(exc).__name__}: {exc}")

    return RepairAttempt(applied=False, error="no repair for this failure class")


# --------------------------------------------------------------------------------------
# Missing import
# --------------------------------------------------------------------------------------


_IMPORT_LINE_RE = re.compile(r"^(?:from\s+(\S+)\s+import\s+.*|import\s+(.+))$", re.MULTILINE)


def _add_import(failure: ClassifiedFailure, source: str) -> RepairAttempt:
    """Add ``import <module>`` for a module the classifier confirmed is in the repo.

    Placed after the last existing top-level import, or after the docstring when the file has
    no imports yet.  A module that is already imported is skipped -- the failure must be
    something else, and adding a duplicate would not help.
    """
    module = failure.symbol.strip()
    if not module:
        return RepairAttempt(applied=False, error="no module name in the failure")

    end = line_ending(source)
    for existing in _IMPORT_LINE_RE.finditer(source):
        imported = existing.group(1) or existing.group(2)
        if imported and imported.split(" as ")[0].strip() == module:
            return RepairAttempt(applied=False, error=f"{module} is already imported")

    statement = f"import {module}"
    lines = source.splitlines(keepends=True)

    # ``_last_import_line`` returns the index *of* the last top-level import, so the new one
    # goes after it; ``_after_leading_docstring`` returns the index *after* the docstring, so
    # the import goes there directly.  With neither, both are 0 and the import opens the file.
    last_import = _last_import_line(lines)
    insert_at = (last_import + 1) if last_import is not None else _after_leading_docstring(lines)

    new_line = with_ending(statement + "\n", end)
    candidate = "".join(lines[:insert_at]) + new_line + "".join(lines[insert_at:])

    if candidate == source:
        return RepairAttempt(applied=False, error="insertion produced no change")

    return RepairAttempt(
        applied=True,
        new_source=candidate,
        description=f"added '{statement}'",
    )


def _last_import_line(lines: list[str]) -> int | None:
    """Index of the last top-level ``import``/``from ... import`` line, or ``None``."""
    last = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # A top-level import starts at column 0.
        if line.startswith(("import ", "from ")):
            last = i
            continue
        # Anything else at column 0 means the import block has ended for good.
        if not line[0].isspace():
            break
    return last


def _after_leading_docstring(lines: list[str]) -> int:
    """Index where an import belongs when the file has a leading docstring and no imports.

    Returns the index *after* the docstring's closing quote, so a one-line docstring yields
    ``1`` and a multi-line one yields the line after the quote.  ``0`` when there is no
    docstring, which puts the import at the top of the file.
    """
    if not lines:
        return 0
    first = lines[0].lstrip()
    for quote in ('"""', "'''"):
        if not first.startswith(quote):
            continue
        # One-line docstring: the closing quote is on the same line, so index 1.
        if len(first) > 3 and quote in first[3:]:
            return min(1, len(lines) - 1)
        # Multi-line: the import goes after the line carrying the closing quote.
        for j in range(1, len(lines)):
            if quote in lines[j]:
                return min(j + 1, len(lines) - 1)
        return 0
    return 0


# --------------------------------------------------------------------------------------
# Misspelt name
# --------------------------------------------------------------------------------------


def _fix_typo(failure: ClassifiedFailure, source: str) -> RepairAttempt:
    """Replace the misspelt name with the close repo symbol the classifier found.

    The classifier already limited the distance to two edits and confirmed the candidate
    exists, so this is the mechanical part of that decision: swap the token, whole words only,
    and refuse if the name does not appear as a standalone token (a substring hit would be a
    different repair entirely).
    """
    typo = failure.symbol
    if not typo or not failure.candidates:
        return RepairAttempt(applied=False, error="no candidate symbol for the undefined name")
    fix = failure.candidates[0]
    if typo == fix:
        return RepairAttempt(applied=False, error="candidate equals the misspelt name")

    pattern = re.compile(rf"\b{re.escape(typo)}\b")
    hits = pattern.findall(source)
    if not hits:
        return RepairAttempt(applied=False, error=f"'{typo}' does not appear as a whole token")

    candidate = pattern.sub(fix, source)
    if candidate == source:
        return RepairAttempt(applied=False, error="substitution produced no change")

    return RepairAttempt(
        applied=True,
        new_source=candidate,
        description=f"renamed '{typo}' -> '{fix}' ({len(hits)} site{'s' if len(hits) != 1 else ''})",
    )


# --------------------------------------------------------------------------------------
# Indentation
# --------------------------------------------------------------------------------------


def _reindent(failure: ClassifiedFailure, source: str) -> RepairAttempt:
    """Indent the line Python says should be indented.

    The failure message is ``expected an indented block after '<stmt>' on line N``, and the
    reported ``line`` is the one that should have been indented.  The repair adds one indent
    unit, matching the file's own indentation width, and refuses if that would nest the line
    more than one level below the statement -- a sign the diagnosis is about something else.
    """
    if failure.line is None or failure.line < 1:
        return RepairAttempt(applied=False, error="no line number for the indentation failure")

    lines = source.splitlines(keepends=True)
    idx = failure.line - 1
    if idx >= len(lines):
        return RepairAttempt(applied=False, error="failure line is past the end of the file")

    unit = _indent_unit(source)
    target = lines[idx]
    if not target.strip():
        return RepairAttempt(applied=False, error="the offending line is blank")

    # The statement the error names sits on the previous non-blank line.
    prev_idx = idx - 1
    while prev_idx >= 0 and not lines[prev_idx].strip():
        prev_idx -= 1
    if prev_idx < 0:
        return RepairAttempt(applied=False, error="no statement before the offending line")

    prev_indent = len(lines[prev_idx]) - len(lines[prev_idx].lstrip(" "))
    new_indent = prev_indent + len(unit)
    if len(target) - len(target.lstrip(" ")) >= new_indent:
        return RepairAttempt(
            applied=False,
            error="the line is already indented at least as far as the repair would go",
        )

    # Rebuild the line with its original trailing newline, so the file's line structure is
    # preserved and CRLF stays CRLF.
    tail = target[len(target.rstrip("\r\n")) :]
    body = target[: len(target) - len(tail)]
    candidate = "".join(lines[:idx]) + f"{' ' * new_indent}{body}{tail}" + "".join(lines[idx + 1 :])
    if candidate == source:
        return RepairAttempt(applied=False, error="indentation produced no change")

    return RepairAttempt(
        applied=True,
        new_source=candidate,
        description=f"indented line {failure.line} by {len(unit)} spaces",
    )


def _indent_unit(source: str, default: str = "    ") -> str:
    """The file's indentation width, from a real indented line, with ``default`` as fallback."""
    widths: set[int] = set()
    for line in source.splitlines():
        stripped = line.lstrip(" ")
        if stripped and line != stripped:
            widths.add(len(line) - len(stripped))
    if not widths:
        return default
    return " " * min(widths)
