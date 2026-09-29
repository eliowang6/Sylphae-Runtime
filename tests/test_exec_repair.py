"""Tests for ``sylphae.exec.repair`` -- the mechanical repairs the verify loop may perform.

The property under test throughout: **a repair either clears the failure or changes nothing.**
Every test either verifies the repaired source compiles/imports as claimed, or verifies the
repair declined to act.  A repair that makes a different change than the one described is a
bug, not a repair -- so the tests check the description against what actually happened.

These tests never write to disk.  ``attempt_repair`` is a pure function of the failure and the
source; the filesystem round trip is the loop's business and is tested there.

Run from the repository root:

    python -m pytest tests/test_exec_repair.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sylphae.exec.diagnose import ClassifiedFailure, FailureClass, FailureFamily  # noqa: E402
from sylphae.exec.repair import attempt_repair, _after_leading_docstring  # noqa: E402


def _semantic(msg: str = "assertion") -> ClassifiedFailure:
    return ClassifiedFailure(
        failure_class=FailureClass.ASSERTION,
        family=FailureFamily.SEMANTIC,
        message=msg,
    )


def _missing_import(module: str) -> ClassifiedFailure:
    return ClassifiedFailure(
        failure_class=FailureClass.MISSING_IMPORT,
        family=FailureFamily.MECHANICAL,
        message=f"cannot import name from '{module}'",
        symbol=module,
    )


def _undefined(typo: str, candidates: list[str]) -> ClassifiedFailure:
    return ClassifiedFailure(
        failure_class=FailureClass.UNDEFINED_NAME,
        family=FailureFamily.MECHANICAL,
        message=f"name '{typo}' is not defined",
        symbol=typo,
        candidates=candidates,
    )


def _indent(line: int) -> ClassifiedFailure:
    return ClassifiedFailure(
        failure_class=FailureClass.INDENTATION,
        family=FailureFamily.MECHANICAL,
        message="expected an indented block",
        file="a.py",
        line=line,
    )


# --------------------------------------------------------------------------------------
# Dispatch: non-repairable failures are refused
# --------------------------------------------------------------------------------------


def test_semantic_failure_is_refused() -> None:
    r = attempt_repair(_semantic(), "x = 1\n")
    assert r.applied is False
    assert r.new_source is None
    assert r.error


def test_syntax_error_is_refused() -> None:
    # A general syntax error has no repair that is safe without knowing intent.
    f = ClassifiedFailure(
        failure_class=FailureClass.SYNTAX,
        family=FailureFamily.MECHANICAL,
        message="invalid syntax",
        file="a.py",
        line=3,
    )
    r = attempt_repair(f, "def f(\n")
    assert r.applied is False


def test_duplicate_definition_is_refused() -> None:
    f = ClassifiedFailure(
        failure_class=FailureClass.DUPLICATE_DEFINITION,
        family=FailureFamily.MECHANICAL,
        message="redefinition of 'f'",
    )
    r = attempt_repair(f, "def f(): pass\n\n\ndef f(): pass\n")
    assert r.applied is False


def test_repair_does_not_raise_on_unexpected_input() -> None:
    # A repair must never take down the loop it serves.
    f = _indent(99)
    r = attempt_repair(f, "only one line\n")
    assert r.applied is False
    assert r.error


# --------------------------------------------------------------------------------------
# add_import
# --------------------------------------------------------------------------------------


def test_add_import_after_existing_imports() -> None:
    src = "import json\n\n\ndef f():\n    return 1\n"
    r = attempt_repair(_missing_import("os"), src)
    assert r.applied is True
    assert r.new_source is not None
    # The new import joins the existing import block, not the middle of the file.
    assert r.new_source.splitlines()[1] == "import os"
    compile(r.new_source, "a.py", "exec")  # must still parse


def test_add_import_after_docstring() -> None:
    src = '"""Module docstring."""\n\n\ndef f():\n    pass\n'
    r = attempt_repair(_missing_import("os"), src)
    assert r.applied is True
    assert r.new_source is not None
    lines = r.new_source.splitlines()
    assert lines[0] == '"""Module docstring."""'
    assert lines[1] == "import os"
    assert r.new_source.startswith('"""Module docstring."""\nimport os\n')
    compile(r.new_source, "a.py", "exec")


def test_add_import_with_no_imports_and_no_docstring() -> None:
    r = attempt_repair(_missing_import("os"), "x = 1\n")
    assert r.applied is True
    assert r.new_source is not None
    assert r.new_source.splitlines()[0] == "import os"


def test_already_imported_is_refused() -> None:
    r = attempt_repair(_missing_import("os"), "import os\n\n\nx = 1\n")
    assert r.applied is False
    assert "already imported" in r.error


def test_add_import_from_form_is_detected_as_present() -> None:
    r = attempt_repair(_missing_import("os"), "from os import path\n\n\nx = 1\n")
    assert r.applied is False


def test_add_import_preserves_line_endings() -> None:
    src = "import json\r\n\r\n\r\nx = 1\r\n"
    r = attempt_repair(_missing_import("os"), src)
    assert r.applied is True
    assert r.new_source is not None
    assert r.new_source == "import json\r\nimport os\r\n\r\n\r\nx = 1\r\n"


def test_add_import_with_no_module_name_is_refused() -> None:
    r = attempt_repair(_missing_import(""), "x = 1\n")
    assert r.applied is False


@pytest.mark.parametrize("quote", ['"""', "'''"])
def test_multiline_docstring_is_skipped(quote: str) -> None:
    src = f'{quote}\nA long\ndocstring.\n{quote}\n\n\ndef f():\n    pass\n'
    r = attempt_repair(_missing_import("os"), src)
    assert r.applied is True
    assert r.new_source is not None
    # The import lands after the closing quote, not inside the docstring.
    assert "import os" not in r.new_source.splitlines()[2]


# --------------------------------------------------------------------------------------
# fix_typo
# --------------------------------------------------------------------------------------


def test_typo_is_replaced_as_whole_words() -> None:
    src = "def double(x):\n    return doble(x)\n"
    r = attempt_repair(_undefined("doble", ["double"]), src)
    assert r.applied is True
    assert r.new_source is not None
    assert "doble" not in r.new_source
    assert "return double(x)" in r.new_source


def test_typo_in_multiple_sites_is_all_replaced() -> None:
    src = "resut = 1\nprint(resut)\n"
    r = attempt_repair(_undefined("resut", ["result"]), src)
    assert r.applied is True
    assert r.new_source is not None
    assert "resut" not in r.new_source
    assert "2 site" in r.description


def test_typo_not_present_is_refused() -> None:
    r = attempt_repair(_undefined("doble", ["double"]), "def f():\n    return 1\n")
    assert r.applied is False
    assert "does not appear" in r.error


def test_typo_with_no_candidate_is_refused() -> None:
    r = attempt_repair(_undefined("doble", []), "doble\n")
    assert r.applied is False


def test_typo_equal_to_candidate_is_refused() -> None:
    r = attempt_repair(_undefined("double", ["double"]), "double\n")
    assert r.applied is False


def test_typo_is_not_replaced_as_a_substring() -> None:
    # 'doble' as a substring of a longer identifier must not be touched.
    src = "x = 'adoble_text'\n"
    r = attempt_repair(_undefined("doble", ["double"]), src)
    # The whole-token regex does not match inside a larger identifier.
    assert r.applied is False or "adoble_text" in (r.new_source or "")


# --------------------------------------------------------------------------------------
# reindent
# --------------------------------------------------------------------------------------


def test_indentation_is_repaired() -> None:
    src = "def f():\nreturn 1\n"
    r = attempt_repair(_indent(2), src)
    assert r.applied is True
    assert r.new_source is not None
    assert r.new_source == "def f():\n    return 1\n"
    compile(r.new_source, "a.py", "exec")


def test_indentation_detects_two_space_indent() -> None:
    src = "def f():\nreturn 1\n\n\ndef g():\n  return 2\n"
    r = attempt_repair(_indent(2), src)
    assert r.applied is True
    assert r.new_source is not None
    assert r.new_source.splitlines()[1] == "  return 1"


def test_indentation_of_blank_line_is_refused() -> None:
    r = attempt_repair(_indent(2), "def f():\n\n")
    assert r.applied is False


def test_indentation_past_end_of_file_is_refused() -> None:
    r = attempt_repair(_indent(5), "def f():\n    return 1\n")
    assert r.applied is False


def test_indentation_without_line_number_is_refused() -> None:
    f = ClassifiedFailure(
        failure_class=FailureClass.INDENTATION,
        family=FailureFamily.MECHANICAL,
        message="expected an indented block",
        file="a.py",
        line=None,
    )
    r = attempt_repair(f, "def f():\nreturn 1\n")
    assert r.applied is False


def test_indentation_when_already_indented_is_refused() -> None:
    # If the line is already at the target depth, the failure is about something else.
    r = attempt_repair(_indent(2), "def f():\n    return 1\n")
    assert r.applied is False


# --------------------------------------------------------------------------------------
# docstring helper
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body,expected",
    [("x = 1\n", 0), ('"""one"""\nx = 1\n', 1), ('"""\nmulti\n"""\nx = 1\n', 3)],
)
def test_after_leading_docstring(body: str, expected: int) -> None:
    """The index an import belongs at: 0 with no docstring, else after the closing quote."""
    lines = body.splitlines(keepends=True)
    assert _after_leading_docstring(lines) == expected


# --------------------------------------------------------------------------------------
# RepairAttempt is the loop's contract
# --------------------------------------------------------------------------------------


def test_refused_repair_has_an_explanation() -> None:
    r = attempt_repair(_semantic(), "x = 1\n")
    assert r.error is not None
    assert r.description == ""


def test_applied_repair_describes_itself() -> None:
    r = attempt_repair(_missing_import("os"), "x = 1\n")
    assert r.applied is True
    assert "import os" in r.description
