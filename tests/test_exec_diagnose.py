"""Tests for ``sylphae.exec.diagnose`` -- the mechanical / semantic failure classifier.

Every test feeds a real-shaped verifier error string, because the classifier's entire job is
to read what pytest and ``compile()`` actually emit.  Strings here are copied from real runs
of ``python -m pytest --tb=short`` against deliberately broken files.

The invariant under test throughout: **the classifier never claims mechanical when it is not
sure.**  Anything ambiguous lands on semantic + escalate, which costs the planner one read;
a false mechanical verdict could silently change behaviour and call it success.

Run from the repository root:

    python -m pytest tests/test_exec_diagnose.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sylphae.exec.diagnose import (  # noqa: E402
    ClassifiedFailure,
    FailureClass,
    FailureFamily,
    _edit_distance,
    _nearest_symbol,
    classify_failure,
)


def _classify(errors: list[str], **kw) -> ClassifiedFailure | None:
    return classify_failure(errors, **kw)


# --------------------------------------------------------------------------------------
# compile: errors are mechanical, and indentation is distinguished from plain syntax
# --------------------------------------------------------------------------------------


def test_empty_errors_is_not_a_failure() -> None:
    assert _classify([]) is None


def test_compile_syntax_error_is_mechanical() -> None:
    err = "compile: mypkg/core.py:4: invalid syntax"
    f = _classify([err])
    assert f is not None
    assert f.failure_class is FailureClass.SYNTAX
    assert f.family is FailureFamily.MECHANICAL
    assert f.auto_repairable is True
    assert f.file == "mypkg/core.py"
    assert f.line == 4


def test_compile_indentation_error_is_classified_as_indentation() -> None:
    err = "compile: mypkg/core.py:9: expected an indented block after 'def f()' on line 8"
    f = _classify([err])
    assert f is not None
    assert f.failure_class is FailureClass.INDENTATION
    assert f.family is FailureFamily.MECHANICAL


def test_compile_tab_error_is_indentation() -> None:
    f = _classify(["compile: a.py:3: inconsistent use of tabs and spaces in indentation"])
    assert f is not None
    assert f.failure_class is FailureClass.INDENTATION


def test_compile_decode_error_is_reported_mechanical_without_a_line() -> None:
    # A file that is not decodable is not a syntax defect, but it still blocks compilation.
    f = _classify(["compile: notes.py: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9"])
    assert f is not None
    assert f.family is FailureFamily.MECHANICAL
    assert f.line is None


# --------------------------------------------------------------------------------------
# plumbing failures: the verifier itself failed, not the edit
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "err",
    [
        "pytest-unavailable: pytest is not importable",
        "pytest-timeout: pytest did not finish within 30s; killed without result",
        "pytest-error: could not read a pytest summary line (exit 4)",
        "pytest-no-tests: pytest executed no tests",
        "no-tests: no test paths were supplied, so nothing was verified",
    ],
)
def test_verifier_failures_are_plumbing(err: str) -> None:
    f = _classify([err])
    assert f is not None
    assert f.family is FailureFamily.PLUMBING
    assert f.failure_class is FailureClass.VERIFIER
    assert f.auto_repairable is False


def test_plumbing_takes_precedence_over_test_failures() -> None:
    # If the verifier says it could not run, that is the operative fact.
    f = _classify(["test-failures: 1 of 1 tests failed", "pytest-error: collection error"])
    assert f is not None
    assert f.family is FailureFamily.PLUMBING


# --------------------------------------------------------------------------------------
# NameError: typo resolution is what separates mechanical from semantic
# --------------------------------------------------------------------------------------


_NAME_ERR = (
    "test-failures: 1 of 1 executed tests failed: "
    "tests/test_core.py:3: in test_foo\n"
    "    from mypkg.core import add\n"
    "mypkg/core.py:2: in <module>\n"
    "    result = doble(a)\n"
    "E   NameError: name 'doble' is not defined"
)


def test_name_error_with_close_repo_symbol_is_mechanical() -> None:
    f = _classify([_NAME_ERR], repo_symbols={"double", "add"})
    assert f is not None
    assert f.failure_class is FailureClass.UNDEFINED_NAME
    assert f.family is FailureFamily.MECHANICAL
    assert f.symbol == "doble"
    assert f.candidates == ["double"]


def test_name_error_without_index_escalates() -> None:
    # No symbol index -> the classifier has no way to know it is a typo, so it must not
    # claim it can repair it.
    f = _classify([_NAME_ERR])
    assert f is not None
    assert f.family is FailureFamily.SEMANTIC
    assert f.candidates == []


def test_name_error_with_no_close_match_escalates() -> None:
    f = _classify([_NAME_ERR], repo_symbols={"completely_different", "add"})
    assert f is not None
    assert f.family is FailureFamily.SEMANTIC
    assert f.candidates == []


def test_name_error_far_distance_is_not_a_typo() -> None:
    err = (
        "test-failures: 1 of 1 executed tests failed: "
        "tests/t.py:4: in test_x\n    f()\n"
        "E   NameError: name 'authentication_service' is not defined"
    )
    f = _classify([err], repo_symbols={"auth", "add", "double"})
    assert f is not None
    assert f.family is FailureFamily.SEMANTIC


def test_frame_location_prefers_project_over_pytest_internals() -> None:
    err = (
        "test-failures: 1 of 1 executed tests failed: "
        "tests/t.py:4: in test_x\n"
        "    _pytest/some_plugin.py:99: in helper\n"
        "mypkg/core.py:12: in f\n"
        "    return undefined_thing\n"
        "E   NameError: name 'undefined_thing' is not defined"
    )
    f = _classify([err], repo_symbols={"undefined_thin"})
    assert f is not None
    assert f.file == "mypkg/core.py"
    assert f.line == 12


# --------------------------------------------------------------------------------------
# ImportError: mechanical only when the module is ours
# --------------------------------------------------------------------------------------


def test_import_error_of_repo_module_is_mechanical() -> None:
    err = (
        "test-failures: 1 of 1 executed tests failed: "
        "tests/t.py:2: in test_x\n    from mypkg.helpers import doubled\n"
        "E   ImportError: cannot import name 'doubled' from 'mypkg.helpers'"
    )
    f = _classify([err], repo_modules={"mypkg.helpers", "mypkg.core"})
    assert f is not None
    assert f.failure_class is FailureClass.MISSING_IMPORT
    assert f.family is FailureFamily.MECHANICAL


def test_module_not_found_of_repo_module_is_mechanical() -> None:
    err = (
        "test-failures: 1 of 1 executed tests failed: "
        "tests/t.py:2: in test_x\n    import mypkg.utils\n"
        "E   ModuleNotFoundError: No module named 'mypkg.utils'"
    )
    f = _classify([err], repo_modules={"mypkg.utils"})
    assert f is not None
    assert f.family is FailureFamily.MECHANICAL


def test_module_not_found_of_third_party_is_semantic() -> None:
    # A package the repo does not have is a dependency decision, not a repair.
    err = (
        "test-failures: 1 of 1 executed tests failed: "
        "tests/t.py:2: in test_x\n    import fastapi\n"
        "E   ModuleNotFoundError: No module named 'fastapi'"
    )
    f = _classify([err], repo_modules={"mypkg.utils"})
    assert f is not None
    assert f.family is FailureFamily.SEMANTIC


# --------------------------------------------------------------------------------------
# Assertion failures are always semantic
# --------------------------------------------------------------------------------------


def test_assertion_failure_is_semantic() -> None:
    err = (
        "test-failures: 1 of 1 executed tests failed: "
        "tests/test_core.py:5: in test_add\n    assert add(1, 2) == 3\n"
        "E   assert 4 == 3"
    )
    f = _classify([err])
    assert f is not None
    assert f.failure_class is FailureClass.ASSERTION
    assert f.family is FailureFamily.SEMANTIC
    assert f.auto_repairable is False


def test_assertion_failure_without_exception_line_is_still_semantic() -> None:
    # pytest can report a failure with no ``E ...`` line; that must not be read as
    # "no exception, so maybe mechanical".
    f = _classify(["test-failures: 2 of 2 executed tests failed: ... truncated output"])
    assert f is not None
    assert f.failure_class is FailureClass.ASSERTION
    assert f.family is FailureFamily.SEMANTIC


def test_runtime_error_is_semantic() -> None:
    err = (
        "test-failures: 1 of 1 executed tests failed: "
        "tests/t.py:6: in test_x\n    f()\n"
        "E   ValueError: invalid literal for int() with base 10: 'abc'"
    )
    f = _classify([err])
    assert f is not None
    assert f.failure_class is FailureClass.RUNTIME
    assert f.family is FailureFamily.SEMANTIC


def test_syntax_error_at_import_time_is_mechanical() -> None:
    err = (
        "test-failures: 1 of 1 executed tests failed: "
        "tests/t.py:2: in test_x\n    from mypkg.core import add\n"
        "mypkg/core.py:4: in <module>\n"
        "E   SyntaxError: invalid syntax"
    )
    f = _classify([err])
    assert f is not None
    assert f.failure_class is FailureClass.SYNTAX
    assert f.family is FailureFamily.MECHANICAL


def test_unknown_error_string_escalates_instead_of_guessing() -> None:
    f = _classify(["something the classifier has never seen"])
    assert f is not None
    assert f.family is FailureFamily.SEMANTIC
    assert f.evidence == "something the classifier has never seen"


# --------------------------------------------------------------------------------------
# summary() is the escalation message
# --------------------------------------------------------------------------------------


def test_summary_includes_location_and_class() -> None:
    f = _classify(["compile: mypkg/core.py:4: invalid syntax"])
    assert f is not None
    assert f.summary() == "[syntax_error] mypkg/core.py:4 invalid syntax"


def test_summary_without_line_still_reads() -> None:
    f = ClassifiedFailure(
        failure_class=FailureClass.RUNTIME,
        family=FailureFamily.SEMANTIC,
        message="nope",
        file="a.py",
    )
    assert f.summary() == "[runtime_error] a.py nope"


# --------------------------------------------------------------------------------------
# typo resolution primitives
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("a,b,d", [("doble", "double", 1), ("resut", "result", 1), ("abc", "xyz", 3)])
def test_edit_distance(a: str, b: str, d: int) -> None:
    assert _edit_distance(a, b, 2) <= d


def test_nearest_symbol_returns_empty_when_nothing_is_close() -> None:
    assert _nearest_symbol("authentication_service", {"auth"}) == ""


def test_nearest_symbol_returns_empty_for_empty_input() -> None:
    assert _nearest_symbol("", {"add"}) == ""
    assert _nearest_symbol("add", set()) == ""
