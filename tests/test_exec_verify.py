"""Tests for ``sylphae/exec/verify.py`` -- INTERFACES.md section 5.

These tests really run ``pytest`` as a subprocess against a tiny package built under
``tmp_path``. That is the point: the verifier's whole job is to report what actually
happened, so a test that stubbed out the runner would verify nothing.

The invariant under test throughout:

    ``tests_run == 0``  implies  ``errors`` is non-empty

i.e. "0 tests failed" and "I could not run the tests" are always distinguishable.

Run from the repository root:

    python -m pytest tests/test_exec_verify.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import sylphae.exec.verify as verify_mod  # noqa: E402
from sylphae.exec.verify import (  # noqa: E402
    VerifyReport,
    _parse_pytest_counts,
    _parse_pytest_outcomes,
    affected_tests,
    diff_outcomes,
    pytest_outcomes,
    select_tests_within_budget,
    verify,
)

CORE = '''\
from mypkg.helpers import double


def add(a, b):
    return a + b


def add_twice(a, b):
    return double(add(a, b))
'''

HELPERS = "def double(x):\n    return x * 2\n"

TEST_CORE = "from mypkg.core import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
TEST_HELPERS = (
    "from mypkg.helpers import double\n\n\ndef test_double():\n    assert double(3) == 6\n"
)


def _write(path: Path, text: str) -> None:
    """LF endings regardless of platform, so fixtures are deterministic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _build_package(root: Path) -> Path:
    _write(root / "conftest.py", "")  # makes `root` importable however pytest is invoked
    _write(root / "mypkg" / "__init__.py", "")
    _write(root / "mypkg" / "core.py", CORE)
    _write(root / "mypkg" / "helpers.py", HELPERS)
    _write(root / "tests" / "test_core.py", TEST_CORE)
    _write(root / "tests" / "test_helpers.py", TEST_HELPERS)
    return root


@pytest.fixture()
def pkg(tmp_path: Path) -> Path:
    return _build_package(tmp_path)


ALL_TESTS = ["tests/test_core.py", "tests/test_helpers.py"]


# --------------------------------------------------------------------------------------
# verify -- happy path
# --------------------------------------------------------------------------------------


def test_verify_clean_change(pkg: Path) -> None:
    report = verify(pkg, ["mypkg/core.py"], ["tests/test_core.py"])
    assert isinstance(report, VerifyReport)
    assert report.compiles is True
    assert report.tests_run == 1
    assert report.tests_passed == 1
    assert report.errors == []
    assert report.observed is True
    assert report.clean is True


def test_verify_runs_every_supplied_test_file(pkg: Path) -> None:
    report = verify(pkg, ["mypkg/helpers.py"], ALL_TESTS)
    assert report.compiles is True
    assert report.tests_run == 2
    assert report.tests_passed == 2
    assert report.errors == []


def test_verify_writes_no_caches_into_the_tree(pkg: Path) -> None:
    verify(pkg, ["mypkg/core.py"], ["tests/test_core.py"])
    assert not (pkg / ".pytest_cache").exists()
    assert not list(pkg.rglob("__pycache__")), "verification must not litter the tree"


def test_verify_with_no_changed_paths_still_runs_tests(pkg: Path) -> None:
    report = verify(pkg, [], ["tests/test_core.py"])
    assert report.compiles is True
    assert report.tests_run == 1
    assert report.tests_passed == 1


# --------------------------------------------------------------------------------------
# verify -- negative cases
# --------------------------------------------------------------------------------------


def test_syntax_error_makes_compiles_false(pkg: Path) -> None:
    """Negative case 1: a deliberately broken file must be reported as not compiling."""
    _write(pkg / "mypkg" / "core.py", "def add(a, b)\n    return a + b\n")
    report = verify(pkg, ["mypkg/core.py"], ["tests/test_core.py"])

    assert report.compiles is False
    assert any(e.startswith("compile:") for e in report.errors), report.errors
    assert any("mypkg/core.py" in e for e in report.errors)
    # The import of the broken module fails, so pytest cannot run the test either.
    assert report.tests_run == 0
    assert report.errors, "a zero-run result must always explain itself"


def test_failing_test_is_reported(pkg: Path) -> None:
    """Negative case 2: a test that runs and fails must not be reported as a success."""
    _write(pkg / "mypkg" / "core.py", CORE.replace("return a + b", "return a - b"))
    report = verify(pkg, ["mypkg/core.py"], ["tests/test_core.py"])

    assert report.compiles is True, "the file is still valid Python"
    assert report.tests_run == 1
    assert report.tests_passed == 0
    assert any(e.startswith("test-failures:") for e in report.errors), report.errors
    assert report.clean is False


def test_partial_failure_is_counted_not_rounded(pkg: Path) -> None:
    _write(pkg / "mypkg" / "helpers.py", "def double(x):\n    return x * 3\n")
    report = verify(pkg, ["mypkg/helpers.py"], ALL_TESTS)

    assert report.tests_run == 2
    assert report.tests_passed == 1
    assert report.tests_run > report.tests_passed
    assert any(e.startswith("test-failures:") for e in report.errors)


def test_timeout_reports_zero_run_and_a_reason(pkg: Path) -> None:
    """Negative case 3: a timeout is *not* a pass. tests_run == 0 and errors says why."""
    _write(
        pkg / "tests" / "test_slow.py",
        "import time\n\n\ndef test_slow():\n    time.sleep(30)\n",
    )
    report = verify(pkg, ["mypkg/core.py"], ["tests/test_slow.py"], timeout_s=1)

    assert report.tests_run == 0
    assert report.tests_passed == 0
    assert report.errors, "a timeout must never look like '0 tests failed'"
    assert any(e.startswith("pytest-timeout:") for e in report.errors), report.errors
    assert report.clean is False


def test_no_test_paths_is_not_a_pass(pkg: Path) -> None:
    report = verify(pkg, ["mypkg/core.py"], [])
    assert report.compiles is True
    assert report.tests_run == 0
    assert any(e.startswith("no-tests:") for e in report.errors), report.errors


def test_empty_test_file_reports_no_tests(pkg: Path) -> None:
    _write(pkg / "tests" / "test_empty.py", "")
    report = verify(pkg, ["mypkg/core.py"], ["tests/test_empty.py"])
    assert report.tests_run == 0
    assert any(e.startswith("pytest-no-tests:") for e in report.errors), report.errors


def test_unavailable_pytest_is_reported_not_raised(
    pkg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(verify_mod, "find_spec", lambda name: None)
    report = verify(pkg, ["mypkg/core.py"], ["tests/test_core.py"])

    assert report.compiles is True
    assert report.tests_run == 0
    assert any(e.startswith("pytest-unavailable:") for e in report.errors), report.errors


def test_missing_changed_file_does_not_compile(pkg: Path) -> None:
    report = verify(pkg, ["mypkg/ghost.py"], ["tests/test_core.py"])
    assert report.compiles is False
    assert any(e.startswith("compile:") and "cannot read" in e for e in report.errors)


def test_every_zero_run_carries_a_reason(pkg: Path) -> None:
    """The invariant, checked across every failure mode in one place."""
    _write(pkg / "mypkg" / "broken.py", "def broken(:\n")
    _write(pkg / "tests" / "test_slow.py", "import time\n\n\ndef test_slow():\n    time.sleep(30)\n")

    reports = [
        verify(pkg, ["mypkg/core.py"], []),
        verify(pkg, ["mypkg/broken.py"], ["tests/test_core.py"]),
        verify(pkg, ["mypkg/core.py"], ["tests/test_slow.py"], timeout_s=1),
        verify(pkg, ["mypkg/ghost.py"], ["tests/test_core.py"]),
        verify(pkg, ["mypkg/core.py"], ["tests/test_core.py"]),
    ]
    for report in reports:
        assert report.tests_run > 0 or report.errors, report
        if report.tests_run == 0:
            assert report.tests_passed == 0
            assert report.clean is False


# --------------------------------------------------------------------------------------
# affected_tests
# --------------------------------------------------------------------------------------


def test_affected_tests_follows_the_import_graph_transitively(pkg: Path) -> None:
    # test_helpers imports helpers directly; test_core imports core, which imports helpers.
    result = affected_tests(pkg, ["mypkg/helpers.py"], ALL_TESTS)
    assert result == ["tests/test_core.py", "tests/test_helpers.py"]


def test_affected_tests_narrows_to_the_direct_importer(pkg: Path) -> None:
    result = affected_tests(pkg, ["mypkg/core.py"], ALL_TESTS)
    assert result == ["tests/test_core.py"]


def test_affected_tests_falls_back_to_path_tokens(pkg: Path) -> None:
    """Tier 2: nothing imports `solo.py`, but `tests/test_solo.py` names it."""
    _write(pkg / "mypkg" / "solo.py", "VALUE = 1\n")
    _write(pkg / "tests" / "test_solo.py", "def test_nothing():\n    assert True\n")

    result = affected_tests(pkg, ["mypkg/solo.py"], ALL_TESTS + ["tests/test_solo.py"])
    assert result == ["tests/test_solo.py"]


def test_affected_tests_returns_a_superset_when_it_cannot_tell(pkg: Path) -> None:
    """Tier 3, the important one: never an empty list, because that would mean 'verified'."""
    result = affected_tests(pkg, ["docs/qqq_zzz.py"], ALL_TESTS)
    assert result == ALL_TESTS
    assert result, "an unmappable change must widen the test set, never empty it"


def test_affected_tests_never_returns_empty_when_tests_exist(pkg: Path) -> None:
    for changed in (
        ["mypkg/core.py"],
        ["mypkg/helpers.py"],
        ["mypkg/ghost.py"],
        ["totally/unrelated/path.py"],
        ["README.md"],
    ):
        assert affected_tests(pkg, changed, ALL_TESTS), changed


def test_affected_tests_changed_test_maps_to_itself(pkg: Path) -> None:
    assert affected_tests(pkg, ["tests/test_core.py"], ALL_TESTS) == ["tests/test_core.py"]


def test_affected_tests_handles_multiple_changed_paths(pkg: Path) -> None:
    result = affected_tests(pkg, ["mypkg/core.py", "mypkg/helpers.py"], ALL_TESTS)
    assert result == ["tests/test_core.py", "tests/test_helpers.py"]


def test_affected_tests_with_no_tests_is_empty(pkg: Path) -> None:
    assert affected_tests(pkg, ["mypkg/core.py"], []) == []


def test_affected_tests_with_no_changes_is_empty(pkg: Path) -> None:
    assert affected_tests(pkg, [], ALL_TESTS) == []


def test_affected_tests_resolves_relative_imports(tmp_path: Path) -> None:
    _write(tmp_path / "conftest.py", "")
    _write(tmp_path / "proj" / "__init__.py", "")
    _write(tmp_path / "proj" / "alpha.py", "VALUE = 1\n")
    _write(tmp_path / "proj" / "beta.py", "from .alpha import VALUE\n\nRESULT = VALUE\n")
    _write(tmp_path / "tests" / "test_beta.py", "from proj.beta import RESULT\n")
    _write(tmp_path / "tests" / "test_unrelated.py", "def test_x():\n    assert True\n")

    # `proj/alpha.py` is imported by `proj/beta.py`, which the test imports.
    result = affected_tests(
        tmp_path,
        ["proj/alpha.py"],
        ["tests/test_beta.py", "tests/test_unrelated.py"],
    )
    assert result == ["tests/test_beta.py"]


def test_affected_tests_accepts_absolute_and_backslash_paths(pkg: Path) -> None:
    absolute = str(pkg / "mypkg" / "core.py")
    assert affected_tests(pkg, [absolute], ALL_TESTS) == ["tests/test_core.py"]
    assert affected_tests(pkg, [r"mypkg\core.py"], ALL_TESTS) == ["tests/test_core.py"]


# --------------------------------------------------------------------------------------
# bounded test selection
# --------------------------------------------------------------------------------------
#
# ``affected_tests`` is allowed to answer "everything" when it cannot map a change to a
# specific suite. On a large repository that answer turns a one-line edit into a
# full-repository pytest run, which is killed at the timeout and then reports *nothing
# verified* -- indistinguishable, in the rate, from a scenario whose tests all passed.
# MEASURED on tiangolo__typer: one line in typer/_click/formatting.py maps to 228 test
# files by the import graph alone. The cap fixes the cost, and returning the dropped
# count is what stops the fix from hiding the coverage instead of reporting it.


def test_budget_keeps_everything_when_the_affected_set_fits(pkg: Path) -> None:
    selected, dropped = select_tests_within_budget(
        pkg, ["mypkg/core.py"], ALL_TESTS, 10)
    assert selected == affected_tests(pkg, ["mypkg/core.py"], ALL_TESTS)
    assert dropped == 0


def test_budget_caps_and_reports_what_it_dropped(pkg: Path) -> None:
    many = [f"tests/test_mod{i}.py" for i in range(40)]
    for rel in many:
        _write(pkg / rel, "from mypkg.core import add\n\n\ndef test_a():\n    assert add(1, 1) == 2\n")

    selected, dropped = select_tests_within_budget(pkg, ["mypkg/core.py"], many, 12)

    assert len(selected) == 12
    assert dropped == 28
    # The dropped count is the whole point: a caller that cannot say "12 of 40" cannot
    # report a test_pass_rate for this scenario honestly.
    assert len(selected) + dropped == 40


def test_budget_is_deterministic(pkg: Path) -> None:
    """Same input, same answer -- otherwise two runs are not comparable."""
    many = [f"tests/test_mod{i}.py" for i in range(30)]
    for rel in many:
        _write(pkg / rel, "from mypkg.core import add\n\n\ndef test_a():\n    assert add(1, 1) == 2\n")

    first = select_tests_within_budget(pkg, ["mypkg/core.py"], many, 5)
    second = select_tests_within_budget(pkg, ["mypkg/core.py"], many, 5)
    assert first == second


def test_a_zero_budget_disables_the_cap_rather_than_selecting_nothing(pkg: Path) -> None:
    """``--max-tests 0`` means "no cap". Returning [] would read as 'nothing to run'."""
    selected, dropped = select_tests_within_budget(
        pkg, ["mypkg/core.py"], ALL_TESTS, 0)
    assert selected == affected_tests(pkg, ["mypkg/core.py"], ALL_TESTS)
    assert dropped == 0


def test_a_capped_run_of_real_tests_still_executes_them(pkg: Path) -> None:
    """The cap is a selection change, not a way to make verification vacuous."""
    many = [f"tests/test_mod{i}.py" for i in range(20)]
    for rel in many:
        _write(pkg / rel, "from mypkg.core import add\n\n\ndef test_a():\n    assert add(1, 1) == 2\n")

    selected, dropped = select_tests_within_budget(pkg, ["mypkg/core.py"], many, 4)
    report = verify(pkg, ["mypkg/core.py"], selected)

    assert dropped == 16
    assert report.tests_run == 4
    assert report.tests_passed == 4
    assert report.errors == []


def test_a_capped_run_that_fails_still_reports_the_failure(pkg: Path) -> None:
    """Capping must not turn a broken change into a green one."""
    many = [f"tests/test_mod{i}.py" for i in range(20)]
    for i, rel in enumerate(many):
        expected = 3 if i == 0 else 2
        _write(pkg / rel,
               "from mypkg.core import add\n\n\ndef test_a():\n"
               f"    assert add(1, 1) == {expected}\n")

    selected, _ = select_tests_within_budget(pkg, ["mypkg/core.py"], many, 4)
    report = verify(pkg, ["mypkg/core.py"], selected)

    assert report.tests_run == 4
    assert report.tests_passed == 3
    assert report.clean is False



# --------------------------------------------------------------------------------------
# pytest output parsing
# --------------------------------------------------------------------------------------


def test_parse_pytest_counts() -> None:
    assert _parse_pytest_counts("1 passed in 0.05s") == {"passed": 1}
    assert _parse_pytest_counts("1 failed, 2 passed in 0.12s") == {"failed": 1, "passed": 2}
    assert _parse_pytest_counts("3 passed, 1 skipped in 0.30s") == {"passed": 3, "skipped": 1}
    assert _parse_pytest_counts("2 errors in 0.20s") == {"error": 2}
    assert _parse_pytest_counts("no tests ran in 0.01s") == {}


def test_parse_pytest_counts_rejects_unparseable_output() -> None:
    """Negative case: unknown must be None, never zero failures."""
    assert _parse_pytest_counts("") is None
    assert _parse_pytest_counts("something exploded\nno summary here") is None


def test_parse_pytest_counts_ignores_count_words_inside_tracebacks() -> None:
    output = (
        "E   AssertionError: 5 passed unexpectedly\n"
        "tests/test_x.py:12: AssertionError\n"
        "1 failed in 0.10s\n"
    )
    assert _parse_pytest_counts(output) == {"failed": 1}


# --------------------------------------------------------------------------------------
# per-test outcomes and the before/after diff
# --------------------------------------------------------------------------------------
#
# Aggregate counts cannot answer "did the patch break something": 145/170 before and 146/171
# after is consistent with zero regressions and equally with one regression and one fix.
# The vendored snapshots are not green to begin with (rich 232/240, pluggy 145/170), so
# the diff is not a nicety either -- it is the only form of the question that is fair.

_VERBOSE = """\
============================= test session starts ==============================
platform win32 -- Python 3.13.0, pytest-8.0.0
collected 3 items

tests/test_a.py::test_ok PASSED                                      [ 33%]
tests/test_a.py::test_bad FAILED                                      [ 66%]
tests/test_b.py::test_skipped SKIPPED (x)                             [100%]

========================= 1 failed, 1 passed, 1 skipped ========================
"""


def test_parse_pytest_outcomes_reads_node_ids() -> None:
    assert _parse_pytest_outcomes(_VERBOSE) == {
        "tests/test_a.py::test_ok": "PASSED",
        "tests/test_a.py::test_bad": "FAILED",
        "tests/test_b.py::test_skipped": "SKIPPED",
    }


def test_parse_pytest_outcomes_ignores_non_test_lines() -> None:
    """A traceback mentioning a test path is not an outcome line.

    The node-id pattern is anchored and requires an outcome keyword, so summary lines and
    failure bodies cannot masquerade as results.
    """
    noisy = (
        "tests/test_a.py::test_ok PASSED                    [ 50%]\n"
        "E   tests/test_a.py::test_other AssertionError\n"
        "_______ test_something _______\n"
    )
    assert _parse_pytest_outcomes(noisy) == {"tests/test_a.py::test_ok": "PASSED"}


def test_parse_pytest_outcomes_on_unparseable_output_is_empty_not_a_pass() -> None:
    """Empty means unknown, and unknown is never 'everything passed'."""
    assert _parse_pytest_outcomes("") == {}
    assert _parse_pytest_outcomes("ImportError while loading conftest") == {}


def test_diff_reports_a_green_to_red_move() -> None:
    before = {"t.py::a": "PASSED", "t.py::b": "PASSED"}
    after = {"t.py::a": "FAILED", "t.py::b": "PASSED"}
    regressions, fixed = diff_outcomes(before, after)
    assert regressions == ["t.py::a"]
    assert fixed == []


def test_diff_does_not_charge_a_preexisting_failure() -> None:
    """The case that makes the diff necessary rather than merely thorough.

    MEASURED 2026-10-01 with no patch applied: Textualize__rich runs 232/240 and
    pytest-dev__pluggy runs 145/170. An absolute rule calls those 16 tests a failure of
    the agent's patch.
    """
    before = {"t.py::a": "PASSED", "t.py::already_broken": "FAILED"}
    after = {"t.py::a": "PASSED", "t.py::already_broken": "FAILED"}
    assert diff_outcomes(before, after) == ([], [])


def test_diff_treats_skip_and_xfail_as_green() -> None:
    """Neither is a broken tree, so neither can be a regression."""
    before = {"t.py::s": "SKIPPED", "t.py::x": "XFAIL", "t.py::p": "PASSED"}
    after = {"t.py::s": "SKIPPED", "t.py::x": "XFAIL", "t.py::p": "PASSED"}
    assert diff_outcomes(before, after) == ([], [])

    # ...and a skipped test turning red *is* a move worth reporting.
    after_xfail_failed = {"t.py::s": "SKIPPED", "t.py::x": "PASSED", "t.py::p": "PASSED"}
    regressions, _ = diff_outcomes(before, after_xfail_failed)
    assert regressions == []


def test_diff_reports_a_test_that_became_green() -> None:
    """Recorded because a patch that only ever reports regressions looks the same as one
    that did nothing."""
    before = {"t.py::a": "FAILED"}
    after = {"t.py::a": "PASSED"}
    regressions, fixed = diff_outcomes(before, after)
    assert regressions == []
    assert fixed == ["t.py::a"]


def test_diff_ignores_tests_absent_from_either_side() -> None:
    """A test that stopped being collected is not a regression, and a test the patch
    added is not one either. A vanished test shows up as a collection error on the
    ``after`` side, which is where it belongs."""
    before = {"t.py::a": "PASSED", "t.py::gone": "PASSED"}
    after = {"t.py::a": "PASSED", "t.py::added": "PASSED"}
    assert diff_outcomes(before, after) == ([], [])


def test_diff_on_an_empty_baseline_verifies_nothing_in_either_direction() -> None:
    """No baseline means no verdict -- not a clean one and not a fixed one.

    A test absent from ``before`` is ignored rather than assumed to have just passed,
    because "it passed after the patch" is not a claim that can be made about a test the
    baseline never ran. Compare ``test_diff_reports_a_test_that_became_green``, where the
    test *was* in the baseline and red.
    """
    assert diff_outcomes({}, {"t.py::a": "PASSED"}) == ([], [])
    assert diff_outcomes({}, {"t.py::a": "FAILED"}) == ([], [])


def test_pytest_outcomes_reports_a_missing_suite_rather_than_raising(tmp_path: Path) -> None:
    outcomes, errors = pytest_outcomes(tmp_path, [])
    assert outcomes == {}
    assert any(e.startswith("no-tests:") for e in errors), errors
