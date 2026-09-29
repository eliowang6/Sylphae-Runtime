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
    affected_tests,
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
