"""Tests for ``sylphae.exec.heal`` -- the verify-and-heal loop.

These tests run real ``pytest`` subprocesses against a small package built under ``tmp_path``,
because the loop's entire promise is that it reports what actually happened after its own edit.
A stubbed runner would verify nothing.

Invariants under test throughout:

* **A clean tree is reported as repaired only after a verification said so** -- never because a
  repair "looked right".
* **A repair that does not clear its defect leaves the file byte-identical to before.**
* **A semantic failure is escalated with its evidence, never repaired.**
* **The loop stops at ``max_attempts``, and stops early when a repair stops helping.**

Run from the repository root:

    python -m pytest tests/test_exec_heal.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sylphae.exec.heal import (  # noqa: E402
    index_repo,
    verify_and_heal,
)
from sylphae.exec.textio import read_source  # noqa: E402

HELPERS = "def double(x):\n    return x * 2\n"
CORE = "from mypkg.helpers import double\n\n\ndef add(a, b):\n    return a + b\n"
TEST_HELPERS = (
    "from mypkg.helpers import double\n\n\ndef test_double():\n    assert double(3) == 6\n"
)
TEST_CORE = "from mypkg.core import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


@pytest.fixture()
def pkg(tmp_path: Path) -> Path:
    _write(tmp_path / "conftest.py", "")
    _write(tmp_path / "mypkg" / "__init__.py", "")
    _write(tmp_path / "mypkg" / "helpers.py", HELPERS)
    _write(tmp_path / "mypkg" / "core.py", "def add(a, b):\n    return a + b\n")
    _write(tmp_path / "tests" / "test_core.py", TEST_CORE)
    _write(tmp_path / "tests" / "test_helpers.py", TEST_HELPERS)
    return tmp_path


ALL_TESTS = ["tests/test_core.py", "tests/test_helpers.py"]


def _syms(pkg: Path) -> tuple[set[str], set[str]]:
    return index_repo(pkg)


# --------------------------------------------------------------------------------------
# index_repo
# --------------------------------------------------------------------------------------


def test_index_repo_finds_top_level_symbols(pkg: Path) -> None:
    symbols, modules = _syms(pkg)
    assert "double" in symbols
    assert "add" in symbols
    assert "mypkg.core" in modules
    assert "mypkg.helpers" in modules


def test_index_repo_skips_unparseable_files(pkg: Path) -> None:
    _write(pkg / "mypkg" / "broken.py", "def broken(:\n")
    symbols, _ = _syms(pkg)
    assert "broken" not in symbols
    # The good files are still indexed.
    assert "double" in symbols


def test_index_repo_on_empty_tree(tmp_path: Path) -> None:
    symbols, modules = index_repo(tmp_path)
    assert symbols == set()
    assert modules == set()


# --------------------------------------------------------------------------------------
# verify_and_heal -- a clean change needs no repair
# --------------------------------------------------------------------------------------


def test_clean_change_reports_repaired_without_attempts(pkg: Path) -> None:
    report, outcome = verify_and_heal(pkg, ["mypkg/core.py"], ALL_TESTS, **_index_kwargs(pkg))
    assert report.clean
    assert outcome.repaired is True
    assert outcome.attempts == []
    assert outcome.verifications == 1


def test_affected_tests_narrows_the_run(pkg: Path) -> None:
    # Only the test that imports the changed module runs, so the loop is not paying for the
    # whole suite on every step.
    _, outcome = verify_and_heal(pkg, ["mypkg/helpers.py"], ALL_TESTS, **_index_kwargs(pkg))
    assert outcome.repaired is True
    final = outcome.final_report
    assert final is not None
    assert final.tests_run == 1


def _index_kwargs(pkg: Path) -> dict:
    symbols, modules = _syms(pkg)
    return {"repo_symbols": symbols, "repo_modules": modules}


# --------------------------------------------------------------------------------------
# Mechanical repairs the loop clears on its own
# --------------------------------------------------------------------------------------


def test_exact_symbol_referenced_unimported_is_escalated_not_renamed(pkg: Path) -> None:
    """``doubled`` is a real repo symbol, not a typo of ``double``.

    Distance-1 names collide easily (``doubled`` vs ``double``), so the classifier must prefer
    the exact match.  The wrong repair -- renaming ``doubled`` into ``double`` -- would compile
    and pass, and silently change which function runs.  The loop declines and escalates
    instead, because resolving "referenced but never imported" to a ``from M import X`` needs
    a symbol-to-module map this increment does not build.
    """
    before_core = "from mypkg.helpers import double\n\n\ndef half(a):\n    return doubled(a)\n"
    _write(pkg / "mypkg" / "core.py", before_core)
    _write(
        pkg / "tests" / "test_core.py",
        "from mypkg.core import half\n\n\ndef test_half():\n    assert half(4) == 4\n",
    )
    report, outcome = verify_and_heal(pkg, ["mypkg/core.py"], ["tests/test_core.py"], **_index_kwargs(pkg))
    assert outcome.repaired is False
    assert outcome.attempts
    a = outcome.attempts[0]
    assert a.failure_class == "undefined_name"
    assert a.cleared is False
    # The file is untouched: a declined repair changes nothing.
    assert read_source(pkg / "mypkg" / "core.py") == before_core


def test_missing_module_import_is_repaired(pkg: Path) -> None:
    """``import <module>`` for a module the repo actually contains."""
    _write(
        pkg / "mypkg" / "core.py",
        "import mypkg.mathops as m\n\n\ndef double_add(a, b):\n    return m.double(a) + b\n",
    )
    _write(pkg / "mypkg" / "mathops.py", "def double(x):\n    return x * 2\n")
    _write(
        pkg / "tests" / "test_core.py",
        "from mypkg.core import double_add\n\n\ndef test_it():\n    assert double_add(1, 2) == 4\n",
    )
    report, outcome = verify_and_heal(pkg, ["mypkg/core.py"], ["tests/test_core.py"], **_index_kwargs(pkg))
    assert outcome.repaired is True
    assert read_source(pkg / "mypkg" / "core.py").count("import mypkg.mathops") == 1


def test_typo_is_repaired_and_tests_pass(pkg: Path) -> None:
    # 'doble' is a misspelling of 'double', which the repo index knows about.
    _write(pkg / "mypkg" / "core.py", CORE.replace("double(add(a, b))", "doble(add(a, b))") + "\n")
    _write(
        pkg / "tests" / "test_core.py",
        "from mypkg.core import add_twice\n\n\ndef test_it():\n    assert add_twice(1, 2) == 6\n",
    )
    CORE_TWICE = (
        "from mypkg.helpers import double\n\n\ndef add(a, b):\n    return a + b\n\n\ndef add_twice(a, b):\n"
        "    return doble(add(a, b))\n"
    )
    _write(pkg / "mypkg" / "core.py", CORE_TWICE)
    report, outcome = verify_and_heal(pkg, ["mypkg/core.py"], ["tests/test_core.py"], **_index_kwargs(pkg))
    assert outcome.repaired is True
    assert read_source(pkg / "mypkg" / "core.py").count("double(add(a, b))") == 1
    final = outcome.final_report
    assert final is not None
    assert final.tests_passed == 1


def test_indentation_error_is_repaired(pkg: Path) -> None:
    _write(pkg / "mypkg" / "helpers.py", "def double(x):\nreturn x * 2\n")
    report, outcome = verify_and_heal(pkg, ["mypkg/helpers.py"], ALL_TESTS, **_index_kwargs(pkg))
    assert outcome.repaired is True
    assert read_source(pkg / "mypkg" / "helpers.py") == HELPERS


def test_repair_that_fails_is_reverted_byte_for_byte(pkg: Path) -> None:
    """The loop's promise: a failed repair leaves the tree exactly as it found it."""
    broken = "def double(x):\n    return x * 2\n\n\ndef half(x):\n    return x // 2 + 1\n"
    _write(pkg / "mypkg" / "helpers.py", broken)
    # A test that will still fail after any mechanical repair, because the defect is semantic.
    _write(pkg / "tests" / "test_helpers.py", "from mypkg.helpers import half\n\n\ndef test_half():\n    assert half(4) == 2\n")
    before = read_source(pkg / "mypkg" / "helpers.py")
    report, outcome = verify_and_heal(
        pkg, ["mypkg/helpers.py"], ["tests/test_helpers.py"], **_index_kwargs(pkg)
    )
    assert outcome.repaired is False
    assert read_source(pkg / "mypkg" / "helpers.py") == before


# --------------------------------------------------------------------------------------
# Semantic failures escalate with evidence
# --------------------------------------------------------------------------------------


def test_assertion_failure_is_escalated_not_repaired(pkg: Path) -> None:
    _write(pkg / "mypkg" / "core.py", CORE.replace("return a + b", "return a - b"))
    before = read_source(pkg / "mypkg" / "core.py")
    report, outcome = verify_and_heal(pkg, ["mypkg/core.py"], ALL_TESTS, **_index_kwargs(pkg))
    assert outcome.repaired is False
    assert outcome.attempts == []
    assert outcome.escalation
    assert "semantic" in outcome.escalation
    # The wrong answer is left for the planner to see, unchanged.
    assert read_source(pkg / "mypkg" / "core.py") == before


def test_undefined_name_with_no_index_escalates(pkg: Path) -> None:
    # Without the symbol index the classifier cannot know it is a typo, so it escalates.
    _write(pkg / "mypkg" / "core.py", CORE.replace("double(add(a, b))", "doble(add(a, b))") + "\n")
    _write(
        pkg / "tests" / "test_core.py",
        "from mypkg.core import add_twice\n\n\ndef test_it():\n    assert add_twice(1, 2) == 6\n",
    )
    _write(
        pkg / "mypkg" / "core.py",
        "from mypkg.helpers import double\n\n\ndef add(a, b):\n    return a + b\n\n\ndef add_twice(a, b):\n"
        "    return doble(add(a, b))\n",
    )
    report, outcome = verify_and_heal(pkg, ["mypkg/core.py"], ["tests/test_core.py"])
    # Without the index the classifier escalates to semantic, and the repair declines for
    # lack of a candidate -- the loop tries once, records the declined attempt, and escalates.
    assert outcome.repaired is False
    assert outcome.escalation
    assert any(a.failure_class == "undefined_name" for a in outcome.attempts)
    for a in outcome.attempts:
        assert a.cleared is False
        assert a.repair.startswith("declined")


# --------------------------------------------------------------------------------------
# Bounded loop
# --------------------------------------------------------------------------------------


def test_loop_respects_max_attempts(pkg: Path) -> None:
    _write(pkg / "mypkg" / "helpers.py", "def double(x):\n    return x * 3\n")
    _, outcome = verify_and_heal(
        pkg, ["mypkg/helpers.py"], ALL_TESTS, max_attempts=1, **_index_kwargs(pkg)
    )
    assert outcome.repaired is False
    assert len(outcome.attempts) <= 1


def test_repeat_guard_stops_a_ineffective_repair_early(pkg: Path) -> None:
    """If a repair does not change the failure, a second attempt is not made."""
    # A syntax error the loop cannot repair safely: the classifier calls it mechanical, the
    # repair declines, and the loop exits on the first attempt rather than retrying.
    _write(pkg / "mypkg" / "helpers.py", "def double(x):\n    return x * 2\n\n\nx = (1 + \n")
    _, outcome = verify_and_heal(pkg, ["mypkg/helpers.py"], ALL_TESTS, max_attempts=5, **_index_kwargs(pkg))
    assert outcome.repaired is False
    assert len(outcome.attempts) <= 1


def test_plumbing_failure_escalates_without_attempts(pkg: Path) -> None:
    # A verifier that cannot run is not a defect the loop can repair.
    _write(pkg / "mypkg" / "core.py", CORE)
    # No tests anywhere -> the verifier reports it could not verify anything.
    (pkg / "tests" / "test_core.py").unlink()
    _, outcome = verify_and_heal(pkg, ["mypkg/core.py"], ALL_TESTS, **_index_kwargs(pkg))
    assert outcome.repaired is False
    assert outcome.attempts == []


# --------------------------------------------------------------------------------------
# The outcome is honest about what it observed
# --------------------------------------------------------------------------------------


def test_attempt_records_the_failure_class_and_repair(pkg: Path) -> None:
    _write(pkg / "mypkg" / "helpers.py", "def double(x):\n    return x * 2\n\n\ndef half(x):\n    return x // 2 + 1\n")
    _write(pkg / "tests" / "test_helpers.py", "from mypkg.helpers import half\n\n\ndef test_half():\n    assert half(4) == 2\n")
    _, outcome = verify_and_heal(pkg, ["mypkg/helpers.py"], ["tests/test_helpers.py"], **_index_kwargs(pkg))
    for attempt in outcome.attempts:
        assert attempt.n >= 1
        assert attempt.failure_class
        assert attempt.repair
    assert outcome.wall_ms >= 0.0
