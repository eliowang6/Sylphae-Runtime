"""Invariants for repository-level ignore declarations.

The shared table in ``polyglot.IGNORED_DIRS`` may only carry directory names whose
meaning is universal.  It used to also carry ``work``, ``scratch``, ``runs`` and
``corpus`` -- this repository's private layout -- which removed a legitimate directory,
silently, from every other project Sylphae indexes.  Rules that describe one repository now
come from that repository, and these cases pin both halves: the shared table stays
narrow, and a project can still exclude whatever it wants.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sylphae.repo.ignore import _compile
from sylphae.repo.polyglot import (
    IGNORED_DIRS,
    is_sylphae_artifact,
    iter_project_files,
)

PRIVATE_NAMES = {"work", "scratch", "runs", "corpus", "release_cache", "reports"}


def _write(root: Path, rel: str, content: str = "x = 1\n") -> Path:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


# --------------------------------------------------------------------------- #
# the shared table
# --------------------------------------------------------------------------- #


def test_shared_table_carries_no_project_layout() -> None:
    assert not (IGNORED_DIRS & PRIVATE_NAMES), (
        "a general-purpose indexer must not skip one project's private directory "
        "names in every other project"
    )


def test_universal_names_are_still_excluded() -> None:
    # Narrowing the table must not have emptied it.
    for name in ("node_modules", ".git", "dist", "build", "__pycache__", "venv"):
        assert name in IGNORED_DIRS


# --------------------------------------------------------------------------- #
# a directory named like this repository's layout is visible elsewhere
# --------------------------------------------------------------------------- #


def test_private_directory_name_is_indexed_in_a_third_party_repo(tmp_path: Path) -> None:
    _write(tmp_path, "src/work/card.css", ".card { height: 40px; }\n")
    _write(tmp_path, "src/normal/btn.css", ".btn { height: 30px; }\n")

    files = iter_project_files(tmp_path)

    assert "src/work/card.css" in files, (
        "without a declaration of its own, a repository's src/work/ is source"
    )
    assert "src/normal/btn.css" in files


def test_a_repository_can_still_exclude_that_directory(tmp_path: Path) -> None:
    _write(tmp_path, "src/work/card.css")
    _write(tmp_path, "src/normal/btn.css")
    (tmp_path / ".gitignore").write_text("src/work/\n", encoding="utf-8")

    files = iter_project_files(tmp_path)

    assert "src/work/card.css" not in files
    assert "src/normal/btn.css" in files


def test_sylphaeignore_applies_without_touching_git(tmp_path: Path) -> None:
    _write(tmp_path, "src/work/card.css")
    _write(tmp_path, "src/normal/btn.css")
    (tmp_path / ".sylphaeignore").write_text("src/normal/\n", encoding="utf-8")

    files = iter_project_files(tmp_path)

    assert "src/normal/btn.css" not in files
    assert "src/work/card.css" in files


def test_sylphaeignore_is_configuration_not_an_artifact() -> None:
    # It carries the artifact prefix because it is Sylphae's, but it declares what to skip
    # and belongs in version control next to .gitignore.  The prefix rule would
    # otherwise swallow the file whose whole purpose is to describe exclusions.
    assert is_sylphae_artifact(".sylphaeignore") is False
    assert is_sylphae_artifact(".sylphae_index.json") is True
    assert is_sylphae_artifact("sub/.sylphae_state.db") is True


# --------------------------------------------------------------------------- #
# the gitignore subset
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "pattern, path, is_dir, expected",
    [
        ("work/", "work", True, True),
        ("work/", "a/b/work", True, True),
        # A rule with a slash inside it is anchored to the repository root.
        ("/src/gen", "src/gen", True, True),
        ("/src/gen", "a/src/gen", True, False),
        ("*.log", "x/y.log", False, True),
        ("**/build/", "a/b/build", True, True),
        ("docs/**", "docs/a/b.md", False, True),
        # A trailing-slash rule does not ignore a file of the same name.
        ("work/", "work", False, False),
        # `*` does not cross a separator.
        ("data/*.npz", "data/a.npz", False, True),
        ("data/*.npz", "data/sub/a.npz", False, False),
        ("a/**/b", "a/x/y/b", False, True),
        ("node_modules", "x/node_modules", True, True),
    ],
)
def test_gitignore_subset(pattern: str, path: str, is_dir: bool, expected: bool) -> None:
    rule = _compile(pattern)
    assert rule is not None, pattern
    matched = bool(rule.pattern.match(path)) and not (rule.dir_only and not is_dir)
    assert matched == expected


def test_comments_and_blanks_carry_no_rule() -> None:
    assert _compile("# a comment") is None
    assert _compile("") is None
    assert _compile("   ") is None


def test_negation_re_enables_a_path(tmp_path: Path) -> None:
    _write(tmp_path, "gen/a.py")
    _write(tmp_path, "gen/keep.py")
    (tmp_path / ".gitignore").write_text("gen/\n!gen/keep.py\n", encoding="utf-8")

    files = iter_project_files(tmp_path)

    # The directory rule prunes gen/ before the negation can re-include a file inside
    # it, which is git's behaviour too: a re-included file needs its parent re-included.
    assert "gen/a.py" not in files


def test_a_repository_without_declarations_still_indexes_normally(tmp_path: Path) -> None:
    _write(tmp_path, "app.py")
    _write(tmp_path, "src/main.ts", "export const a = 1;\n")

    files = iter_project_files(tmp_path)

    assert "app.py" in files
    assert "src/main.ts" in files
