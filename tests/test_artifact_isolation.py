"""Regression tests for Sylphae's own artifacts being mistaken for project source.

Sylphae writes runtime state into the repo it operates on.  The cached index is the awkward
one: it is a *serialisation of the project*, so it holds every path and every symbol in the
tree.  That makes it a sticky false positive for two independent reasons, neither of which
has anything to do with what the user asked for:

1. **It sorts first.**  The name starts with ``.``, and ``iter_project_files`` returns a
   sorted list, so when no tool produces a candidate the fallback picks an arbitrary file
   out of the list and picks the index.  Observed: a request whose only plausible target
   was ``app.py`` reported ``target_path: .sylphae_index.json``.
2. **Content search matches it.**  Any keyword the request contains is already in the
   file's contents.  The pure-Python content search scores by keyword coverage, and the
   index covers everything.

The rule is a *prefix* rather than a filename, because the failure mode is "a new artifact
was excluded from the discovery paths that got updated and not the others".  Three ignore
tables had grown up independently (``repo/polyglot``, ``agent/tools``, ``repo/index``),
which is why the index was excluded from some paths and not others -- so this file also
freezes the single-source arrangement that replaced them.

Invariants frozen here:

1. **Artifacts are never project source**, including names that do not exist yet.
2. **The rule is narrow.**  A dot-prefixed file that *is* real config (``.eslintrc.json``)
   stays source; "skip dotfiles" would have been the wrong fix.
3. **Writer, readers and scanners agree on the index filename**, derived from one binding.
4. **The ripgrep globs are ``**/``-prefixed.**  A bare ``dir/**`` matches nothing when the
   search root is absolute, which is how the previous hardcoded exclusions came to be inert.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from sylphae.agent.effort import LowEffort
from sylphae.agent.explorer import ParallelExplorer
from sylphae.agent.reflex import ReflexAgent
from sylphae.agent.tools import _ripgrep_exclude_globs
from sylphae.repo.index import REPO_INDEX_FILE_NAME
from sylphae.repo.polyglot import (
    SYLPHAE_ARTIFACT_PREFIX,
    IGNORED_DIRS,
    is_sylphae_artifact,
    is_ignored_file,
    iter_project_files,
)

INDEX_JSON = '{"version": 1, "root": ".", "files": [], "edges": []}'


def _write(tmp_path: Path, rel: str, content: str) -> Path:
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


async def _ripgrep(root: Path, keyword: str):
    from sylphae.agent.tools import ripgrep_search

    return await ripgrep_search([keyword], root)


# --------------------------------------------------------------------------- #
# Invariant 1: the artifact namespace is recognised, present and future.
# --------------------------------------------------------------------------- #


def test_index_filename_is_in_the_artifact_namespace() -> None:
    # The coupling that keeps the rule from depending on a list somebody has to remember
    # to update: the writer's filename is *derived from* the prefix the scanner tests.
    assert REPO_INDEX_FILE_NAME.startswith(SYLPHAE_ARTIFACT_PREFIX)
    assert is_sylphae_artifact(REPO_INDEX_FILE_NAME)


def test_artifact_namespace_covers_names_that_do_not_exist_yet() -> None:
    assert is_sylphae_artifact(".sylphae_index.json")
    assert is_sylphae_artifact(".sylphae_state.db")
    assert is_sylphae_artifact(".SYLPHAE_INDEX.JSON")  # case-insensitive
    assert is_sylphae_artifact("nested/dir/.sylphae_index.json")  # basename, any depth


def test_artifact_namespace_does_not_swallow_neighbours() -> None:
    # Prefix, not substring: a project file that merely mentions "sylphae" is still source.
    assert not is_sylphae_artifact("app.py")
    assert not is_sylphae_artifact("sylphae_index.json")
    assert not is_sylphae_artifact("src/sylphae/agent.py")
    assert not is_sylphae_artifact(".eslintrc.json")


# --------------------------------------------------------------------------- #
# Invariant 2: the rule is narrow.
# --------------------------------------------------------------------------- #


def test_ignored_files_cover_artifacts_lockfiles_and_bundles() -> None:
    assert is_ignored_file(".sylphae_index.json")
    assert is_ignored_file("package-lock.json")
    assert is_ignored_file("bundle.min.js")


def test_real_config_is_still_source() -> None:
    # `.eslintrc.json` and friends are dot-prefixed and are legitimate edit targets.
    # A blanket "skip dotfiles" rule would have hidden them, which is why the fix is a
    # namespace predicate instead.
    assert not is_ignored_file(".eslintrc.json")
    assert not is_ignored_file(".gitlab-ci.yml")
    assert not is_ignored_file("app.py")


def test_iter_project_files_excludes_artifacts_but_keeps_dotfile_config(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "app.py", "TIMEOUT = 30\n")
    _write(tmp_path, ".eslintrc.json", '{"rules": {}}')
    _write(tmp_path, SYLPHAE_ARTIFACT_PREFIX + "_index.json", INDEX_JSON)
    _write(tmp_path, SYLPHAE_ARTIFACT_PREFIX + "_state.db", "binary")
    _write(tmp_path, "nested/" + SYLPHAE_ARTIFACT_PREFIX + "_index.json", INDEX_JSON)

    files = iter_project_files(tmp_path)

    assert "app.py" in files
    assert ".eslintrc.json" in files
    assert not [f for f in files if SYLPHAE_ARTIFACT_PREFIX in Path(f).name]


# --------------------------------------------------------------------------- #
# Invariant 3: writer and readers agree.
# --------------------------------------------------------------------------- #


def test_agent_writes_the_cache_under_the_shared_name(tmp_path: Path) -> None:
    _write(tmp_path, "app.py", "TIMEOUT = 30\n")
    ReflexAgent(root=tmp_path, effort=LowEffort)

    cache = tmp_path / REPO_INDEX_FILE_NAME
    assert cache.exists(), "the cache must land where the readers look for it"
    assert not [p.name for p in tmp_path.iterdir() if p.name.startswith(SYLPHAE_ARTIFACT_PREFIX) and p.name != REPO_INDEX_FILE_NAME]


def test_agent_cache_is_not_a_candidate(tmp_path: Path) -> None:
    # The observed failure: the cache exists, the request matches nothing, and the
    # fallback hands back the cache file instead of the only real source file.
    _write(tmp_path, "app.py", "TIMEOUT = 30\n\ndef helper():\n    return TIMEOUT\n")
    ReflexAgent(root=tmp_path, effort=LowEffort)
    assert (tmp_path / REPO_INDEX_FILE_NAME).exists()

    explorer = ParallelExplorer(tmp_path, repo_index=None, effort=LowEffort)
    results = asyncio.run(explorer.explore("请修复"))

    paths = [c.path for c in results]
    assert paths, "the fallback must still offer the real source file"
    assert not [p for p in paths if is_sylphae_artifact(p)]
    assert "app.py" in paths


# --------------------------------------------------------------------------- #
# Invariant 4: the ripgrep exclusions actually exclude.
# --------------------------------------------------------------------------- #


def test_ripgrep_globs_are_anchored_so_they_match() -> None:
    globs = _ripgrep_exclude_globs()

    dir_globs = [g for g in globs if g.startswith("!") and g.endswith("/**")]
    assert dir_globs, "directory exclusions must be present"
    # Measured: with an absolute search root, `--glob '!work/**'` returned `work/a.py`
    # anyway; only `--glob '!**/work/**'` dropped it.  The bare form is inert, so this
    # assertion is what stops it from creeping back.
    assert all(g.startswith("!**/") for g in dir_globs), dir_globs


def test_ripgrep_globs_track_the_shared_directory_table() -> None:
    globs = set(_ripgrep_exclude_globs())
    for directory in IGNORED_DIRS:
        assert f"!**/{directory}/**" in globs
    assert f"!**/{SYLPHAE_ARTIFACT_PREFIX}*" in globs


def test_ripgrep_skips_ignored_directories_end_to_end(tmp_path: Path) -> None:
    # The unit assertions above check the spelling of the globs; this one checks that the
    # executable honours them, because the bug was precisely a spelling that looked right.
    from sylphae.agent.tools import get_ripgrep_binary

    if not get_ripgrep_binary():
        pytest.skip("ripgrep not installed")

    (tmp_path / "node_modules").mkdir()
    _write(tmp_path, "node_modules/a.py", "MARKER = 1\n")
    _write(tmp_path, "b.py", "MARKER = 1\n")

    matches = asyncio.run(_ripgrep(tmp_path, "MARKER"))
    paths = {m.path for m in matches}

    assert "b.py" in paths, "a plain source file must still be found"
    assert not [p for p in paths if p.startswith("node_modules/")], \
        "directories in the shared table must be skipped"


def test_ripgrep_honours_repo_declared_ignores_without_a_git_checkout(tmp_path: Path) -> None:
    """A repository's own `.gitignore` must prune, even outside a git repository.

    This is the case the shared table deliberately does *not* cover: `work`, `scratch`
    and `runs` are one repository's private layout, not universal names, so the rule
    that excludes them lives in that repository's `.gitignore`
    (``sylphae/repo/ignore.py`` says so).  ripgrep applies gitignore rules only inside a
    repository unless told otherwise, so the search path and
    ``iter_project_files`` -- which reads the rules itself -- used to disagree about
    what source is, depending on whether anyone had run ``git init``.
    """
    from sylphae.agent.tools import get_ripgrep_binary

    if not get_ripgrep_binary():
        pytest.skip("ripgrep not installed")

    (tmp_path / "work").mkdir()
    _write(tmp_path, "work/a.py", "MARKER = 1\n")
    _write(tmp_path, "b.py", "MARKER = 1\n")
    _write(tmp_path, ".gitignore", "work/\n")

    assert not (tmp_path / ".git").exists(), "the point of this test is that there is no repo"

    matches = asyncio.run(_ripgrep(tmp_path, "MARKER"))
    paths = {m.path for m in matches}

    assert "b.py" in paths, "a plain source file must still be found"
    assert not [p for p in paths if p.startswith("work/")], (
        "a directory the repository declares ignored in .gitignore must be skipped even "
        "when the search root is not a git checkout"
    )


def test_ripgrep_does_not_prune_a_directory_the_repo_never_ignored(tmp_path: Path) -> None:
    """The complement: nothing declares `work/`, so it is source and must be offered.

    Without this the rule above could be satisfied by hardcoding `work` into the shared
    table, which is exactly the brittleness `sylphae/repo/ignore.py` removed -- it would
    silently skip `src/work/card.css` in every other project.
    """
    from sylphae.agent.tools import get_ripgrep_binary

    if not get_ripgrep_binary():
        pytest.skip("ripgrep not installed")

    (tmp_path / "work").mkdir()
    _write(tmp_path, "work/a.py", "MARKER = 1\n")

    matches = asyncio.run(_ripgrep(tmp_path, "MARKER"))
    paths = {m.path for m in matches}

    assert "work/a.py" in paths, (
        "`work` is not in the shared table and this repository never ignored it, so it "
        "is ordinary source and must be found"
    )
