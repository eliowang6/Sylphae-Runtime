"""Shared pytest fixtures.

``sylphae/`` is an implicit namespace package (no ``__init__.py`` at the root) and
pytest's rootdir insertion only covers ``tests/``, so we add the repo root here.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


#: ROUND 50. Point ``tempfile`` at a directory on the repository's own drive, before any test
#: module is imported and before any ``mkdtemp`` in the codebase runs.
#:
#: WHY. 12.83 measured what happens without this: C: reached **0 GB free** while D: had 55.7, and
#: the suite produced ``OSError: [WinError 112] 磁盘空间不足`` across 28 failures and 33 errors on a
#: clean tree at a known-good commit. ``--basetemp=D:\tmp\...`` was on D: the whole time and did not
#: help, because the code's own ``tempfile.mkdtemp()`` calls ignore it and go to the OS temp
#: directory. The wall time doubled too -- 393s to 811s -- which is a degrading disk, not a slow
#: machine.
#:
#: AGENTS.md sets a floor for D: and says nothing about C:, so the rule did not cover the disk
#: actually being written. This closes that for the test suite; it does not change the rule.
#:
#: Respecting an explicit ``TMPDIR``/``TEMP``/``TMP`` means a caller who has already chosen a
#: location is not overridden.
_MIN_FREE_GB = 2.0


def _relocate_tempdir() -> str | None:
    explicit = os.environ.get("TMPDIR") or os.environ.get("TEMP") or os.environ.get("TMP")
    target_root = ROOT.parent / "tmp" / "sylphae-temp"
    current = Path(tempfile.gettempdir())
    try:
        free_gb = shutil.disk_usage(current).free / 1024 ** 3
    except OSError:
        return None
    if free_gb >= _MIN_FREE_GB and current.drive.lower() == target_root.drive.lower():
        return None  # already on the repo drive with room
    if explicit and str(current).lower().startswith(str(target_root).lower()):
        return None  # the caller pointed at us already
    try:
        target_root.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(target_root)
        os.environ["TMPDIR"] = str(target_root)
        os.environ["TEMP"] = str(target_root)
        os.environ["TMP"] = str(target_root)
    except OSError:
        return None
    return f"{current} ({free_gb:.1f} GB free) -> {target_root}"


_TEMP_MOVED = _relocate_tempdir()


def pytest_report_header(config):
    """Say where temp lives. ROUND 50: a silent relocation is indistinguishable from a mystery."""
    cur = tempfile.gettempdir()
    free_gb = shutil.disk_usage(cur).free / 1024 ** 3
    moved = f" (relocated this session: {_TEMP_MOVED})" if _TEMP_MOVED else ""
    return f"tempdir: {cur} -- {free_gb:.1f} GB free{moved}"


@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_path_factory):
    """A small but real byte-level BPE tokenizer, trained on local stdlib files."""
    from sylphae.model.tokenizer import stdlib_corpus_paths, train_tokenizer

    files = stdlib_corpus_paths(max_files=25, max_bytes=400_000)
    assert files, "no stdlib corpus files found"
    out = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    return train_tokenizer(files, vocab_size=768, out_path=out)
