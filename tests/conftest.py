"""Shared pytest fixtures.

``sylphae/`` is an implicit namespace package (no ``__init__.py`` at the root) and
pytest's rootdir insertion only covers ``tests/``, so we add the repo root here.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_path_factory):
    """A small but real byte-level BPE tokenizer, trained on local stdlib files."""
    from sylphae.model.tokenizer import stdlib_corpus_paths, train_tokenizer

    files = stdlib_corpus_paths(max_files=25, max_bytes=400_000)
    assert files, "no stdlib corpus files found"
    out = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    return train_tokenizer(files, vocab_size=768, out_path=out)
