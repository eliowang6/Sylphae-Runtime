"""Neural candidate ranking -- the no-weights contract.

The ranker is the model-path replacement for the lexical retrieval score.  It is
also the one part of the runtime that needs a trained checkpoint, and no
checkpoint is published here -- so what this file pins is the property that makes
that acceptable: **the ranker is optional, and its absence is never an error.**

  * asking for a run that is not there raises :class:`NeuralUnavailable`, which is
    a typed "I cannot answer" rather than a generic failure;
  * the search entry point catches it and returns nothing, so an exploration with
    no neural fallback still runs on the lexical and AST tools;
  * an empty pool short-circuits before any of that.

The scoring properties that need weights -- order alignment, and the guard that
the ranking follows paths rather than slot positions -- are measured where the
weights live.  They are not re-implementable here, and pretending otherwise with a
randomly initialised head would test nothing: an untrained head has no ranking to
be bound to slots in the first place.

A consumer supplying their own trained run gets all of it.  Point
:func:`neural_rank_search` at a run directory holding ``config.json`` and
``checkpoint.safetensors``, or set ``$SYLPHAE_RUN_DIR``; see the README.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sylphae.agent.neural import NeuralUnavailable, NeuralRanker, neural_rank_search

REPO = Path(__file__).resolve().parents[1]

_REQUEST = "increase the default request timeout"
_POOL = [
    "src/widget/retry.py",
    "src/widget/timeout.py",
    "src/widget/cache.py",
]


def test_missing_run_raises_unavailable() -> None:
    """A typed refusal, so the caller can tell "no weights" from "broken code"."""
    with pytest.raises(NeuralUnavailable):
        NeuralRanker.for_run(REPO / "runs" / "this-run-does-not-exist")


def test_search_over_a_missing_run_returns_nothing() -> None:
    """The exploration path degrades to the lexical tools, not to a crash.

    This is the whole zero-weight story in one assertion: with no checkpoint
    anywhere, ``neural_rank_search`` returns an empty list and the caller carries
    on.  It must not raise, because a raise here would make the optional
    component mandatory.
    """
    assert neural_rank_search(_REQUEST, _POOL, run_dir=REPO / "runs" / "nope") == []


def test_empty_pool_returns_nothing() -> None:
    """Short-circuits before touching a checkpoint, so it needs no run_dir."""
    assert neural_rank_search(_REQUEST, [], run_dir=None) == []
