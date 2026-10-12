"""Shared summary statistics, deliberately neutral and free of heavy imports.

Two reasons this module exists at the top level of ``sylphae`` rather than inside
``eval``:

**It was two implementations.** :mod:`sylphae.eval.bench_latency` used linear
interpolation and raised on an empty sample; :mod:`sylphae.repo.bench_recall` used
nearest-rank and returned ``None``. The same data produced two different p95 numbers
depending on which evaluator reported it -- exactly the failure
:mod:`sylphae.eval.bench_latency`'s own docstring warns about ("a percentile function
nobody can hand-check is how a latency claim ends up unverifiable"), already realised.

**It is imported by both sides of the runtime split.** ``sylphae.repo`` ships in the
runtime and ``tests/test_runtime_split.py`` rule 2 forbids the runtime importing
``sylphae.eval``, so a shared definition cannot live under ``eval/``. Rule 3's existing
shared modules (``ops/params.py``, ``model/io.py``, ``model/decode.py``) are named for
the coupling they break; this one breaks runtime -> eval, and the neutrality it asserts
is also why it imports nothing but ``math``: a module both sides import is worthless if
it leans on either side, and torch would be leaning on the training side.

One definition now lives here, and it is the one pinned by hand-computed tests in
``tests/test_eval_bench_latency.py``: linear interpolation, the numpy default, so
``percentile([1, 2, 3, 4], 50) == 2.5`` exactly.
"""

from __future__ import annotations

import math

__all__ = ["percentile", "percentile_or_none", "latency_summary"]


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (the numpy default), on raw samples.

    Raises on an empty sample: a percentile of nothing does not exist, and returning
    ``0.0`` would invent a measurement. Callers that want to *report* an absence rather
    than fail should use :func:`percentile_or_none`.
    """
    if not values:
        raise ValueError(
            "percentile: empty sample list; a percentile of nothing does not exist"
        )
    if not isinstance(q, (int, float)) or isinstance(q, bool):
        raise TypeError(f"percentile: q must be a number, got {type(q).__name__}")
    if not 0.0 <= float(q) <= 100.0:
        raise ValueError(f"percentile: q must lie in [0, 100], got {q}")
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * (float(q) / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def percentile_or_none(values: list[float], q: float) -> float | None:
    """:func:`percentile`, or ``None`` when there is nothing to measure.

    The reporting variant. ``None`` and ``0.0`` are different answers and conflating them
    is how an unmeasured phase gets reported as an instant one.
    """
    if not values:
        return None
    return percentile(values, q)


def latency_summary(values: list[float | None]) -> dict[str, float | int | None]:
    """p50/p95/mean/min/max/std over the *measured* samples, plus how many there were.

    ``None`` entries are dropped rather than counted as zero, and ``n`` is reported
    alongside every figure so a reader can see the population -- a p95 over 4 samples is
    not a p95 in any useful sense, and printing it without ``n`` would hide that.
    """
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return {"n": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None,
                "min_ms": None, "max_ms": None, "std_ms": None}
    n = len(xs)
    mean = sum(xs) / n
    variance = sum((x - mean) ** 2 for x in xs) / n
    return {
        "n": n,
        "mean_ms": mean,
        "p50_ms": percentile(xs, 50.0),
        "p95_ms": percentile(xs, 95.0),
        "min_ms": min(xs),
        "max_ms": max(xs),
        "std_ms": math.sqrt(variance),
    }