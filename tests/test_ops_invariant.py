"""Tests that exist to *fail* if the round-trip invariant is ever violated.

The corpus test in ``test_ops_roundtrip.py`` asserts the invariant directly.  This
module adds two independent reasons it cannot pass vacuously:

* a deterministic fuzz over mutated real-ish sources, which catches detectors that
  only happen to work on the hand-written examples;
* a sabotage test that breaks an applier on purpose and requires ``detect_op`` to
  stop emitting that structured kind — proving the invariant is enforced by
  *replay*, not by trusting the detectors.

Together they mean the round-trip test above can only pass if the invariant
actually holds.
"""

from __future__ import annotations

import random

import pytest

from sylphae.ops import apply_op, detect_op
from sylphae.ops import apply as apply_mod
from sylphae.ops import detect as detect_mod

PATH = "pkg/mod.py"

BASE_SOURCES = [
    "import os\nimport sys\n\n\nTIMEOUT = 30\n\n\ndef read(path):\n    return open(path).read()\n",
    (
        "def connect(host, port=5432, timeout=30):\n"
        "    if host is None:\n"
        '        raise ValueError("host is required")\n'
        "    return (host, port, timeout)\n"
    ),
    (
        "class Client:\n"
        "    def __init__(self, base_url):\n"
        "        self.base_url = base_url\n"
        "\n"
        "    def get(self, path):\n"
        '        return request("GET", self.base_url + path)\n'
    ),
    (
        "def parse(raw):\n"
        "    try:\n"
        "        return int(raw)\n"
        "    except ValueError:\n"
        "        return 0\n"
    ),
]

SPARE_LINES = [
    "x = 1\n",
    "import json\n",
    "    return None\n",
    "    if x is None:\n",
    "        raise RuntimeError('nope')\n",
    "def helper(a, b=2):\n",
    "# a comment\n",
    "    pass\n",
]


def _mutations(src: str, rng: random.Random, n: int):
    lines = src.splitlines(keepends=True)
    out = []
    for _ in range(n):
        cur = list(lines)
        op = rng.randrange(5)
        if not cur:
            break
        i = rng.randrange(len(cur))
        if op == 0:
            del cur[i]
        elif op == 1:
            cur.insert(i, cur[i])
        elif op == 2:
            cur.insert(i, rng.choice(SPARE_LINES))
        elif op == 3:
            cur[i] = rng.choice(SPARE_LINES)
        else:
            j = rng.randrange(len(cur))
            cur[i], cur[j] = cur[j], cur[i]
        out.append("".join(cur))
    return out


def test_fuzz_round_trip_invariant():
    rng = random.Random(20260921)
    checked = 0
    structured = 0
    for base in BASE_SOURCES:
        for mutated in _mutations(base, rng, 120):
            if mutated == base:
                continue
            spec = detect_op(base, mutated, PATH)
            assert spec is not None, "a real change must never yield None"
            checked += 1
            if spec.kind == "freeform_patch":
                continue
            structured += 1
            assert apply_op(base, spec) == mutated, (
                f"round-trip violated: kind={spec.kind} span={spec.span} "
                f"props={spec.props!r}"
            )
    assert checked >= 400
    # A fuzz run that finds no structured edit at all would be a vacuous pass.
    assert structured > 0, "fuzz produced no structured ops; the test proves nothing"


def test_apply_op_never_returns_the_original_source():
    """The classic silent sabotage: an applier that just returns ``old_src``.

    It would make a naive ``apply_op(...) == new`` check fail loudly, but a
    detector that trusted itself would still emit a structured label.  Assert the
    appliers actually do something.
    """
    rng = random.Random(7)
    seen = 0
    for base in BASE_SOURCES:
        for mutated in _mutations(base, rng, 40):
            if mutated == base:
                continue
            spec = detect_op(base, mutated, PATH)
            if spec is None or spec.kind == "freeform_patch":
                continue
            seen += 1
            assert apply_op(base, spec) != base
    assert seen > 0


def test_sabotaged_applier_makes_detect_op_fall_back(monkeypatch):
    """Break an applier; ``detect_op`` must refuse the kind, not emit a bad spec.

    This is the mechanism the whole module rests on.  If ``detect_op`` trusted its
    detectors instead of replaying them, the sabotage below would leak a
    ``replace_literal`` spec whose replay is ``old_src``, and
    ``test_round_trip_is_exact`` in ``test_ops_roundtrip.py`` would go red.
    """
    old = "TIMEOUT = 30\nRETRIES = 3\n"
    new = "TIMEOUT = 60\nRETRIES = 3\n"

    healthy = detect_op(old, new, PATH)
    assert healthy is not None and healthy.kind == "replace_literal"

    def sabotaged(_old_src, _spec):
        return _old_src  # wrong on purpose

    # `detect.py` binds `apply_op` at import time, so patch it there.
    monkeypatch.setattr(detect_mod, "apply_op", sabotaged)

    broken = detect_op(old, new, PATH)
    assert broken is not None
    assert broken.kind == "freeform_patch", (
        "detect_op emitted a structured spec that does not replay; "
        "the round-trip invariant is not being enforced"
    )
    # The real applier is untouched, so a healthy call still labels it.
    assert apply_mod.apply_op(old, healthy) == new


@pytest.mark.parametrize("base", BASE_SOURCES, ids=range(len(BASE_SOURCES)))
def test_detect_op_is_deterministic(base):
    rng = random.Random(99)
    for mutated in _mutations(base, rng, 30):
        first = detect_op(base, mutated, PATH)
        second = detect_op(base, mutated, PATH)
        assert first == second
