"""Tests for ``detect_sequence`` / ``apply_sequence`` — the decomposition.

``detect_op`` forces a whole file change to be *one* op.  ``detect_sequence``
decomposes it per hunk and returns an ordered list.  The invariant this file
exists to protect:

    for any returned sequence whose ops are not all ``freeform_patch``,
    ``apply_sequence(old, seq) == new`` exactly.

The sequence is emitted **top-to-bottom** but each op's span is in the
**original** source, so it must be applied **back-to-front**.  The tests below
pin that contract down, including for sequences whose ops touch adjacent lines.
"""

from __future__ import annotations

import functools
import json

import pytest

from sylphae.ops import (
    OpSpec,
    apply_op,
    apply_sequence,
    detect_op,
    detect_sequence,
)
from sylphae.ops.vocab import FREE_FORM_KIND

PATH = "pkg/mod.py"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _is_structured(seq: list[OpSpec]) -> bool:
    return bool(seq) and all(spec.kind != FREE_FORM_KIND for spec in seq)


def _assert_round_trip(old: str, new: str) -> list[OpSpec]:
    seq = detect_sequence(old, new, PATH)
    assert seq, "a real change must never yield an empty sequence"
    if _is_structured(seq):
        assert apply_sequence(old, seq) == new, (
            f"sequence replay diverged: {[s.kind for s in seq]}"
        )
        # The literal wording of the invariant: back-to-front reduction.
        assert functools.reduce(apply_op, reversed(seq), old) == new
    else:
        assert len(seq) == 1 and seq[0].kind == FREE_FORM_KIND
        with pytest.raises(Exception):
            apply_op(old, seq[0])
    return seq


# --------------------------------------------------------------------------- #
# degenerate cases
# --------------------------------------------------------------------------- #


def test_identical_sources_return_empty_sequence():
    for src in ("x = 1\n", "", "\n\n\n", "def f(a, b=2):\n    return a + b\n"):
        assert detect_sequence(src, src, PATH) == []


def test_detect_op_is_the_length_one_special_case():
    """Whenever ``detect_op`` finds a structured op, the sequence is just it."""
    cases = [
        ("TIMEOUT = 30\nRETRIES = 3\n", "TIMEOUT = 60\nRETRIES = 3\n"),
        ("import os\n", "import json\nimport os\n"),
        ("def f():\n    return len(x)\n", "def f():\n    return len(x) + 1\n"),
    ]
    for old, new in cases:
        spec = detect_op(old, new, PATH)
        assert spec is not None and spec.kind != FREE_FORM_KIND
        assert detect_sequence(old, new, PATH) == [spec]


def test_undescribable_change_falls_back_to_one_freeform():
    # A comment-only insertion.  ``insert_statement`` needs the inserted lines to
    # tile complete statements and a bare comment is not one, so no decomposition
    # reaches it -- which is the point: the fallback must stay reachable.
    seq = detect_sequence("x = 1\n", "x = 1\n# a note\n", PATH)
    assert len(seq) == 1 and seq[0].kind == FREE_FORM_KIND


def test_growing_replace_is_split_into_a_replacement_and_an_insertion():
    """A one-line-to-two-lines replace is two ops, not one undescribable change.

    This is the shape a JSON / object-literal member addition arrives in: the
    previous line has to gain its terminator *and* a line has to be inserted, and
    ``difflib`` reports both as a single growing ``replace``.  No single op can
    express that, so the sequence decomposer peels the first line pair and
    explains the tail as an insertion.
    """
    old = "def f(x):\n    return x\n"
    new = "def f(x):\n    y = x * 2\n    return y\n"
    seq = _assert_round_trip(old, new)
    assert [s.kind for s in seq] == ["replace_line", "insert_statement"]
    assert seq[0].props["old_text"] == "    return x\n"
    assert seq[1].props["text"] == "    return y\n"


# --------------------------------------------------------------------------- #
# sequence lengths 2, 3 and 5
# --------------------------------------------------------------------------- #


def _numbered(n: int, edits: dict[int, str]) -> tuple[str, str]:
    old = "".join(f"v{i} = {i}\n" for i in range(n))
    new = old
    for i, value in edits.items():
        new = new.replace(f"v{i} = {i}\n", f"v{i} = {value}\n")
    return old, new


def test_sequence_of_length_2():
    old, new = _numbered(20, {2: 100, 15: 200})
    seq = _assert_round_trip(old, new)
    assert _is_structured(seq) and len(seq) == 2


def test_sequence_of_length_3():
    old, new = _numbered(20, {1: 100, 10: 200, 18: 300})
    seq = _assert_round_trip(old, new)
    assert _is_structured(seq) and len(seq) == 3


def test_sequence_of_length_5():
    old, new = _numbered(24, {1: 11, 6: 22, 11: 33, 16: 44, 21: 55})
    seq = _assert_round_trip(old, new)
    assert _is_structured(seq) and len(seq) == 5


def test_sequence_touching_adjacent_lines():
    """Two edits with no equal line between them must still be two ops."""
    old = "a = 1\nb = 2\nc = 3\n"
    new = "a = 10\nb = 20\nc = 3\n"
    seq = _assert_round_trip(old, new)
    assert _is_structured(seq) and len(seq) == 2
    assert [s.span for s in seq] == [(1, 1), (2, 2)]


def test_sequence_touching_three_adjacent_lines():
    old = "a = 1\nb = 2\nc = 3\n"
    new = "a = 10\nb = 20\nc = 30\n"
    seq = _assert_round_trip(old, new)
    assert _is_structured(seq) and len(seq) == 3
    assert [s.span for s in seq] == [(1, 1), (2, 2), (3, 3)]


def test_sequence_mixes_kinds():
    old = "import os\n\nX = 1\n\nmsg = 'hi'\n\ndef f():\n    return X\n"
    new = "import os\nimport sys\n\nX = 2\n\nmsg = 'bye'\n\ndef f():\n    return X\n"
    seq = _assert_round_trip(old, new)
    assert _is_structured(seq)
    assert [s.kind for s in seq] == ["add_import", "replace_literal", "modify_string"]


def test_sequence_with_a_wrap_and_a_literal():
    old = "def f():\n    for x in y:\n        print(x)\n\nTIMEOUT = 30\n"
    new = "def f():\n    with ctx():\n        for x in y:\n            print(x)\n\nTIMEOUT = 60\n"
    seq = _assert_round_trip(old, new)
    assert _is_structured(seq)
    assert [s.kind for s in seq] == ["wrap_block", "replace_literal"]


def test_sequence_is_emitted_top_to_bottom():
    old, new = _numbered(20, {2: 100, 15: 200})
    seq = _assert_round_trip(old, new)
    starts = [s.span[0] for s in seq]
    assert starts == sorted(starts), "ops must be emitted in ascending line order"


def test_sequence_is_deterministic():
    old, new = _numbered(20, {1: 100, 10: 200, 18: 300})
    assert detect_sequence(old, new, PATH) == detect_sequence(old, new, PATH)


# --------------------------------------------------------------------------- #
# adversarial / negative
# --------------------------------------------------------------------------- #


def test_a_broken_sequence_is_rejected_rather_than_returned(monkeypatch):
    """If replay does not reproduce ``new``, ``detect_sequence`` must not emit it.

    This is the mechanism the whole decomposition rests on: a wrong sequence is a
    lie the model would learn.  Sabotaging ``apply_sequence`` must degrade
    coverage (to one freeform), not corrupt the label.
    """
    from sylphae.ops import detect as detect_mod

    old, new = _numbered(20, {2: 100, 15: 200})
    assert _is_structured(detect_sequence(old, new, PATH))

    def sabotaged(_old_src, _seq):
        return _old_src  # wrong on purpose

    monkeypatch.setattr(detect_mod, "apply_sequence", sabotaged)
    seq = detect_sequence(old, new, PATH)
    assert len(seq) == 1 and seq[0].kind == FREE_FORM_KIND


def test_apply_sequence_rejects_a_freeform_spec():
    seq = detect_sequence("x = 1\n", "x = 1\n# a note\n", PATH)
    assert seq[0].kind == FREE_FORM_KIND
    with pytest.raises(Exception):
        apply_sequence("x = 1\n", seq)


def test_apply_sequence_does_not_clamp_a_bad_span():
    bad = OpSpec("replace_line", PATH, (99, 99), {"line": "99",
                                                  "old_text": "z\n", "new_text": "y\n"})
    with pytest.raises(Exception):
        apply_sequence("a = 1\n", [bad])


def test_sequence_specs_survive_json_round_trip():
    old, new = _numbered(20, {1: 100, 10: 200, 18: 300})
    seq = _assert_round_trip(old, new)
    revived = [OpSpec.from_dict(json.loads(json.dumps(s.to_dict()))) for s in seq]
    assert revived == seq
    assert apply_sequence(old, revived) == new


def test_sequence_ops_all_carry_original_source_spans():
    """Every span must be valid in the *original* source, not an intermediate."""
    old, new = _numbered(20, {1: 100, 10: 200, 18: 300})
    seq = _assert_round_trip(old, new)
    n_old = len(old.splitlines())
    for spec in seq:
        a, b = spec.span
        assert 1 <= a <= b <= n_old + 1
