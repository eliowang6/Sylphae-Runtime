"""``apply_op`` must reject, never coerce — and ``OpSpec`` must reject bad props."""

from __future__ import annotations

import json

import pytest

from sylphae.ops import (
    ID_TO_OP,
    N_OPS,
    OP_KINDS,
    OP_TO_ID,
    OpApplyError,
    OpSpec,
    apply_op,
    detect_op,
)

PATH = "pkg/mod.py"
SRC = "a = 1\nb = 2\nc = 3\n"


# --------------------------------------------------------------------------- #
# vocabulary
# --------------------------------------------------------------------------- #


def test_vocab_is_frozen():
    # The vocabulary grew from 16 to 26 when ``detect_sequence`` landed.  Ids
    # 0..15 are unchanged (the corpus and the model op head index into them);
    # the ten new kinds are appended at 16..25.
    assert N_OPS == 26
    assert len(OP_KINDS) == N_OPS == len(set(OP_KINDS))
    assert OP_KINDS[0] == "no_op"
    assert OP_KINDS[15] == "freeform_patch"
    assert OP_KINDS[16:] == [
        "replace_line",
        "add_call_arg",
        "remove_call_arg",
        "add_dict_entry",
        "remove_dict_entry",
        "wrap_block",
        "modify_string_multi",
        "reindent",
        "add_list_element",
        "remove_list_element",
    ]
    for i, kind in enumerate(OP_KINDS):
        assert OP_TO_ID[kind] == i
        assert ID_TO_OP[i] == kind


# --------------------------------------------------------------------------- #
# span handling: raise, never clamp
# --------------------------------------------------------------------------- #


def _literal_spec(span, old_text="a = 1\n", new_text="a = 9\n"):
    return OpSpec(
        "replace_literal",
        PATH,
        span,
        {"line": str(span[0]), "old_text": old_text, "new_text": new_text,
         "old": "1", "new": "9"},
    )


@pytest.mark.parametrize(
    "span",
    [(1, 4), (4, 5), (1, 99)],
)
def test_out_of_range_replace_span_raises(span):
    with pytest.raises(OpApplyError):
        apply_op(SRC, _literal_spec(span))


@pytest.mark.parametrize("span", [(0, 1), (2, 1), (-1, 1), (3, 2)])
def test_structurally_invalid_span_is_rejected_at_construction(span):
    """``1 <= a <= b`` is checked where the spec is built, not where it is applied."""
    with pytest.raises(ValueError):
        OpSpec("replace_literal", PATH, span, {})


def test_out_of_range_span_is_not_clamped():
    """The specific failure mode we care about: silently applying to line 1."""
    spec = _literal_spec((99, 99))
    with pytest.raises(OpApplyError):
        apply_op(SRC, spec)


@pytest.mark.parametrize("point", [5, 99])
def test_out_of_range_insertion_point_raises(point):
    spec = OpSpec(
        "insert_statement",
        PATH,
        (point, point),
        {"line": str(point), "text": "d = 4\n", "n_stmts": "1"},
    )
    with pytest.raises(OpApplyError):
        apply_op(SRC, spec)


def test_insertion_point_at_end_is_legal():
    spec = OpSpec(
        "insert_statement",
        PATH,
        (4, 4),
        {"line": "4", "text": "d = 4\n", "n_stmts": "1"},
    )
    assert apply_op(SRC, spec) == SRC + "d = 4\n"


def test_delete_span_out_of_range_raises():
    spec = OpSpec("delete_statement", PATH, (3, 9), {"text": "c = 3\n", "n_stmts": "1"})
    with pytest.raises(OpApplyError):
        apply_op(SRC, spec)


# --------------------------------------------------------------------------- #
# recorded text must match the source it is applied to
# --------------------------------------------------------------------------- #


def test_stale_old_text_is_rejected():
    """A spec built from a different revision must not be applied to the wrong line."""
    spec = _literal_spec((2, 2), old_text="z = 9\n", new_text="z = 8\n")
    with pytest.raises(OpApplyError):
        apply_op(SRC, spec)


def test_delete_with_wrong_text_is_rejected():
    spec = OpSpec("delete_statement", PATH, (2, 2), {"text": "zzz\n", "n_stmts": "1"})
    with pytest.raises(OpApplyError):
        apply_op(SRC, spec)


def test_rename_with_wrong_occurrence_count_is_rejected():
    spec = OpSpec("rename_symbol", PATH, (1, 1), {"old": "a", "new": "z", "count": "5"})
    with pytest.raises(OpApplyError):
        apply_op(SRC, spec)


def test_rename_to_the_same_name_is_rejected():
    spec = OpSpec("rename_symbol", PATH, (1, 1), {"old": "a", "new": "a", "count": "1"})
    with pytest.raises(OpApplyError):
        apply_op(SRC, spec)


def test_missing_prop_is_rejected():
    spec = OpSpec("replace_literal", PATH, (1, 1), {"line": "1"})
    with pytest.raises(OpApplyError):
        apply_op(SRC, spec)


def test_span_prop_mismatch_is_rejected():
    spec = OpSpec(
        "insert_statement", PATH, (2, 2), {"line": "3", "text": "d = 4\n", "n_stmts": "1"}
    )
    with pytest.raises(OpApplyError):
        apply_op(SRC, spec)


# --------------------------------------------------------------------------- #
# freeform is not replayable
# --------------------------------------------------------------------------- #


def test_freeform_patch_raises_instead_of_guessing():
    spec = OpSpec(
        "freeform_patch", PATH, (1, 1), {"reason": "x", "n_regions": "1",
                                         "old_sha256": "0" * 64, "new_sha256": "1" * 64}
    )
    with pytest.raises(OpApplyError):
        apply_op(SRC, spec)


def test_no_op_is_identity_and_still_range_checked():
    spec = OpSpec("no_op", PATH, (1, 1), {})
    assert apply_op(SRC, spec) == SRC
    with pytest.raises(OpApplyError):
        apply_op(SRC, OpSpec("no_op", PATH, (9, 9), {}))


def test_unknown_kind_never_reaches_apply_op():
    with pytest.raises(ValueError):
        OpSpec("teleport_file", PATH, (1, 1), {})


# --------------------------------------------------------------------------- #
# OpSpec validation
# --------------------------------------------------------------------------- #


def test_non_string_props_are_rejected_not_coerced():
    with pytest.raises(ValueError):
        OpSpec("replace_literal", PATH, (1, 1), {"old": 30, "new": "60"})
    with pytest.raises(ValueError):
        OpSpec("replace_literal", PATH, (1, 1), {7: "30"})


def test_bad_spans_are_rejected():
    for span in ((0, 1), (3, 1), (1,), "1,2", (1.0, 2.0)):
        with pytest.raises(ValueError):
            OpSpec("replace_literal", PATH, span, {})


def test_empty_path_is_rejected():
    with pytest.raises(ValueError):
        OpSpec("replace_literal", "", (1, 1), {})


def test_spec_json_round_trip_preserves_types():
    spec = detect_op("x = 30\n", "x = 60\n", PATH)
    assert spec is not None
    revived = OpSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert revived == spec
    assert isinstance(revived.span, tuple)
    assert all(isinstance(v, str) for v in revived.props.values())


def test_from_dict_rejects_malformed_input():
    with pytest.raises(ValueError):
        OpSpec.from_dict({"kind": "replace_literal", "path": PATH, "span": [1, 1]})
    with pytest.raises(ValueError):
        OpSpec.from_dict("not a dict")


def test_detect_op_rejects_bad_arguments():
    with pytest.raises(TypeError):
        detect_op(b"x = 1\n", "x = 2\n", PATH)
    with pytest.raises(ValueError):
        detect_op("x = 1\n", "x = 2\n", "")
