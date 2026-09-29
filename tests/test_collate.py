"""Invariants for ``sylphae/model/collate.py``.

The two things that matter: ``cand_index`` must actually point at the ``<cand>``
tokens, and the label functions must be pure functions of their input list.
"""

from __future__ import annotations

import torch

from sylphae.model.collate import (
    N_OPS,
    OP_TO_ID,
    Batch,
    collate,
    conf_labels,
    default_index_provider,
    mem_write_labels,
    param_answerable,
    param_target_string,
)
from sylphae.model.config import ModelConfig
from sylphae.model.tokenizer import CAND_ID, ENDCAND_ID


def make_example(i=0, kind="replace_literal", repo="a/b", request="set timeout to 60",
                 target="pkg/mod.py", repo_files=None, props=None, new_source=None):
    old_source = "TIMEOUT = 30\n"
    return {
        "ex_id": f"test:{i}",
        "source": "test",
        "repo": repo,
        "commit": "0" * 40,
        "request": request,
        "target_path": target,
        "old_source": old_source,
        "new_source": new_source if new_source is not None else "TIMEOUT = 60\n",
        "op": {
            "kind": kind,
            "path": target,
            "span": [1, 1],
            "props": props if props is not None else {
                "old_text": old_source, "new_text": "TIMEOUT = 60\n",
            },
        },
        "repo_files": repo_files if repo_files is not None else [
            target, "pkg/other.py", "pkg/util.py", "tests/test_mod.py",
        ],
        "test_paths": ["tests/test_mod.py"],
        "split": "train",
    }


def tiny_cfg(**over):
    base = dict(
        d_model=32, n_layers=1, d_state=8, n_heads=2, chunk_size=8,
        expand=2, d_conv=4, vocab_size=768, max_len=128, d_ff=32,
        n_candidates=4, max_params=32,
    )
    base.update(over)
    return ModelConfig(**base)


def test_batch_shapes(tiny_tokenizer):
    cfg = tiny_cfg()
    examples = [make_example(i) for i in range(3)]
    b = collate(examples, tiny_tokenizer, None, cfg)

    assert isinstance(b, Batch)
    B, L = b.input_ids.shape
    assert B == 3
    assert L <= cfg.max_len
    assert b.attention_mask.shape == (B, L)
    assert b.cand_index.shape == (B, cfg.n_candidates)
    assert b.cand_mask.shape == (B, cfg.n_candidates)
    assert b.cand_mask.dtype == torch.bool
    assert b.op_labels.shape == (B,)
    assert b.param_ids.shape[0] == B and b.param_ids.shape[1] <= cfg.max_params
    assert b.param_mask.shape == b.param_ids.shape
    assert b.conf_label.shape == (B,)
    assert b.mem_write_label.shape == (B,)


def test_cand_index_points_at_cand_tokens(tiny_tokenizer):
    cfg = tiny_cfg()
    examples = [make_example(i, target=f"pkg/mod{i}.py") for i in range(3)]
    b = collate(examples, tiny_tokenizer, None, cfg)

    for i in range(b.batch_size):
        real = b.cand_mask[i].nonzero().flatten().tolist()
        assert real, "every example must keep at least one candidate"
        for c in real:
            pos = int(b.cand_index[i, c])
            # The head scores the CLOSING </cand>: the SSD state at the opening
            # marker has not yet read this candidate's path, so a head gathering
            # there cannot see what it scores and can only learn the label
            # marginal (ln k under the uniform permutation).  See
            # tests/test_cand_index_position.py.
            assert int(b.input_ids[i, pos]) == ENDCAND_ID, (
                f"cand_index[{i},{c}]={pos} does not point at </cand>"
            )
            # the path sits between the opening marker and this position
            head = b.input_ids[i, : pos + 1].tolist()
            assert head.count(CAND_ID) >= 1, "no opening <cand> precedes the score position"


def test_sequence_starts_with_bos_and_contains_sep(tiny_tokenizer):
    cfg = tiny_cfg()
    b = collate([make_example(0)], tiny_tokenizer, None, cfg)
    assert int(b.input_ids[0, 0]) == tiny_tokenizer.bos_id
    assert tiny_tokenizer.sep_id in b.input_ids[0].tolist()


def test_padding_and_attention_mask(tiny_tokenizer):
    cfg = tiny_cfg()
    long_ex = make_example(0, request="a very long request " * 30)
    short_ex = make_example(1, request="short")
    b = collate([long_ex, short_ex], tiny_tokenizer, None, cfg)

    for i in range(b.batch_size):
        n = int(b.attention_mask[i].sum())
        assert n > 0
        assert b.attention_mask[i, :n].min() == 1
        assert b.attention_mask[i, n:].max() == 0 if n < b.seq_len else True
        assert (b.input_ids[i, n:] == tiny_tokenizer.pad_id).all()


def test_target_is_not_always_first(tiny_tokenizer):
    """A positional shortcut would make the target head meaningless."""
    cfg = tiny_cfg(n_candidates=4)
    positions = set()
    for i in range(30):
        ex = make_example(i, target="pkg/mod.py", repo=f"repo{i}")
        b = collate([ex], tiny_tokenizer, None, cfg)
        # which slot holds the target path?  decode each candidate segment
        idx = int(b.cand_index[0, 0])
        # find which candidate slot has the target by checking the index provider
        cands = default_index_provider(ex, 4)
        positions.add(next(j for j, c in enumerate(cands) if c.path == "pkg/mod.py"))
    assert len(positions) > 1, f"target always in the same slot: {positions}"


def test_collate_is_deterministic(tiny_tokenizer):
    cfg = tiny_cfg()
    examples = [make_example(i) for i in range(4)]
    b1 = collate(examples, tiny_tokenizer, None, cfg)
    b2 = collate(examples, tiny_tokenizer, None, cfg)
    for f in b1.__dict__:
        assert torch.equal(getattr(b1, f), getattr(b2, f)), f"{f} not deterministic"


def test_op_labels(tiny_tokenizer):
    cfg = tiny_cfg()
    examples = [
        make_example(0, kind="replace_literal"),
        make_example(1, kind="freeform_patch"),
        make_example(2, kind="rename_symbol"),
    ]
    b = collate(examples, tiny_tokenizer, None, cfg)
    assert b.op_labels.tolist() == [
        OP_TO_ID["replace_literal"], OP_TO_ID["freeform_patch"], OP_TO_ID["rename_symbol"],
    ]
    assert N_OPS == 26
    assert len(OP_TO_ID) == 26


def test_param_ids_encode_the_target_string(tiny_tokenizer):
    cfg = tiny_cfg()
    # The request must name the value: under param_answerable_only a value absent
    # from the request is unanswerable and is masked out of L_param rather than
    # supervised (12.12).  "set timeout to 5" is answerable; "set timeout to 60"
    # would not be, and the mask is what says so.
    ex = make_example(0, request="set timeout to 5", props={"old": "1", "new": "5"})
    # The target is the single new value; the eval decoder reconstructs the rest.
    target = param_target_string(ex)
    assert target == "5"
    assert param_answerable(ex)
    b = collate([ex], tiny_tokenizer, None, cfg)
    n = int(b.param_mask[0].sum())
    assert n >= 1
    decoded = tiny_tokenizer.decode(b.param_ids[0, :n].tolist())
    assert decoded == "5"


def test_param_mask_removes_unanswerable_supervision(tiny_tokenizer):
    """A value the request does not name is masked, not supervised.

    This is the whole of the 12.12 fix: ~74% of parameterised rows carry a value
    that appears nowhere in the request, and supervising them taught the head the
    old-source prior instead of a request->value copy.  The row keeps its target /
    op / conf labels -- only the parameter emission is declared unlearnable.
    """
    cfg = tiny_cfg()
    answerable = make_example(0, request="rename foo to bar",
                              props={"old": "foo", "new": "bar"})
    unanswerable = make_example(1, request="shorten the timeout a bit",
                                props={"old": "60", "new": "30"})
    assert param_answerable(answerable)
    assert not param_answerable(unanswerable)

    b = collate([answerable, unanswerable], tiny_tokenizer, None, cfg)
    assert int(b.param_mask[0].sum()) >= 1
    assert int(b.param_mask[1].sum()) == 0


def test_param_target_string_is_the_single_new_value():
    """The parameter target is ONE value, not JSON.

    Iteration 5 measured that a small decoder cannot emit well-formed JSON props
    (59/59 held-out payloads came back undecodable), but it can choose a short value;
    ``default_decode_spec`` is tolerant and reconstructs the full props from it.
    """
    assert param_target_string(
        {"op": {"kind": "rename_symbol", "props": {"old": "foo", "new": "bar", "count": "2"}}}
    ) == "bar"
    assert param_target_string(
        {"op": {"kind": "insert_statement", "props": {"line": "4", "text": "x = 1\n"}}}
    ) == "x = 1\n"
    assert param_target_string(
        {"op": {"kind": "delete_statement", "props": {"text": "x = 1\n"}}}) == "x = 1\n"
    assert param_target_string({"op": {"kind": "no_op", "props": {}}}) == ""
    assert param_target_string({"op": {"kind": "freeform_patch", "props": {}}}) == ""
    assert param_target_string({"op": {}}) == ""


def test_param_ids_truncated_at_max_params(tiny_tokenizer):
    cfg = tiny_cfg(max_params=8)
    # The request must name the value: under param_answerable_only a value absent
    # from the request is masked, and this test is about truncation, not masking.
    value = "z" * 400
    ex = make_example(0, request=f"set the marker to {value}",
                      props={"old": "a", "new": value})
    assert param_answerable(ex)
    b = collate([ex], tiny_tokenizer, None, cfg)
    assert b.param_ids.shape[1] <= 8
    assert b.param_mask[0].all(), "a truncated param must still fill every slot"


def test_conf_labels_structured_vs_freeform(tiny_tokenizer):
    examples = [
        make_example(0, kind="replace_literal"),
        make_example(1, kind="freeform_patch"),
        make_example(2, kind="no_op", new_source="TIMEOUT = 30\n"),  # unchanged
    ]
    assert conf_labels(examples) == [1.0, 0.0, 1.0]
    cfg = tiny_cfg()
    b = collate(examples, tiny_tokenizer, None, cfg)
    assert b.conf_label.tolist() == [1.0, 0.0, 1.0]


def test_conf_labels_rejects_a_lying_no_op():
    """no_op claims nothing changed; if the source did change it is not exact."""
    ex = make_example(0, kind="no_op")  # old != new
    assert conf_labels([ex]) == [0.0]


def test_conf_labels_verifies_the_round_trip_when_ops_is_available():
    """A structured kind whose props do not reproduce new_source is NOT exact."""
    good = make_example(0, kind="replace_literal")
    bad = make_example(1, kind="replace_literal",
                       props={"old_text": "TIMEOUT = 30\n", "new_text": "WRONG\n"})
    assert conf_labels([good, bad]) == [1.0, 0.0]


def test_mem_write_labels_counts_per_repo():
    examples = [
        make_example(0, kind="replace_literal", repo="r1"),
        make_example(1, kind="replace_literal", repo="r1"),   # 2nd in r1 -> 1.0
        make_example(2, kind="rename_symbol", repo="r1"),     # 1st in r1 -> 0.0
        make_example(3, kind="replace_literal", repo="r2"),   # 1st in r2 -> 0.0
        make_example(4, kind="rename_symbol", repo="r2"),
        make_example(5, kind="rename_symbol", repo="r2"),     # 2nd+ -> 1.0
    ]
    assert mem_write_labels(examples) == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]


def test_mem_write_labels_is_a_pure_function_of_the_list():
    examples = [make_example(i, kind="replace_literal", repo="r") for i in range(3)]
    assert mem_write_labels(examples) == mem_write_labels(examples)
    # order must not matter
    rev = list(reversed(examples))
    assert mem_write_labels(examples) == mem_write_labels(rev)
    # a single-example list has no repetition
    assert mem_write_labels([examples[0]]) == [0.0]


def test_precomputed_mem_write_label_wins(tiny_tokenizer):
    cfg = tiny_cfg()
    ex = make_example(0)
    ex["mem_write_label"] = 1.0
    b = collate([ex], tiny_tokenizer, None, cfg)
    assert b.mem_write_label.tolist() == [1.0]


def test_batch_to_device(tiny_tokenizer):
    cfg = tiny_cfg()
    b = collate([make_example(0)], tiny_tokenizer, None, cfg)
    b2 = b.to("cpu")
    assert torch.equal(b2.input_ids, b.input_ids)


# --------------------------------------------------------------------------
# negative
# --------------------------------------------------------------------------


def test_negative_wrong_kind_maps_to_freeform(tiny_tokenizer):
    cfg = tiny_cfg()
    ex = make_example(0, kind="not_a_real_op")
    b = collate([ex], tiny_tokenizer, None, cfg)
    assert b.op_labels.tolist() == [OP_TO_ID["freeform_patch"]]
    assert b.conf_label.tolist() == [0.0]


def test_negative_empty_examples_rejected(tiny_tokenizer):
    import pytest

    with pytest.raises(ValueError):
        collate([], tiny_tokenizer, None, tiny_cfg())


def test_negative_tiny_max_len_rejected(tiny_tokenizer):
    import pytest

    with pytest.raises(ValueError):
        collate([make_example(0)], tiny_tokenizer, None, tiny_cfg(max_len=4))


def test_max_params_widened_but_bounded():
    """P is now a knob (32 truncates the median props payload), not a hard <=32.

    Iteration 4 measured that the props-JSON the parameter head must emit has
    median 50 tokens, so a 32-wide cap truncated ~78% of payloads and made them
    undecodable.  A wider value (64) must be accepted; the field is still bounded,
    so an absurd value is rejected.
    """
    import pytest

    tiny_cfg(max_params=64)          # no longer rejected
    tiny_cfg(max_params=256)         # the bound itself is allowed
    with pytest.raises(ValueError):
        tiny_cfg(max_params=257)


def test_negative_bad_index_provider_rejected(tiny_tokenizer):
    import pytest

    with pytest.raises(TypeError):
        collate([make_example(0)], tiny_tokenizer, 12345, tiny_cfg())
