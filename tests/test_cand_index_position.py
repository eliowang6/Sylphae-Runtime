"""The target head must score a candidate AFTER reading that candidate's path.

``cand_index[b, c]`` used to point at the opening ``<cand>``.  The SSD recurrent
state at that position has consumed the request and the preceding candidates but
NOT this candidate's path tokens, which follow the marker.  A head gathering the
hidden state there cannot see what it is scoring, so the only learnable signal is
the label marginal -- and because ``candidate_permutation`` is uniform, that
marginal is ``ln(k)``.  Training measured exactly this: ``L_target`` sat at 2.05
~= ln(8) for every one of 2000 steps while ``L_op`` fell 1.65 -> 0.89.

The fix is one character of index arithmetic; this test pins it down so it cannot
silently come back.  A reverted ``_encode_one`` makes the assertion fail.

Nothing here needs a trained artifact: the tokenizer is the session-scoped
``tiny_tokenizer`` fixture (a real byte-level BPE fitted on local stdlib
sources), and the candidate pool comes from the shipped default index provider.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sylphae.model.collate import _encode_one, collate  # noqa: E402
from sylphae.model.config import ModelConfig  # noqa: E402


def _example() -> dict:
    return {
        "ex_id": "x1",
        "repo": "acme/widget",
        "request": "increase the default request timeout",
        "target_path": "src/widget/core.py",
        "repo_files": [
            "src/widget/core.py",
            "src/widget/cli.py",
            "src/widget/utils.py",
            "tests/test_core.py",
        ],
        "old_source": "timeout = 5\n",
        "new_source": "timeout = 30\n",
        "op": {"kind": "replace_literal", "props": {}},
    }


def test_cand_index_points_past_the_candidate_path(tiny_tokenizer):
    tok = tiny_tokenizer
    ex = _example()
    cfg = ModelConfig(d_model=64, n_layers=1, n_candidates=4, max_len=512, ctx_tokens=0)

    seq, cand_index, cand_mask, _gold = _encode_one(ex, tok, None, 4, cfg.max_len, cfg.ctx_tokens)

    assert cand_mask and all(cand_mask), "every retrieved candidate must be scored"
    for pos in cand_index:
        assert seq[pos] == tok.endcand_id, (
            f"cand_index {pos} must be the closing </cand> (token {tok.endcand_id}), "
            f"got {seq[pos]} -- the head would score before reading the path"
        )


def test_cand_index_is_not_the_opening_marker(tiny_tokenizer):
    """The regression guard: the old code recorded the position BEFORE the span."""
    tok = tiny_tokenizer
    ex = _example()
    cfg = ModelConfig(d_model=64, n_layers=1, n_candidates=4, max_len=512, ctx_tokens=0)

    seq, cand_index, _, _gold = _encode_one(ex, tok, None, 4, cfg.max_len, cfg.ctx_tokens)

    for pos in cand_index:
        assert seq[pos] != tok.cand_id, (
            "cand_index points at the opening <cand>; the SSD state there has not "
            "yet read this candidate's path, so the head cannot use content"
        )
        # the path tokens must sit strictly between an opening marker and this
        # position -- otherwise the index is not inside a candidate span at all
        opened = max(i for i, t in enumerate(seq[: pos + 1]) if t == tok.cand_id)
        assert opened < pos, "no opening <cand> precedes the scoring position"


def test_collate_batch_carries_the_closing_positions(tiny_tokenizer):
    """End-to-end: the Batch the model gathers from holds </cand> positions."""
    tok = tiny_tokenizer
    ex = _example()
    cfg = ModelConfig(d_model=64, n_layers=1, n_candidates=4, max_len=512, ctx_tokens=0)

    batch = collate([ex] * 2, tok, None, cfg)

    assert batch.cand_index.shape == (2, 4)
    for b in range(2):
        for c in range(4):
            if batch.cand_mask[b, c]:
                pos = int(batch.cand_index[b, c])
                assert int(batch.input_ids[b, pos]) == tok.endcand_id


def test_positions_are_strictly_increasing_and_in_bounds(tiny_tokenizer):
    tok = tiny_tokenizer
    ex = _example()
    cfg = ModelConfig(d_model=64, n_layers=1, n_candidates=4, max_len=256, ctx_tokens=64)

    seq, cand_index, cand_mask, _gold = _encode_one(ex, tok, None, 4, cfg.max_len, cfg.ctx_tokens)

    assert [p for p, m in zip(cand_index, cand_mask) if m] == sorted(
        p for p, m in zip(cand_index, cand_mask) if m
    )
    for pos, keep in zip(cand_index, cand_mask):
        if keep:
            assert 0 <= pos < len(seq)
