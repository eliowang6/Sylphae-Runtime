"""Token-level late interaction (MaxSim) -- contract tests.

Two contracts are pinned here:

1. ``_max_sim`` must be an explicit set-intersection score: identical token sets
   score +1, disjoint sets score ~0, and padding is inert (never wins a max,
   never penalises an empty candidate).
2. ``_token_grids`` must recover exactly the request span and each candidate's
   content span from a real collated batch -- the walk back from ``</cand>`` to
   the *nearest* ``<cand>`` is what makes per-candidate spans correct, and a
   first-marker bug would silently score every candidate against candidate 0.
"""
from __future__ import annotations

import torch

from sylphae.model.collate import Batch
from sylphae.model.policy import SylphaeModel, TargetScoreHead
from sylphae.model.tokenizer import (CAND_ID, ENDCAND_ID, SEP_ID)
from tests.test_collate import tiny_cfg

BOS, PAD = 1, 0


def _batch(*rows: list[int], n_cand: int, L: int) -> Batch:
    """Build a Batch whose ids spell out ``<bos> req <sep> <cand> .. </cand>``."""
    out = torch.full((len(rows), L), PAD, dtype=torch.long)
    attn = torch.zeros(len(rows), L, dtype=torch.long)
    for i, row in enumerate(rows):
        out[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        attn[i, : len(row)] = 1
    cand_index = torch.zeros(len(rows), n_cand, dtype=torch.long)
    for i, row in enumerate(rows):
        closes = [j for j, t in enumerate(row) if t == ENDCAND_ID]
        for c, j in enumerate(closes[:n_cand]):
            cand_index[i, c] = j
    return Batch(
        input_ids=out,
        attention_mask=attn,
        cand_index=cand_index,
        cand_mask=torch.ones(len(rows), n_cand, dtype=torch.bool),
        op_labels=torch.zeros(len(rows), dtype=torch.long),
        param_ids=torch.zeros(len(rows), 2, dtype=torch.long),
        param_mask=torch.ones(len(rows), 2, dtype=torch.bool),
        conf_label=torch.zeros(len(rows)),
        mem_write_label=torch.zeros(len(rows)),
    )


def _one_hot_grids(model: SylphaeModel, batch: Batch) -> tuple:
    """Feed an index map so gathered values *are* positions."""
    B, L = batch.input_ids.shape
    d = model.cfg.d_model
    index_map = torch.arange(L, dtype=torch.float32).view(1, L, 1).expand(B, L, d)
    with torch.no_grad():
        return model._token_grids(batch, index_map)


# -- max_sim -----------------------------------------------------------------


def _norm(t: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(t, dim=-1)


def test_max_sim_identical_sets_score_one() -> None:
    d = 8
    v = _norm(torch.randn(4, d))                     # shared token set
    rq = v.unsqueeze(0).expand(2, 4, d).clone()      # (B, Lq, d)
    cd = v.unsqueeze(0).unsqueeze(0).expand(2, 1, 4, d).clone()
    ms = TargetScoreHead._max_sim(rq, torch.ones(2, 4, dtype=torch.bool),
                                  cd, torch.ones(2, 1, 4, dtype=torch.bool))
    assert torch.allclose(ms, torch.ones(2, 1), atol=1e-5)


def test_max_sim_disjoint_sets_score_zero() -> None:
    d = 4
    rq = torch.eye(d)[:2].unsqueeze(0).expand(1, 2, d)      # e0, e1
    cd = torch.eye(d)[2:].unsqueeze(0).unsqueeze(0)          # e2, e3
    ms = TargetScoreHead._max_sim(rq, torch.ones(1, 2, dtype=torch.bool),
                                  cd, torch.ones(1, 1, 2, dtype=torch.bool))
    assert torch.allclose(ms, torch.zeros(1, 1), atol=1e-6)


def test_max_sim_empty_candidate_is_neutral_not_negative() -> None:
    d = 4
    rq = torch.randn(1, 3, d)
    cd = torch.randn(1, 2, 5, d)
    cd_mask = torch.zeros(1, 2, 5, dtype=torch.bool)
    cd_mask[:, 0, :3] = True                          # candidate 1 fully padded
    ms = TargetScoreHead._max_sim(rq, torch.ones(1, 3, dtype=torch.bool),
                                  cd, cd_mask)
    assert torch.isclose(ms[0, 1], torch.tensor(0.0), atol=1e-6), \
        "a candidate with no live tokens must score the neutral 0, not -1"


def test_max_sim_masked_candidate_token_never_wins_max() -> None:
    """A masked token that is cos=+1 with the request must not win the max."""
    d = 2
    rq = _norm(torch.tensor([[[1.0, 0.0]]]))
    # token 0 is live but only ~0.994 similar; token 1 is cos=+1 but masked
    cd = _norm(torch.tensor([[[[0.9, 0.1], [1.0, 0.0]]]]))    # (1,1,2,d)
    cd_mask = torch.tensor([[[True, False]]])
    ms = TargetScoreHead._max_sim(rq, torch.ones(1, 1, dtype=torch.bool),
                                  cd, cd_mask)
    assert ms[0, 0].item() > 0.9
    assert ms[0, 0].item() < 1.0 - 1e-3, \
        "the masked cos=+1 token must not win the max"


def test_max_sim_all_masked_request_scores_zero() -> None:
    d = 4
    rq = torch.randn(1, 3, d)
    cd = torch.randn(1, 1, 2, d)
    ms = TargetScoreHead._max_sim(rq, torch.zeros(1, 3, dtype=torch.bool),
                                  cd, torch.ones(1, 1, 2, dtype=torch.bool))
    assert torch.allclose(ms, torch.zeros(1, 1), atol=1e-6)


def test_max_sim_masked_request_tokens_dont_change_live_mean() -> None:
    """Masking request tokens keeps the mean over the live ones: the masked
    ones contribute exactly 0, so the score is the live-token mean."""
    d = 4
    rq = _norm(torch.randn(1, 4, d))
    cd = _norm(torch.randn(1, 1, 2, d))
    live = torch.tensor([[True, True, False, False]])
    part = TargetScoreHead._max_sim(rq, live, cd,
                                    torch.ones(1, 1, 2, dtype=torch.bool))
    # exact contract: part == mean over the two live request tokens
    sim = torch.einsum("qk,cmk->cqm", rq[0], cd[0]).max(-1).values
    expected = sim[0, :2].mean()
    assert torch.isclose(part[0, 0], expected, atol=1e-5)


# -- token grids -------------------------------------------------------------


def test_token_grids_recovers_request_and_candidate_spans() -> None:
    cfg = tiny_cfg()
    model = SylphaeModel(cfg)
    # <bos> a b c <sep> <cand> d e </cand> <cand> f g h </cand>
    row = [BOS, 10, 11, 12, SEP_ID, CAND_ID, 20, 21, ENDCAND_ID,
           CAND_ID, 30, 31, 32, ENDCAND_ID]
    batch = _batch(row, n_cand=2, L=len(row))

    rq_pos, rq_mask, cd_pos, cd_mask = _one_hot_grids(model, batch)

    got_req = rq_pos[0][rq_mask[0]][..., 0].long().tolist()
    assert got_req == [1, 2, 3], f"request span is <bos> .. first <sep>, got {got_req}"

    got_c0 = cd_pos[0, 0][cd_mask[0, 0]][..., 0].long().tolist()
    got_c1 = cd_pos[0, 1][cd_mask[0, 1]][..., 0].long().tolist()
    assert got_c0 == [6, 7], f"candidate 0 content excludes both markers, got {got_c0}"
    assert got_c1 == [10, 11, 12], f"candidate 1 must not reuse candidate 0's span, got {got_c1}"


def test_token_grains_empty_spans_do_not_crash() -> None:
    cfg = tiny_cfg()
    model = SylphaeModel(cfg)
    # a request with <sep> but candidates fully padded (cand_mask all False)
    row = [BOS, 10, SEP_ID]
    batch = _batch(row, n_cand=2, L=8)
    rq_pos, rq_mask, cd_pos, cd_mask = _one_hot_grids(model, batch)

    got_req = rq_pos[0][rq_mask[0]][..., 0].long().tolist()
    assert got_req == [1]
    assert not cd_mask[0].any(), "no <cand> markers means no candidate content"


def test_heads_without_sep_falls_back_cleanly() -> None:
    """streaming sees a single token with no <sep>/<cand> -- must not crash."""
    cfg = tiny_cfg()
    model = SylphaeModel(cfg)
    model.eval()
    batch = _batch([BOS, 10], n_cand=2, L=4)
    with torch.no_grad():
        out = model.forward(batch)[0]
    assert out.target_logits.shape == (1, 2)
    assert torch.isfinite(out.target_logits).all()


def test_forward_matches_streaming_when_grids_absent() -> None:
    """Without token grids the head degrades to the plain interaction path."""
    cfg = tiny_cfg()
    model = SylphaeModel(cfg)
    model.eval()
    head = model.target_score
    # the gate is initialised open, so both paths already differ by
    # construction; force it to 1.0 anyway so the test stays exact
    with torch.no_grad():
        head.match_gate.fill_(1.0)
    h_req = torch.randn(2, cfg.d_model)
    h_cand = torch.randn(2, 3, cfg.d_model)
    with torch.no_grad():
        plain = head(h_req, h_cand)
        with_grids = head(h_req, h_cand,
                          torch.randn(2, 4, cfg.d_model),
                          torch.ones(2, 4, dtype=torch.bool),
                          torch.randn(2, 3, 5, cfg.d_model),
                          torch.ones(2, 3, 5, dtype=torch.bool))
    assert plain.shape == (2, 3)
    assert with_grids.shape == (2, 3)
    assert torch.isfinite(plain).all() and torch.isfinite(with_grids).all()
    assert not torch.allclose(plain, with_grids), "grids must add a channel"


def test_zero_gate_makes_max_sim_inert() -> None:
    """``match_gate=0`` switches the channel off: score is then identical with
    and without token grids.  The gate is initialised open (not 0) precisely
    because a 0 gate zeroes the projection gradients and the channel can never
    learn from scratch; this test pins the off-switch property itself."""
    cfg = tiny_cfg()
    model = SylphaeModel(cfg)
    model.eval()
    head = model.target_score
    with torch.no_grad():
        head.match_gate.zero_()
    h_req = torch.randn(2, cfg.d_model)
    h_cand = torch.randn(2, 3, cfg.d_model)
    with torch.no_grad():
        plain = head(h_req, h_cand)
        with_grids = head(h_req, h_cand,
                          torch.randn(2, 4, cfg.d_model),
                          torch.ones(2, 4, dtype=torch.bool),
                          torch.randn(2, 3, 5, cfg.d_model),
                          torch.ones(2, 3, 5, dtype=torch.bool))
    assert torch.allclose(plain, with_grids, atol=1e-6)


def test_gate_open_at_init_lets_grids_move_the_score() -> None:
    """The default init must be open, not 0, or the channel is untrainable
    from scratch (see the cold-start note on ``match_gate``)."""
    cfg = tiny_cfg()
    model = SylphaeModel(cfg)
    model.eval()
    head = model.target_score
    h_req = torch.randn(2, cfg.d_model)
    h_cand = torch.randn(2, 3, cfg.d_model)
    with torch.no_grad():
        plain = head(h_req, h_cand)
        with_grids = head(h_req, h_cand,
                          torch.randn(2, 4, cfg.d_model),
                          torch.ones(2, 4, dtype=torch.bool),
                          torch.randn(2, 3, 5, cfg.d_model),
                          torch.ones(2, 3, 5, dtype=torch.bool))
    assert torch.isfinite(with_grids).all()
    assert not torch.allclose(plain, with_grids, atol=1e-6), \
        "an open gate must let the token grids change the score"
