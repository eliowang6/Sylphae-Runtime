"""Invariants for ``sylphae/model/policy.py`` and ``sylphae/model/config.py``.

The policy-level claims that matter:

* every head has the shape INTERFACES.md section 4 promises;
* ``MemState`` has ``detach`` / ``to`` / ``zeros`` and is the SSD recurrent
  state, not a decoration;
* the memory-write gate scales what persists, and a cold start is zeros;
* a single-token call with a block cache reproduces a full forward exactly --
  the end-to-end version of the streaming property.
"""

from __future__ import annotations

import pytest
import torch

from sylphae.model.collate import Batch
from sylphae.model.config import ModelConfig
from sylphae.model.policy import SylphaeModel, MemState, PolicyOutput


def tiny_cfg(**over):
    base = dict(
        d_model=32, n_layers=2, d_state=8, n_heads=2, chunk_size=8, expand=2,
        d_conv=4, vocab_size=768, max_len=64, d_ff=32,
        n_candidates=3, max_params=8, param_dec_hidden=32, param_emb_dim=16,
    )
    base.update(over)
    return ModelConfig(**base)


def make_batch(B=2, L=12, C=3, P=6, vocab=768, seed=0):
    g = torch.Generator().manual_seed(seed)
    return Batch(
        input_ids=torch.randint(7, vocab, (B, L), generator=g),
        attention_mask=torch.ones(B, L, dtype=torch.long),
        cand_index=torch.randint(0, L, (B, C), generator=g),
        cand_mask=torch.ones(B, C, dtype=torch.bool),
        op_labels=torch.randint(0, 16, (B,), generator=g),
        param_ids=torch.randint(7, vocab, (B, P), generator=g),
        param_mask=torch.ones(B, P, dtype=torch.bool),
        conf_label=torch.randint(0, 2, (B,), generator=g).float(),
        mem_write_label=torch.randint(0, 2, (B,), generator=g).float(),
    )


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


@pytest.mark.parametrize("over", [
    dict(),
    dict(d_model=48, n_layers=3, d_state=12, n_heads=3, expand=3, d_ff=96,
         param_dec_hidden=40, param_emb_dim=24),
    dict(d_model=64, n_layers=1, d_state=16, n_heads=4, expand=2, d_ff=64,
         param_dec_hidden=64, param_emb_dim=32),
])
def test_n_params_estimate_matches_the_model(over):
    cfg = ModelConfig(**{**tiny_cfg().__dict__, **over})
    model = SylphaeModel(cfg)
    assert cfg.n_params_estimate() == model.num_parameters()


def test_config_resolves_to_ssd_cfg():
    cfg = tiny_cfg()
    ssd = cfg.ssd_cfg()
    for name in ("d_model", "n_layers", "d_state", "n_heads", "chunk_size",
                 "expand", "d_conv", "vocab_size", "max_len"):
        assert getattr(ssd, name) == getattr(cfg, name)
    assert cfg.head_dim == ssd.head_dim
    assert cfg.d_conv_channels == ssd.d_conv_channels


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        ModelConfig(d_model=10, n_heads=3)
    # P was frozen at <=32; iteration 4 relaxed it to a knob bounded at 256, so
    # 64 is now accepted and only an absurd value is rejected.
    ModelConfig(max_params=64)
    with pytest.raises(ValueError):
        ModelConfig(max_params=300)
    with pytest.raises(ValueError):
        ModelConfig(dropout=1.5)
    with pytest.raises(ValueError):
        ModelConfig(n_candidates=0)


def test_config_to_dict_round_trips():
    cfg = tiny_cfg()
    d = cfg.to_dict()
    assert ModelConfig(**d).to_dict() == d


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------


def test_forward_shapes():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    model = SylphaeModel(cfg).eval()
    b = make_batch(B=2, L=12, C=3, P=6, vocab=cfg.vocab_size)
    with torch.no_grad():
        out, mem = model(b)

    assert isinstance(out, PolicyOutput)
    assert out.target_logits.shape == (2, 3)
    assert out.op_logits.shape == (2, 16)
    assert out.param_logits.shape == (2, 6, cfg.vocab_size)
    assert out.confidence.shape == (2,)
    assert out.mem_write.shape == (2,)
    assert (out.confidence >= 0).all() and (out.confidence <= 1).all()
    assert (out.mem_write >= 0).all() and (out.mem_write <= 1).all()
    assert torch.isfinite(out.target_logits).all()
    assert mem.n_layers == cfg.n_layers
    assert mem.batch_size == 2


def test_pooled_uses_the_last_non_pad_position():
    """Padding must not change the pooled representation."""
    cfg = tiny_cfg()
    torch.manual_seed(0)
    model = SylphaeModel(cfg).eval()
    b = make_batch(B=1, L=10, C=2, P=4, vocab=cfg.vocab_size)
    b.attention_mask[0, 6:] = 0
    with torch.no_grad():
        out_padded, _ = model(b)
        b2 = make_batch(B=1, L=10, C=2, P=4, vocab=cfg.vocab_size)
        b2.input_ids = b.input_ids.clone()
        b2.attention_mask = b.attention_mask.clone()
        b2.cand_index = b.cand_index.clone()
        b2.input_ids[0, 6:] = cfg.vocab_size - 1  # garbage in the padded region
        out_other, _ = model(b2)
    # op/conf/mem are read from the pooled position, so they must not move
    assert torch.allclose(out_padded.op_logits, out_other.op_logits, atol=1e-5)
    assert torch.allclose(out_padded.confidence, out_other.confidence, atol=1e-5)


def test_candidate_logits_come_from_cand_positions():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    model = SylphaeModel(cfg).eval()
    b = make_batch(B=1, L=10, C=3, P=4, vocab=cfg.vocab_size)
    with torch.no_grad():
        out, _ = model(b)
        # moving a candidate's index must change that candidate's score
        b2 = make_batch(B=1, L=10, C=3, P=4, vocab=cfg.vocab_size)
        for f in ("input_ids", "attention_mask", "param_ids", "param_mask"):
            setattr(b2, f, getattr(b, f).clone())
        b2.cand_index = b.cand_index.clone()
        b2.cand_mask = b.cand_mask.clone()
        b2.cand_index[0, 0] = (int(b.cand_index[0, 0]) + 3) % 10
        out2, _ = model(b2)
    assert not torch.allclose(out.target_logits, out2.target_logits, atol=1e-5)


def test_masked_candidates_are_never_selectable():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    model = SylphaeModel(cfg).eval()
    b = make_batch(B=1, L=10, C=3, P=4, vocab=cfg.vocab_size)
    b.cand_mask[0, 2] = False
    with torch.no_grad():
        out, _ = model(b)
    assert out.target_logits[0, 2] < out.target_logits[0, :2].min() - 1.0


# --------------------------------------------------------------------------
# MemState
# --------------------------------------------------------------------------


def test_memstate_zeros_detach_to():
    cfg = tiny_cfg()
    mem = MemState.zeros(4, cfg, device="cpu")
    assert mem.n_layers == cfg.n_layers
    assert mem.batch_size == 4
    assert mem.ssm[0].shape == (4, cfg.n_heads, cfg.head_dim, cfg.d_state)
    assert mem.ssm[0].abs().sum() == 0

    mem2 = mem.detach()
    assert all(not s.requires_grad for s in mem2.ssm)
    mem3 = mem2.to("cpu")
    assert mem3.device == torch.device("cpu")
    assert mem3.ssm[0].shape == mem.ssm[0].shape


def test_memstate_index_select():
    cfg = tiny_cfg()
    mem = MemState.zeros(4, cfg, device="cpu")
    mem = MemState([s + i for i, s in enumerate(mem.ssm)])
    sub = mem.index_select(torch.tensor([3, 1]))
    assert sub.batch_size == 2
    assert torch.equal(sub.ssm[0][0], mem.ssm[0][3])
    assert torch.equal(sub.ssm[0][1], mem.ssm[0][1])


def test_memstate_batch_mismatch_rejected():
    cfg = tiny_cfg()
    model = SylphaeModel(cfg).eval()
    b = make_batch(B=2, L=8, C=2, P=4, vocab=cfg.vocab_size)
    mem = MemState.zeros(3, cfg)
    with pytest.raises(ValueError):
        model(b, mem)


def test_memstate_layer_mismatch_rejected():
    cfg = tiny_cfg()
    other = tiny_cfg(n_layers=5)
    model = SylphaeModel(cfg).eval()
    b = make_batch(B=1, L=8, C=2, P=4, vocab=cfg.vocab_size)
    with pytest.raises(ValueError):
        model(b, MemState.zeros(1, other))


def test_memory_conditions_the_next_call():
    """A non-zero memory must change the policy's output."""
    cfg = tiny_cfg()
    torch.manual_seed(0)
    model = SylphaeModel(cfg).eval()
    b = make_batch(B=1, L=8, C=2, P=4, vocab=cfg.vocab_size)
    with torch.no_grad():
        cold, _ = model(b)
        warm_mem = MemState([torch.randn_like(s) * 0.5 for s in MemState.zeros(1, cfg).ssm])
        warm, _ = model(b, warm_mem)
    assert not torch.allclose(cold.op_logits, warm.op_logits, atol=1e-5)


def test_mem_write_gate_scales_what_persists():
    """gate=0 keeps the old memory exactly; gate=g does old + g*(raw - old)."""
    cfg = tiny_cfg()
    torch.manual_seed(0)
    model = SylphaeModel(cfg).eval()
    b = make_batch(B=1, L=8, C=2, P=4, vocab=cfg.vocab_size)

    with torch.no_grad():
        old = MemState([torch.randn_like(s) for s in MemState.zeros(1, cfg).ssm])
        # the ungated states this batch produces when seeded with `old`
        raw = model.backbone(
            b.input_ids, init_states=old.ssm, return_ssm_states=True
        )[1]
        _, kept = model(b, mem=old, mem_write_override=0.0)
        _, written = model(b, mem=old, mem_write_override=1.0)
        _, half = model(b, mem=old, mem_write_override=0.5)

    for a, c in zip(kept.ssm, old.ssm):
        assert torch.allclose(a, c, atol=0)
    for a, r in zip(written.ssm, raw):
        assert torch.allclose(a, r, atol=1e-6)
    for a, o, r in zip(half.ssm, old.ssm, raw):
        assert torch.allclose(a, o + 0.5 * (r - o), atol=1e-6)


def test_cold_start_is_zero_memory():
    """With mem=None the write gate scales the raw state from a zero start."""
    cfg = tiny_cfg()
    torch.manual_seed(0)
    model = SylphaeModel(cfg).eval()
    b = make_batch(B=1, L=8, C=2, P=4, vocab=cfg.vocab_size)
    with torch.no_grad():
        _, cold = model(b, mem=None, mem_write_override=0.0)
    assert all(s.abs().sum().item() == 0.0 for s in cold.ssm)


def test_gate_is_a_sigmoid_in_range():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    model = SylphaeModel(cfg)
    b = make_batch(B=4, L=6, C=2, P=4, vocab=cfg.vocab_size)
    out, _ = model(b)
    assert out.mem_write.min() > 0.0 and out.mem_write.max() < 1.0


# --------------------------------------------------------------------------
# streaming through the whole policy
# --------------------------------------------------------------------------


def _single_token(b, t, C):
    return Batch(
        input_ids=b.input_ids[:, t : t + 1],
        attention_mask=torch.ones(b.batch_size, 1, dtype=torch.long),
        cand_index=torch.zeros(b.batch_size, C, dtype=torch.long),
        cand_mask=torch.ones(b.batch_size, C, dtype=torch.bool),
        op_labels=b.op_labels,
        param_ids=b.param_ids,
        param_mask=b.param_mask,
        conf_label=b.conf_label,
        mem_write_label=b.mem_write_label,
    )


def test_single_token_decode_matches_the_full_forward():
    """The end-to-end streaming property: L tokens in one call == L calls of one.

    The late-interaction channel is the one exception: it scores request tokens
    against candidate-content tokens, which only exist once the whole sequence
    has been fed, so the streaming path degrades to the interact-only score.
    Everything causal -- op logits, confidence, mem write, and the interact
    component of the target score -- must still match bit for bit."""
    cfg = tiny_cfg()
    torch.manual_seed(0)
    model = SylphaeModel(cfg).double().eval()
    B, L, C, P = 1, 9, 1, 4
    b = make_batch(B=B, L=L, C=C, P=P, vocab=cfg.vocab_size, seed=1)
    b.attention_mask = torch.ones(B, L, dtype=torch.long)
    b.cand_index = torch.full((B, C), L - 1, dtype=torch.long)  # candidate at the last token

    with torch.no_grad():
        full, _ = model(b)
        cache = model.backbone.init_cache(B, b.input_ids.device, torch.float64)
        last = None
        for t in range(L):
            last, cache, _ = model.stream_step(_single_token(b, t, C), cache)
        # the interact-only reference: the channel switched off
        model.target_score.match_gate.zero_()
        interact_only, _ = model(b)

    assert torch.allclose(last.target_logits, interact_only.target_logits, atol=1e-4)
    assert not torch.allclose(full.target_logits, interact_only.target_logits,
                              atol=1e-4), "the channel must move the full score"
    assert torch.allclose(last.op_logits, full.op_logits, atol=1e-4)
    assert torch.allclose(last.confidence, full.confidence, atol=1e-4)
    assert torch.allclose(last.mem_write, full.mem_write, atol=1e-4)


def test_negative_policy_streaming_detects_a_corrupt_cache():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    model = SylphaeModel(cfg).double().eval()
    B, L, C, P = 1, 9, 1, 4
    b = make_batch(B=B, L=L, C=C, P=P, vocab=cfg.vocab_size, seed=1)
    b.cand_index = torch.full((B, C), L - 1, dtype=torch.long)

    with torch.no_grad():
        full, _ = model(b)
        cache = model.backbone.init_cache(B, b.input_ids.device, torch.float64)
        last = None
        for t in range(L):
            if t == 5:
                c0, s0 = cache[0]
                cache[0] = (c0, s0 + 1.0)
            last, cache, _ = model.stream_step(_single_token(b, t, C), cache)

    assert (last.op_logits - full.op_logits).abs().max().item() > 1e-3


def test_negative_mem_and_cache_together_rejected():
    cfg = tiny_cfg()
    model = SylphaeModel(cfg).eval()
    b = make_batch(B=1, L=1, C=1, P=4, vocab=cfg.vocab_size)
    cache = model.backbone.init_cache(1, b.input_ids.device, torch.float32)
    with pytest.raises(ValueError):
        model(b, MemState.zeros(1, cfg), cache=cache)


def test_negative_multitoken_with_cache_rejected():
    cfg = tiny_cfg()
    model = SylphaeModel(cfg).eval()
    b = make_batch(B=1, L=5, C=1, P=4, vocab=cfg.vocab_size)
    cache = model.backbone.init_cache(1, b.input_ids.device, torch.float32)
    with pytest.raises(ValueError):
        model(b, cache=cache)


# --------------------------------------------------------------------------
# training mechanics
# --------------------------------------------------------------------------


def test_loss_terms_are_differentiable():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    model = SylphaeModel(cfg)
    b = make_batch(B=2, L=10, C=3, P=5, vocab=cfg.vocab_size)
    out, _ = model(b)

    loss = (
        torch.nn.functional.cross_entropy(out.target_logits, torch.tensor([0, 1]))
        + torch.nn.functional.cross_entropy(out.op_logits, b.op_labels)
        + torch.nn.functional.cross_entropy(
            out.param_logits.reshape(-1, cfg.vocab_size),
            b.param_ids.reshape(-1),
            ignore_index=tok_pad_id(),
        )
        + torch.nn.functional.binary_cross_entropy(out.confidence, b.conf_label)
        + torch.nn.functional.binary_cross_entropy(out.mem_write, b.mem_write_label)
    )
    loss.backward()
    n_with_grad = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            n_with_grad += 1
            assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
    assert n_with_grad > 0
    # every parameter that participates in the forward must have a gradient
    assert n_with_grad == sum(1 for _ in model.parameters())


def tok_pad_id():
    return 0


def test_grad_checkpointing_matches_plain():
    cfg = tiny_cfg(grad_checkpoint=True)
    torch.manual_seed(0)
    model = SylphaeModel(cfg).train()
    b = make_batch(B=2, L=10, C=2, P=4, vocab=cfg.vocab_size)

    out, _ = model(b)
    (out.op_logits.pow(2).mean() + out.confidence.mean()).backward()
    ckpt = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    cfg2 = tiny_cfg(grad_checkpoint=False)
    torch.manual_seed(0)
    model2 = SylphaeModel(cfg2).train()
    model2.load_state_dict(model.state_dict())
    model2.zero_grad(set_to_none=True)
    out2, _ = model2(b)
    (out2.op_logits.pow(2).mean() + out2.confidence.mean()).backward()
    plain = {n: p.grad.clone() for n, p in model2.named_parameters() if p.grad is not None}

    assert ckpt.keys() == plain.keys()
    for n in plain:
        assert (ckpt[n] - plain[n]).abs().max().item() < 1e-5, n


def test_generate_params_shape():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    model = SylphaeModel(cfg).eval()
    b = make_batch(B=3, L=8, C=2, P=6, vocab=cfg.vocab_size)
    ids = model.generate_params(b)
    assert ids.shape == (3, 6)
    assert ids.min() >= 0 and ids.max() < cfg.vocab_size


def test_generate_params_stops_changing_on_repeat():
    """Greedy decoding must be deterministic."""
    cfg = tiny_cfg()
    torch.manual_seed(0)
    model = SylphaeModel(cfg).eval()
    b = make_batch(B=1, L=8, C=2, P=6, vocab=cfg.vocab_size)
    assert torch.equal(model.generate_params(b), model.generate_params(b))
