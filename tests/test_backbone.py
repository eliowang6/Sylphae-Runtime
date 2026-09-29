"""Invariants for ``sylphae/model/backbone.py``.

The important one: stepping the whole backbone one token at a time, carrying the
block cache, must reproduce a single full forward.  That is the property the
latency argument in DESIGN.md rests on, and it covers the depthwise conv, every
SSD layer, both residual branches and the final norm.
"""

from __future__ import annotations

import pytest
import torch

from sylphae.model.backbone import Mamba2Backbone, Mamba2Block, RMSNorm, SwiGLU
from sylphae.model.ssd import SSDCfg


def tiny_cfg(**over):
    base = dict(
        d_model=32, n_layers=3, d_state=16, n_heads=4, chunk_size=8,
        expand=2, d_conv=4, vocab_size=128, max_len=64,
    )
    base.update(over)
    return SSDCfg(**base)


def test_block_shapes_and_residual():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    block = Mamba2Block(cfg, d_ff=48)
    x = torch.randn(2, 9, cfg.d_model)
    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_backbone_shapes():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    bb = Mamba2Backbone(cfg, d_ff=48)
    ids = torch.randint(7, cfg.vocab_size, (2, 11))
    h = bb(ids)
    assert h.shape == (2, 11, cfg.d_model)
    assert torch.isfinite(h).all()


def test_backbone_streaming_equals_full_forward():
    """L single-token steps with a cache == one full forward."""
    cfg = tiny_cfg()
    torch.manual_seed(0)
    bb = Mamba2Backbone(cfg, d_ff=48).double().eval()
    B, L = 2, 13
    ids = torch.randint(7, cfg.vocab_size, (B, L))

    with torch.no_grad():
        full = bb(ids)

        cache = bb.init_cache(B, ids.device, torch.float64)
        steps = []
        for t in range(L):
            h, cache = bb(ids[:, t : t + 1], cache=cache)
            steps.append(h)
        streamed = torch.cat(steps, dim=1)

    assert streamed.shape == full.shape
    d = (streamed - full).abs().max().item()
    assert d < 1e-4, f"backbone streaming mismatch {d:.3e}"


def test_backbone_full_forward_cache_is_resumable():
    """A cache primed by stepping a prefix reproduces the tail of a full forward."""
    cfg = tiny_cfg()
    torch.manual_seed(3)
    bb = Mamba2Backbone(cfg, d_ff=48).double().eval()
    B = 2
    head = torch.randint(7, cfg.vocab_size, (B, 9))
    tail = torch.randint(7, cfg.vocab_size, (B, 4))

    with torch.no_grad():
        cache = bb.init_cache(B, head.device, torch.float64)
        for t in range(head.shape[1]):
            _, cache = bb(head[:, t : t + 1], cache=cache)
        steps = []
        for t in range(tail.shape[1]):
            h, cache = bb(tail[:, t : t + 1], cache=cache)
            steps.append(h)
        streamed = torch.cat(steps, dim=1)
        whole = bb(torch.cat([head, tail], dim=1))[:, head.shape[1]:]

    assert (streamed - whole).abs().max().item() < 1e-4


def test_memory_state_conditions_the_forward():
    """A non-zero initial SSD state must change the output."""
    cfg = tiny_cfg()
    torch.manual_seed(0)
    bb = Mamba2Backbone(cfg, d_ff=48).eval()
    ids = torch.randint(7, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        h0 = bb(ids)
        mem = [
            torch.randn(1, cfg.n_heads, cfg.head_dim, cfg.d_state) * 0.5
            for _ in range(cfg.n_layers)
        ]
        h1 = bb(ids, init_states=mem)
    assert (h0 - h1).abs().max().item() > 1e-3


def test_backward_through_backbone():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    bb = Mamba2Backbone(cfg, d_ff=48)
    ids = torch.randint(7, cfg.vocab_size, (2, 10))
    out = bb(ids)
    out.pow(2).mean().backward()
    for name, p in bb.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_gradient_checkpointing_matches_plain_backward():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    bb = Mamba2Backbone(cfg, d_ff=48).train()
    ids = torch.randint(7, cfg.vocab_size, (2, 10))

    bb.zero_grad(set_to_none=True)
    bb(ids).pow(2).mean().backward()
    plain = {n: p.grad.clone() for n, p in bb.named_parameters() if p.grad is not None}

    bb.zero_grad(set_to_none=True)
    bb(ids, use_checkpoint=True).pow(2).mean().backward()
    ckpt = {n: p.grad.clone() for n, p in bb.named_parameters() if p.grad is not None}

    assert plain.keys() == ckpt.keys()
    for n in plain:
        d = (plain[n] - ckpt[n]).abs().max().item()
        assert d < 1e-6, f"checkpointed grad differs for {n}: {d:.3e}"


# --------------------------------------------------------------------------
# negative
# --------------------------------------------------------------------------


def test_negative_streaming_detects_a_wrong_cache():
    """Corrupting the carried state must break streaming equivalence."""
    cfg = tiny_cfg()
    torch.manual_seed(0)
    bb = Mamba2Backbone(cfg, d_ff=48).double().eval()
    B, L = 1, 9
    ids = torch.randint(7, cfg.vocab_size, (B, L))

    with torch.no_grad():
        full = bb(ids)
        cache = bb.init_cache(B, ids.device, torch.float64)
        steps = []
        for t in range(L):
            if t == 4:  # corrupt layer 0's carried ssm state mid-stream
                c0, s0 = cache[0]
                cache[0] = (c0, s0 + 1.0)
            h, cache = bb(ids[:, t : t + 1], cache=cache)
            steps.append(h)
        streamed = torch.cat(steps, dim=1)

    assert (streamed - full).abs().max().item() > 1e-3


def test_negative_wrong_init_states_length_rejected():
    cfg = tiny_cfg()
    bb = Mamba2Backbone(cfg, d_ff=48)
    ids = torch.randint(7, cfg.vocab_size, (1, 4))
    with pytest.raises(ValueError):
        bb(ids, init_states=[torch.zeros(1, cfg.n_heads, cfg.head_dim, cfg.d_state)])


def test_negative_multitoken_streaming_rejected():
    cfg = tiny_cfg()
    bb = Mamba2Backbone(cfg, d_ff=48)
    ids = torch.randint(7, cfg.vocab_size, (1, 4))
    cache = bb.init_cache(1, ids.device, torch.float32)
    with pytest.raises(ValueError):
        bb(ids, cache=cache)


def test_norms_and_mlp_shapes():
    n = RMSNorm(8)
    x = torch.randn(3, 5, 8)
    assert n(x).shape == x.shape
    mlp = SwiGLU(8, 16)
    assert mlp(x).shape == x.shape
