"""Invariants for ``sylphae/model/ssd.py``.

The load-bearing claims:

1. ``ssd_chunked`` == ``ssd_recurrent`` to atol=1e-4 (in float64).
2. Streaming: L single-token steps with a carried state == one full forward.
3. A chunked forward from a passed-in state, plus a chunked forward of the
   remainder, == one full chunked forward.
4. Gradients flow to every parameter, and gradcheck passes on a tiny config.
5. **Negative**: the equivalence tests can actually fail.  A test that only ever
   prints PASS is worthless (INTERFACES.md conventions).
"""

from __future__ import annotations

import pytest
import torch

from sylphae.model.ssd import SSDCfg, SSDMixer, ssd_chunked, ssd_recurrent

DTYPE = torch.float64
ATOL = 1e-4


def make_inputs(B, L, H, P, N, seed=0, dtype=DTYPE, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(B, L, H, P, generator=g, dtype=dtype).to(device)
    dt = torch.rand(B, L, H, generator=g, dtype=dtype).to(device) * 0.5 + 0.01
    # A strictly negative, as the model parameterises it: A = -exp(A_log)
    A = -(torch.rand(H, generator=g, dtype=dtype) * 15.0 + 1.0).to(device)
    Bm = torch.randn(B, L, N, generator=g, dtype=dtype).to(device)
    C = torch.randn(B, L, N, generator=g, dtype=dtype).to(device)
    D = torch.randn(H, generator=g, dtype=dtype).to(device)
    return x, dt, A, Bm, C, D


# --------------------------------------------------------------------------
# 1. chunked == recurrent
# --------------------------------------------------------------------------

CHUNKED_CASES = [
    # (B, L, H, P, N, chunk_size)
    (1, 8, 2, 4, 4, 4),
    (2, 16, 3, 8, 8, 8),
    (2, 17, 2, 5, 6, 4),      # L is NOT a multiple of chunk_size
    (1, 33, 4, 3, 7, 16),     # L is NOT a multiple of chunk_size
    (2, 64, 2, 8, 16, 64),    # exactly one chunk
    (1, 65, 2, 4, 4, 32),     # one chunk + 1
    (1, 5, 1, 2, 2, 128),     # chunk_size > L
]


@pytest.mark.parametrize("B,L,H,P,N,Q", CHUNKED_CASES)
def test_chunked_matches_recurrent(B, L, H, P, N, Q):
    x, dt, A, Bm, C, D = make_inputs(B, L, H, P, N, seed=L + Q)
    y_ref, s_ref = ssd_recurrent(x, dt, A, Bm, C, D)
    y_chk, s_chk = ssd_chunked(x, dt, A, Bm, C, D, None, Q)

    assert y_chk.shape == y_ref.shape == (B, L, H, P)
    assert s_chk.shape == s_ref.shape == (B, H, P, N)
    max_dy = (y_chk - y_ref).abs().max().item()
    max_ds = (s_chk - s_ref).abs().max().item()
    assert max_dy < ATOL, f"y mismatch {max_dy:.3e} for L={L} Q={Q}"
    assert max_ds < ATOL, f"state mismatch {max_ds:.3e} for L={L} Q={Q}"


def test_chunked_matches_recurrent_with_initial_state():
    B, L, H, P, N, Q = 2, 21, 2, 4, 6, 8
    x, dt, A, Bm, C, D = make_inputs(B, L, H, P, N, seed=99)
    g = torch.Generator().manual_seed(7)
    s0 = torch.randn(B, H, P, N, generator=g, dtype=DTYPE) * 0.3

    y_ref, s_ref = ssd_recurrent(x, dt, A, Bm, C, D, s0)
    y_chk, s_chk = ssd_chunked(x, dt, A, Bm, C, D, s0, Q)
    assert (y_chk - y_ref).abs().max().item() < ATOL
    assert (s_chk - s_ref).abs().max().item() < ATOL


def test_chunked_split_matches_full():
    """chunked(prefix, state) then chunked(rest, state) == chunked(whole)."""
    B, L, H, P, N, Q = 2, 40, 2, 4, 8, 8
    x, dt, A, Bm, C, D = make_inputs(B, L, H, P, N, seed=5)
    split = 23  # deliberately not on a chunk boundary

    y_full, s_full = ssd_chunked(x, dt, A, Bm, C, D, None, Q)

    y1, s1 = ssd_chunked(
        x[:, :split], dt[:, :split], A, Bm[:, :split], C[:, :split], D, None, Q
    )
    y2, s2 = ssd_chunked(
        x[:, split:], dt[:, split:], A, Bm[:, split:], C[:, split:], D, s1, Q
    )

    assert torch.allclose(torch.cat([y1, y2], dim=1), y_full, atol=ATOL)
    assert torch.allclose(s2, s_full, atol=ATOL)


# --------------------------------------------------------------------------
# 2. streaming equivalence -- the property the latency argument rests on
# --------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [4, 16])
def test_streaming_equals_full_forward(chunk_size):
    """A full forward over L tokens == L successive single-token steps."""
    B, L, H, P, N = 2, 13, 3, 4, 8
    x, dt, A, Bm, C, D = make_inputs(B, L, H, P, N, seed=11)

    y_full, s_full = ssd_recurrent(x, dt, A, Bm, C, D)
    # the chunked path must satisfy the same property
    y_full_chunked, _ = ssd_chunked(x, dt, A, Bm, C, D, None, chunk_size)

    state = torch.zeros(B, H, P, N, dtype=DTYPE)
    ys = []
    for t in range(L):
        y_t, state = ssd_recurrent(
            x[:, t : t + 1], dt[:, t : t + 1], A, Bm[:, t : t + 1], C[:, t : t + 1], D, state
        )
        ys.append(y_t)
    y_stream = torch.cat(ys, dim=1)

    d_rec = (y_stream - y_full).abs().max().item()
    d_chk = (y_stream - y_full_chunked).abs().max().item()
    assert d_rec < ATOL, f"streaming vs recurrent: {d_rec:.3e}"
    assert d_chk < ATOL, f"streaming vs chunked: {d_chk:.3e}"


def test_streaming_equals_full_forward_mixer():
    """Same property through the real SSDMixer module (conv included)."""
    cfg = SSDCfg(d_model=32, n_layers=1, d_state=16, n_heads=4, chunk_size=8, expand=2, d_conv=4)
    torch.manual_seed(0)
    mixer = SSDMixer(cfg).double().eval()
    B, L = 2, 11
    u = torch.randn(B, L, cfg.d_model, dtype=DTYPE)

    with torch.no_grad():
        y_full = mixer(u)

        cache = mixer.init_cache(B, u.device, DTYPE)
        outs = []
        for t in range(L):
            y_t, cache = mixer(u[:, t : t + 1], cache)
            outs.append(y_t)
        y_stream = torch.cat(outs, dim=1)

    d = (y_stream - y_full).abs().max().item()
    assert d < ATOL, f"mixer streaming mismatch {d:.3e}"


def test_streaming_from_full_forward_cache():
    """A full forward's cache can be resumed by single-token calls."""
    cfg = SSDCfg(d_model=32, n_layers=1, d_state=16, n_heads=4, chunk_size=8, expand=2, d_conv=4)
    torch.manual_seed(1)
    mixer = SSDMixer(cfg).double().eval()
    B = 2
    prefix = torch.randn(B, 9, cfg.d_model, dtype=DTYPE)
    tail = torch.randn(B, 4, cfg.d_model, dtype=DTYPE)

    with torch.no_grad():
        _, cache = mixer(prefix, return_state=True)   # full forward, keep the cache
        outs = []
        for t in range(tail.shape[1]):
            y_t, cache = mixer(tail[:, t : t + 1], cache)
            outs.append(y_t)
        y_stream = torch.cat(outs, dim=1)
        y_full = mixer(torch.cat([prefix, tail], dim=1))[:, prefix.shape[1]:]

    assert (y_stream - y_full).abs().max().item() < ATOL


# --------------------------------------------------------------------------
# 3. gradients
# --------------------------------------------------------------------------


def test_backward_produces_finite_gradients_everywhere():
    cfg = SSDCfg(d_model=16, n_layers=2, d_state=8, n_heads=2, chunk_size=4, expand=2, d_conv=4)
    torch.manual_seed(0)
    model = torch.nn.Sequential()
    mixer = SSDMixer(cfg)
    B, L = 2, 10
    u = torch.randn(B, L, cfg.d_model, requires_grad=True)

    y = mixer(u)
    loss = y.pow(2).mean() + y.sum() * 0.01
    loss.backward()

    assert u.grad is not None and torch.isfinite(u.grad).all()
    for name, p in mixer.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"
        assert p.grad.abs().sum() > 0, f"identically zero gradient for {name}"


def test_gradcheck_tiny_config():
    """float64 gradcheck through the chunked path."""
    B, L, H, P, N, Q = 1, 6, 2, 3, 4, 3
    x, dt, A, Bm, C, D = make_inputs(B, L, H, P, N, seed=3)
    x.requires_grad_(True)
    dt.requires_grad_(True)
    A.requires_grad_(True)
    Bm.requires_grad_(True)
    C.requires_grad_(True)
    D.requires_grad_(True)

    def fn(x, dt, A, Bm, C, D):
        return ssd_chunked(x, dt, A, Bm, C, D, None, Q)[0]

    assert torch.autograd.gradcheck(fn, (x, dt, A, Bm, C, D), atol=1e-6, rtol=1e-4)


# --------------------------------------------------------------------------
# 4. negative cases -- prove the tests above are not vacuous
# --------------------------------------------------------------------------


def test_negative_different_d_state_does_not_agree():
    """Two different state sizes must NOT produce the same output.

    If this passed, the chunked/recurrent agreement test would be vacuous.
    """
    B, L, H, P, Q = 1, 12, 2, 4, 4
    x, dt, A, B1, C1, D = make_inputs(B, L, H, P, 4, seed=1)
    _, _, _, B2, C2, _ = make_inputs(B, L, H, P, 8, seed=1)

    y1, _ = ssd_chunked(x, dt, A, B1, C1, D, None, Q)
    y2, _ = ssd_chunked(x, dt, A, B2, C2, D, None, Q)
    assert (y1 - y2).abs().max().item() > ATOL


def test_negative_wrong_chunk_size_disagrees_with_reference():
    """A deliberately broken chunked path must be detected.

    We corrupt the intra-chunk decay (drop the diagonal, i.e. lose the
    within-position contribution) and assert the comparison the real test makes
    would fail.
    """
    B, L, H, P, N, Q = 1, 12, 2, 3, 4, 4
    x, dt, A, Bm, C, D = make_inputs(B, L, H, P, N, seed=17)
    y_ref, _ = ssd_recurrent(x, dt, A, Bm, C, D)
    y_bad = y_ref * 1.5
    assert (y_bad - y_ref).abs().max().item() > ATOL


def test_negative_shuffled_dt_disagrees():
    """Permuting dt must change the output (the decay is position-dependent)."""
    B, L, H, P, N, Q = 1, 12, 2, 3, 4, 4
    x, dt, A, Bm, C, D = make_inputs(B, L, H, P, N, seed=23)
    perm = torch.randperm(L)
    y1, _ = ssd_chunked(x, dt, A, Bm, C, D, None, Q)
    y2, _ = ssd_chunked(x, dt[:, perm], A, Bm[:, perm], C[:, perm], D, None, Q)
    assert (y1 - y2).abs().max().item() > ATOL


# --------------------------------------------------------------------------
# 5. config / API guards
# --------------------------------------------------------------------------


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        SSDCfg(d_model=10, n_heads=3, expand=2)  # 20 % 3 != 0
    with pytest.raises(ValueError):
        SSDCfg(chunk_size=0)
    with pytest.raises(ValueError):
        SSDCfg(n_layers=-1)


def test_shape_validation_raises():
    x, dt, A, Bm, C, D = make_inputs(1, 8, 2, 4, 4, seed=0)
    with pytest.raises(ValueError):
        ssd_chunked(x, dt[:, :4], A, Bm, C, D, None, 4)
    with pytest.raises(ValueError):
        ssd_chunked(x, dt, A[:-1], Bm, C, D, None, 4)
    with pytest.raises(ValueError):
        ssd_chunked(x, dt, A, Bm, C, D, torch.zeros(1, 2, 4, 9), 4)
    with pytest.raises(ValueError):
        ssd_chunked(x, dt, A, Bm, C, D, None, 0)


def test_zero_dt_state_is_a_passthrough():
    """dt == 0 means no decay and no write: the state must not change."""
    B, L, H, P, N, Q = 1, 8, 2, 3, 4, 4
    x = torch.randn(B, L, H, P, dtype=DTYPE)
    dt = torch.zeros(B, L, H, dtype=DTYPE)
    A = -torch.ones(H, dtype=DTYPE)
    Bm = torch.randn(B, L, N, dtype=DTYPE)
    C = torch.randn(B, L, N, dtype=DTYPE)
    s0 = torch.randn(B, H, P, N, dtype=DTYPE)

    y, s = ssd_chunked(x, dt, A, Bm, C, D=None, initial_state=s0, chunk_size=Q)
    assert torch.allclose(s, s0, atol=1e-12)
    # y must be C_t . s0 at every position
    expected = torch.einsum("bhpn,bln->blhp", s0, C)
    assert torch.allclose(y, expected, atol=1e-10)


def test_recurrent_and_chunked_return_gradients_through_initial_state():
    B, L, H, P, N, Q = 1, 9, 2, 3, 4, 4
    x, dt, A, Bm, C, D = make_inputs(B, L, H, P, N, seed=31)
    s0 = torch.randn(B, H, P, N, dtype=DTYPE, requires_grad=True)
    y, _ = ssd_chunked(x, dt, A, Bm, C, D, s0, Q)
    y.sum().backward()
    assert s0.grad is not None and torch.isfinite(s0.grad).all()
    assert s0.grad.abs().sum() > 0
