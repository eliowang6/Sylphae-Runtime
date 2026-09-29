"""Mamba-2 SSD (state space duality) sequence mixer -- pure PyTorch.

Why this file exists
--------------------
DESIGN.md section 2.3: we implement SSD ourselves rather than depending on
``mamba-ssm``.  A fused CUDA extension is a build-flag lottery on Windows and
would put the critical path behind something we do not control.  The chunked
kernel below is ~150 lines, numerically transparent, and testable against a
naive recurrent reference that is obviously correct.

The two implementations
-----------------------
``ssd_recurrent``
    The literal recurrence, one Python step per position.  O(L) Python loop.
    Slow, obviously correct, and the *semantic ground truth*: every claim the
    chunked path makes is checked against this one.

``ssd_chunked``
    The chunked semiseparable-matmul form used for training.  O(L/chunk_size)
    Python loop (a scan over chunk boundaries, not over positions) and matmuls
    everywhere else.

Math
----
Per head ``h``, per head-dimension ``p``, per state dimension ``n``, with a
scalar-per-head decay ``A[h] < 0`` (the Mamba-2 "A is a scalar times identity"
simplification) and a per-position positive step ``dt[t, h]``:

    h_t[p, n] = exp(dt_t A) h_{t-1}[p, n] + dt_t * x_t[p] * B_t[n]
    y_t[p]    = sum_n C_t[n] h_t[p, n] + D[p] x_t[p]

Unrolling, with the cumulative step ``c_i = sum_{u<=i} dt_u``:

    decay(j -> i) = exp(A (c_i - c_j))                 for j <= i
    h_i           = sum_{j<=i} decay(j->i) dt_j x_j (outer) B_j

Because ``dt`` is a scalar per head rather than per channel, the inner sum
factorises into a difference of two cumulative sums, and ``decay(j->i)``
becomes an outer product of a function of ``i`` and a function of ``j``.  That
factorisation is exactly what turns the quadratic attention-like form

    y_intra_i = sum_{j<=i} exp(A(c_i - c_j)) dt_j (C_i . B_j) x_j

into one masked matmul per chunk.  It is the "state space duality": the same
operator is a linear recurrence and a semiseparable matrix.

Numerical notes
---------------
* ``A(c_i - c_j) <= 0`` for ``j <= i`` (A is negative, cumulative sums are
  non-decreasing), so every exponential we materialise is in ``(0, 1]``.  The
  masked upper triangle is filled with ``-inf`` *before* ``exp`` so we never
  evaluate ``exp(positive)``.
* The cumulative sums are taken *within* a chunk, so the exponents are bounded
  by the chunk's total decay rather than by the whole sequence length.
* ``dt`` is produced by ``softplus`` and is therefore strictly positive.  Zero
  ``dt`` is only ever produced by explicit right-padding, where ``B`` is also
  zero, so the padded region contributes nothing to the state.

Tensor layout
-------------
``x``  : (B, L, H, P)      -- ``H * P == expand * d_model``
``dt`` : (B, L, H)
``A``  : (H,)              -- strictly negative
``B``  : (B, L, N)
``C``  : (B, L, N)
``D``  : (H,) or None      -- skip connection, applied per head dimension
state  : (B, H, P, N)

``B`` and ``C`` are shared across heads (``n_groups == 1``, the Mamba-2
default and the only setting ``SSDCfg`` exposes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SSDCfg",
    "ssd_recurrent",
    "ssd_chunked",
    "SSDMixer",
    "RMSNormGated",
    "Mamba2Block",
]


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


@dataclass
class SSDCfg:
    """Backbone geometry.  Frozen by INTERFACES.md section 4."""

    d_model: int = 1024
    n_layers: int = 24
    d_state: int = 128
    n_heads: int = 16
    chunk_size: int = 64
    expand: int = 2
    d_conv: int = 4
    vocab_size: int = 32768
    max_len: int = 2048

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise on impossible geometry.  Never coerce."""
        for name in ("d_model", "n_layers", "d_state", "n_heads", "chunk_size",
                     "expand", "d_conv", "vocab_size", "max_len"):
            v = getattr(self, name)
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"SSDCfg.{name} must be a positive int, got {v!r}")
        if self.d_conv < 1:
            raise ValueError("SSDCfg.d_conv must be >= 1")
        if self.d_inner % self.n_heads != 0:
            raise ValueError(
                f"expand * d_model ({self.d_inner}) must be divisible by n_heads "
                f"({self.n_heads}); change d_model, expand or n_heads."
            )
        if self.head_dim % 1 != 0:
            raise ValueError("head_dim must be an integer")

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def head_dim(self) -> int:
        """P -- channels per head."""
        return self.d_inner // self.n_heads

    @property
    def d_conv_channels(self) -> int:
        """Channels the depthwise conv runs over: x and B and C concatenated."""
        return self.d_inner + 2 * self.d_state


# --------------------------------------------------------------------------
# shared validation / helpers
# --------------------------------------------------------------------------


def _check_ssd_inputs(x, dt, A, B, C, D, initial_state):
    if x.dim() != 4:
        raise ValueError(f"x must be (B, L, H, P), got shape {tuple(x.shape)}")
    if dt.dim() != 3:
        raise ValueError(f"dt must be (B, L, H), got shape {tuple(dt.shape)}")
    if B.dim() != 3 or C.dim() != 3:
        raise ValueError(
            f"B and C must be (B, L, N), got {tuple(B.shape)} and {tuple(C.shape)}"
        )
    Bs, L, H, P = x.shape
    if dt.shape != (Bs, L, H):
        raise ValueError(f"dt shape {tuple(dt.shape)} != {(Bs, L, H)}")
    if B.shape != (Bs, L, B.shape[-1]) or C.shape != B.shape:
        raise ValueError(f"B/C shapes disagree: {tuple(B.shape)} vs {tuple(C.shape)}")
    if B.shape[0] != Bs or B.shape[1] != L:
        raise ValueError(f"B shape {tuple(B.shape)} incompatible with x {tuple(x.shape)}")
    if A.dim() != 1 or A.shape[0] != H:
        raise ValueError(f"A must be (H,) = ({H},), got {tuple(A.shape)}")
    if D is not None and (D.dim() != 1 or D.shape[0] != H):
        raise ValueError(f"D must be (H,) = ({H},), got {tuple(D.shape)}")
    N = B.shape[-1]
    if initial_state is not None and tuple(initial_state.shape) != (Bs, H, P, N):
        raise ValueError(
            f"initial_state must be {(Bs, H, P, N)}, got {tuple(initial_state.shape)}"
        )
    return Bs, L, H, P, N


# --------------------------------------------------------------------------
# 1. the reference: naive step-by-step recurrence
# --------------------------------------------------------------------------


def ssd_recurrent(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive O(L) recurrence.  The reference implementation.

    Returns ``(y, final_state)`` with ``y`` of shape (B, L, H, P) and
    ``final_state`` of shape (B, H, P, N).

    Deliberately written as the literal definition so that it can be read
    against the paper without any mental compilation.
    """
    Bs, L, H, P, N = _check_ssd_inputs(x, dt, A, B, C, D, initial_state)

    if initial_state is None:
        state = x.new_zeros(Bs, H, P, N)
    else:
        state = initial_state

    ys: list[torch.Tensor] = []
    for t in range(L):
        dA = dt[:, t] * A                                  # (B, H)
        state = state * torch.exp(dA)[:, :, None, None]     # decay
        # outer product x_t (B,H,P) with B_t (B,N), scaled by dt_t
        state = state + (
            dt[:, t, :, None, None] * x[:, t, :, :, None] * B[:, t, None, None, :]
        )
        ys.append(torch.einsum("bhpn,bn->bhp", state, C[:, t]))

    y = torch.stack(ys, dim=1)                              # (B, L, H, P)
    if D is not None:
        y = y + D[None, None, :, None] * x
    return y, state


# --------------------------------------------------------------------------
# 2. the training path: chunked semiseparable matmul
# --------------------------------------------------------------------------


def ssd_chunked(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked SSD.  ``O(L/chunk_size)`` python steps, matmuls elsewhere.

    Splits the sequence into chunks of ``chunk_size``:

    * **intra-chunk** -- the causal masked matmul
      ``M[i, j] = exp(A(c_i - c_j)) * (C_i . B_j) * dt_j`` for ``j <= i``,
      contracted against ``x``.
    * **inter-chunk** -- a sequential scan over chunk boundaries carrying the
      (B, H, P, N) state, plus a rank-1 readout of that state at every
      position.

    The python loop is over chunks, never over positions.
    """
    Bs, L, H, P, N = _check_ssd_inputs(x, dt, A, B, C, D, initial_state)
    Q = int(chunk_size)
    if Q <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    nchunks = (L + Q - 1) // Q
    pad = nchunks * Q - L

    x_orig = x
    if pad:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, pad))

    xc = x.reshape(Bs, nchunks, Q, H, P)
    dtc = dt.reshape(Bs, nchunks, Q, H)
    Bc = B.reshape(Bs, nchunks, Q, N)
    Cc = C.reshape(Bs, nchunks, Q, N)

    # within-chunk inclusive cumulative sum of dA = dt * A.  The cumsum must be
    # taken *per chunk*, not over the whole sequence: the exponents below are
    # relative to the chunk start and that is what keeps them bounded.
    dAc = (dt * A.view(1, 1, H)).reshape(Bs, nchunks, Q, H)
    dA_cs_h = torch.cumsum(dAc, dim=2).permute(0, 1, 3, 2)  # (B, nc, H, Q)

    # ---- intra-chunk -----------------------------------------------------
    diff = dA_cs_h.unsqueeze(-1) - dA_cs_h.unsqueeze(-2)    # [i] - [j]
    tril = torch.tril(
        torch.ones(Q, Q, dtype=torch.bool, device=x.device)
    )
    diff = diff.masked_fill(~tril, float("-inf"))
    decay = torch.exp(diff)                                 # (B, nc, H, Q, Q)

    # C_i . B_j
    CB = torch.einsum("bciN,bcjN->bcij", Cc, Bc)            # (B, nc, Q, Q)
    M = decay * CB.unsqueeze(2) * dtc.permute(0, 1, 3, 2).unsqueeze(-2)
    xp = xc.permute(0, 1, 3, 2, 4)                          # (B, nc, H, Q, P)
    y_intra = torch.einsum("bchij,bchjp->bcihp", M, xp)     # (B, nc, Q, H, P)

    # ---- inter-chunk -----------------------------------------------------
    # contribution of chunk c to the state at the chunk's end
    rel = dA_cs_h[..., -1:] - dA_cs_h                       # <= 0
    w = torch.exp(rel) * dtc.permute(0, 1, 3, 2)            # (B, nc, H, Q)
    contrib = torch.einsum("bchj,bcjhp,bcjN->bchpN", w, xc, Bc)

    decay_chunk = torch.exp(dA_cs_h[..., -1])               # (B, nc, H)
    if initial_state is None:
        state = x.new_zeros(Bs, H, P, N)
    else:
        state = initial_state

    decay_pos = torch.exp(dA_cs_h)                          # (B, nc, H, Q)
    y_inter: list[torch.Tensor] = []
    for c in range(nchunks):
        # read out the incoming state at every position of this chunk
        decay_c = decay_pos[:, c].permute(0, 2, 1).unsqueeze(-1)   # (B, Q, H, 1)
        y_inter.append(
            decay_c * torch.einsum("biN,bhpN->bihp", Cc[:, c], state)
        )
        state = decay_chunk[:, c, :, None, None] * state + contrib[:, c]

    y = y_intra + torch.stack(y_inter, dim=1)               # (B, nc, Q, H, P)
    y = y.reshape(Bs, nchunks * Q, H, P)                    # nc and Q are adjacent
    if pad:
        y = y[:, :L]
    if D is not None:
        y = y + D[None, None, :, None] * x_orig
    return y, state


# --------------------------------------------------------------------------
# 3. the mixer module
# --------------------------------------------------------------------------


class RMSNormGated(nn.Module):
    """RMSNorm with a SiLU gate, as used inside the Mamba-2 mixer."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        if gate is not None:
            x = x * F.silu(gate)
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


class SSDMixer(nn.Module):
    """in_proj -> depthwise conv -> SSD -> gated norm -> out_proj.

    This is the Mamba-2 mixer.  ``forward`` accepts a ``cache`` so the module
    can be stepped one token at a time; the cache is a plain
    ``(conv_state, ssm_state)`` tuple of tensors, which is what
    ``Mamba2Block.forward`` hands back to its caller.
    """

    def __init__(self, cfg: SSDCfg, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        d_inner, N, H = cfg.d_inner, cfg.d_state, cfg.n_heads
        self.d_inner, self.d_state, self.n_heads = d_inner, N, H
        self.head_dim = cfg.head_dim

        self.in_proj = nn.Linear(
            cfg.d_model, 2 * d_inner + 2 * N + H, bias=False
        )
        self.conv1d = nn.Conv1d(
            cfg.d_conv_channels,
            cfg.d_conv_channels,
            kernel_size=cfg.d_conv,
            groups=cfg.d_conv_channels,          # depthwise
            padding=cfg.d_conv - 1,
            bias=True,
        )
        # A is a scalar per head: A = -exp(A_log) < 0
        self.A_log = nn.Parameter(torch.log(torch.arange(1, H + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(H))
        dt_min, dt_max = 1e-3, 1e-1
        dt = torch.exp(
            torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        self.dt_bias = nn.Parameter(_inverse_softplus(dt))
        self.norm = RMSNormGated(d_inner)
        self.out_proj = nn.Linear(d_inner, cfg.d_model, bias=False)

    # -- cache -------------------------------------------------------------

    def init_cache(
        self,
        batch_size: int,
        device=None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A zero ``(conv_state, ssm_state)`` pair for the first streaming step."""
        dtype = dtype if dtype is not None else self.in_proj.weight.dtype
        device = device if device is not None else self.in_proj.weight.device
        conv = torch.zeros(
            batch_size,
            self.cfg.d_conv - 1,
            self.cfg.d_conv_channels,
            device=device,
            dtype=dtype,
        )
        ssm = torch.zeros(
            batch_size, self.n_heads, self.head_dim, self.d_state,
            device=device, dtype=dtype,
        )
        return conv, ssm

    # -- conv --------------------------------------------------------------

    def _conv(self, xBC: torch.Tensor) -> torch.Tensor:
        """Causal depthwise conv over the (d_inner + 2 d_state) channels."""
        Bs, L, C = xBC.shape
        y = self.conv1d(xBC.transpose(1, 2))[:, :, :L].transpose(1, 2)
        return F.silu(y)

    def _conv_cache_from(self, xBC: torch.Tensor) -> torch.Tensor:
        """The last ``d_conv - 1`` *pre-conv* inputs, so a full forward can be
        resumed by single-token calls."""
        k = self.cfg.d_conv - 1
        Bs, L, C = xBC.shape
        if k == 0:
            return xBC.new_zeros(Bs, 0, C)
        if L >= k:
            return xBC[:, L - k:].contiguous()
        pad = xBC.new_zeros(Bs, k - L, C)
        return torch.cat([pad, xBC], dim=1)

    def _conv_step(self, xBC_t: torch.Tensor, conv_state: torch.Tensor):
        """One-token causal conv step.  ``conv_state``: (B, d_conv-1, C)."""
        k = self.cfg.d_conv
        window = torch.cat([conv_state, xBC_t.unsqueeze(1)], dim=1)   # (B, k, C)
        weight = self.conv1d.weight.squeeze(1).t()                   # (k, C)
        y = (window * weight.unsqueeze(0)).sum(1) + self.conv1d.bias   # (B, C)
        new_state = window[:, 1:] if k > 1 else window[:, :0]
        return F.silu(y), new_state

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        u: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        initial_state: torch.Tensor | None = None,
        return_state: bool = False,
    ):
        """u: (B, L, d_model).

        ``cache is None``     -> training / full-sequence path, returns y
                                 (or (y, ssm_state) when ``return_state``).
        ``cache is not None`` -> streaming path, returns (y, new_cache).

        ``initial_state`` (B, H, P, N) seeds the recurrence.
        """
        Bs, L, _ = u.shape
        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.d_state, self.n_heads], dim=-1
        )
        dt = F.softplus(dt + self.dt_bias)
        A = -torch.exp(self.A_log.float()).to(u.dtype)

        if cache is None:
            conv_state = self._conv_cache_from(xBC)
            xBC = self._conv(xBC)
            x, B, C = torch.split(
                xBC, [self.d_inner, self.d_state, self.d_state], dim=-1
            )
            x = x.reshape(Bs, L, self.n_heads, self.head_dim)
            y, ssm_state = ssd_chunked(
                x, dt, A, B, C, self.D, initial_state, self.cfg.chunk_size
            )
            new_cache = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = cache
            if L != 1:
                raise ValueError(
                    "streaming with a cache expects exactly one new token per call; "
                    f"got L={L}. Use the full-sequence path (cache=None) otherwise."
                )
            x_t, conv_state = self._conv_step(xBC[:, 0], conv_state)
            x, B, C = torch.split(
                x_t, [self.d_inner, self.d_state, self.d_state], dim=-1
            )
            # x_t / B / C are single-position here, i.e. (B, C) rather than (B, L, C)
            x = x.reshape(Bs, 1, self.n_heads, self.head_dim)
            dA = dt[:, 0] * A                                     # (B, H)
            ssm_state = ssm_state * torch.exp(dA)[:, :, None, None] + (
                dt[:, 0, :, None, None]
                * x[:, 0, :, :, None]
                * B[:, None, None, :]
            )
            y = torch.einsum("bhpn,bn->bhp", ssm_state, C).unsqueeze(1)
            y = y + self.D[None, None, :, None] * x
            new_cache = (conv_state, ssm_state)

        # the gated norm operates on the flat d_inner axis, so flatten heads first
        y = y.reshape(Bs, L, self.d_inner)
        y = self.norm(y, z.reshape(Bs, L, self.d_inner))
        out = self.out_proj(y)
        if cache is not None or return_state:
            return out, new_cache
        return out


# --------------------------------------------------------------------------
# 4. block (lazy re-export of the definition that lives in backbone.py)
# --------------------------------------------------------------------------


def __getattr__(name: str):
    """``Mamba2Block`` is defined in ``backbone.py`` (see the task brief).

    INTERFACES.md groups it under the ``ssd.py`` heading; the task brief puts it
    in ``backbone.py``.  It lives in ``backbone.py`` and is re-exported here
    lazily so that ``from sylphae.model.ssd import Mamba2Block`` also works without
    creating a circular import.
    """
    if name == "Mamba2Block":
        from .backbone import Mamba2Block

        return Mamba2Block
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
