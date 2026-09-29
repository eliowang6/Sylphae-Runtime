"""Backbone: pre-norm residual stack of Mamba-2 SSD blocks.

Per INTERFACES.md section 4 and the task brief:

    Mamba2Block:  RMSNorm -> SSD mixer -> residual
                  RMSNorm -> SwiGLU MLP -> residual

Pre-norm, ``n_layers`` stacked, with a small depthwise conv (``d_conv``) inside
the mixer before the SSD (that is the Mamba-2 arrangement and it lives in
``SSDMixer``).

The cache handed out by ``Mamba2Block.forward`` is the mixer's
``(conv_state, ssm_state)`` tuple.  ``conv_state`` is local to one request (a
sliding window); ``ssm_state`` is the fixed-size recurrent state that doubles as
the design's persistent latent memory.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssd import SSDCfg, SSDMixer

__all__ = ["RMSNorm", "SwiGLU", "Mamba2Block", "Mamba2Backbone"]

BlockCache = tuple[torch.Tensor, torch.Tensor]


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    """``down(silu(gate(x)) * up(x))`` -- 3 matrices, no bias."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Mamba2Block(nn.Module):
    """One pre-norm residual block: SSD mixer + SwiGLU MLP."""

    def __init__(self, cfg: SSDCfg, layer_idx: int = 0, d_ff: int | None = None,
                 p_drop: float = 0.0):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.norm1 = RMSNorm(cfg.d_model)
        self.mixer = SSDMixer(cfg, layer_idx=layer_idx)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, d_ff if d_ff is not None else cfg.d_model)
        # Regularisation applied to each sub-layer output, before the residual
        # add (the standard Mamba-2 placement).  ``p_drop`` defaults to 0.0 so
        # every existing construction stays numerically identical.
        self.drop = nn.Dropout(p_drop)

    def forward(
        self,
        x: torch.Tensor,
        cache: BlockCache | None = None,
        initial_state: torch.Tensor | None = None,
        return_state: bool = False,
    ):
        """x: (B, L, d_model).

        ``cache is None``     -> returns y (or (y, ssm_state) if ``return_state``)
        ``cache is not None`` -> returns (y, new_cache), exactly one new token

        ``initial_state`` (B, H, P, N) seeds the SSD recurrence; this is how the
        persistent memory conditions the computation.
        """
        if cache is None:
            mixed, blk_cache = self.mixer(
                self.norm1(x), initial_state=initial_state, return_state=True
            )
            ssm_state = blk_cache[1]
            x = x + self.drop(mixed)
        else:
            mixed, new_cache = self.mixer(self.norm1(x), cache)
            x = x + self.drop(mixed)
        x = x + self.drop(self.mlp(self.norm2(x)))
        if cache is not None:
            return x, new_cache
        if return_state:
            return x, ssm_state
        return x


class Mamba2Backbone(nn.Module):
    """Embedding + ``n_layers`` x Mamba2Block + final norm."""

    def __init__(self, cfg: SSDCfg, d_ff: int | None = None, p_drop: float = 0.0):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_drop = nn.Dropout(p_drop)
        self.layers = nn.ModuleList(
            [Mamba2Block(cfg, layer_idx=i, d_ff=d_ff, p_drop=p_drop) for i in range(cfg.n_layers)]
        )
        self.norm_f = RMSNorm(cfg.d_model)

    def init_cache(self, batch_size: int, device=None, dtype=None) -> list[BlockCache]:
        """Zero per-layer caches, for stepping the backbone one token at a time."""
        return [layer.mixer.init_cache(batch_size, device, dtype) for layer in self.layers]

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: list[BlockCache] | None = None,
        init_states: list[torch.Tensor] | None = None,
        use_checkpoint: bool = False,
        return_ssm_states: bool = False,
    ):
        """input_ids: (B, L) long -> hidden (B, L, d_model).

        ``cache`` is a per-layer list of block caches.  When given, ``L`` must
        be 1 and the return value is ``(hidden, new_cache)``.

        ``init_states`` is a per-layer list of (B, H, P, N) SSD states used to
        seed the recurrence -- the persistent memory.

        ``return_ssm_states`` returns ``(hidden, [per-layer final ssm states])``.
        """
        x = self.embed_drop(self.embed(input_ids))

        if cache is not None:
            new_cache: list[BlockCache] = []
            for layer, layer_cache in zip(self.layers, cache):
                x, nc = layer(x, layer_cache)
                new_cache.append(nc)
            return self.norm_f(x), new_cache

        if init_states is not None and len(init_states) != len(self.layers):
            raise ValueError(
                f"init_states has {len(init_states)} entries, expected {len(self.layers)}"
            )

        ssm_states: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            init = None if init_states is None else init_states[i]
            if use_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint

                x, s = checkpoint(
                    layer, x, None, init, True, use_reentrant=False
                )
            else:
                x, s = layer(x, initial_state=init, return_state=True)
            ssm_states.append(s)
        x = self.norm_f(x)
        if return_ssm_states:
            return x, ssm_states
        return x
