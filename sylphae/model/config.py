"""``ModelConfig`` -- the single resolved config for the whole policy.

It mirrors every ``SSDCfg`` field (so a ``ModelConfig`` can be passed anywhere an
``SSDCfg`` is expected, including ``collate`` and ``MemState.zeros``) and adds
the head sizes and the MLP width.  ``n_params_estimate()`` is analytic and is
checked against ``SylphaeModel.num_parameters()`` in the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, asdict

from .ssd import SSDCfg

__all__ = [
    "ModelConfig",
    "SSD_FIELD_NAMES",
    "REFERENCE_CONFIG",
    "PARAM_DEC_LAYERS",
    "PARAM_DEC_HEADS",
    "PARAM_DEC_MAX_POS",
]


SSD_FIELD_NAMES = tuple(f.name for f in fields(SSDCfg))

#: Parameter-decoder geometry.  Structural, not config: the decoder it replaced
#: (a ``GRUCell`` over ``pooled``) had a fixed depth too, and the parameter count
#: in :meth:`ModelConfig.n_params_estimate` is asserted against
#: ``SylphaeModel.num_parameters()`` in the suite, so a geometry that is not
#: reflected here fails loudly rather than shipping a wrong config.
#:
#: ``PARAM_DEC_MAX_POS`` bounds the learned positional table.  It matches
#: ``ModelConfig.max_params``'s own documented ceiling (256), so every admissible
#: ``P`` has a position.
PARAM_DEC_LAYERS = 2
PARAM_DEC_HEADS = 4
PARAM_DEC_MAX_POS = 256


@dataclass
class ModelConfig:
    # ---- backbone geometry (mirrors SSDCfg) ------------------------------
    d_model: int = 1024
    n_layers: int = 24
    d_state: int = 128
    n_heads: int = 16
    chunk_size: int = 64
    expand: int = 2
    d_conv: int = 4
    vocab_size: int = 32768
    max_len: int = 2048

    # ---- MLP width -------------------------------------------------------
    # SwiGLU hidden width.  d_ff = d_model keeps the 24-layer stack at ~217M
    # parameters, inside the 150-250M band DESIGN.md section 3 targets.
    d_ff: int = 1024

    # ---- heads -----------------------------------------------------------
    n_ops: int = 16
    n_candidates: int = 8          # default k for collate
    #: P, the parameter-head sequence width.  INTERFACES.md section 4 first froze it
    #: at <= 32, which the parameterised-payload measurement now shows is too small:
    #: the props-JSON the head must emit has median 50 tokens / p75 70, so a 32-wide
    #: cap truncates ~78% of structured payloads and makes them unparseable at decode
    #: (see ``work/iter4_ceilings.md``).  32 is kept as the default (the value every
    #: previously-recorded run used); it is a knob now, bounded at 256.
    max_params: int = 32
    param_dec_hidden: int = 512
    param_emb_dim: int = 256

    #: Tokens of ``old_source`` around the edit that ``collate`` appends to each
    #: sequence (opened by ``<mask>``).  ``0`` keeps the original path-only input.
    #: It is a geometry-neutral input knob -- it changes what the heads can see,
    #: not the parameter count -- so it lives here so ``collate`` can read it off
    #: the same config object the model is built from.
    ctx_tokens: int = 0

    # ---- misc ------------------------------------------------------------
    dropout: float = 0.0
    grad_checkpoint: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.ssd_cfg()  # raises on bad backbone geometry
        for name in ("d_ff", "n_ops", "n_candidates", "max_params",
                     "param_dec_hidden", "param_emb_dim"):
            v = getattr(self, name)
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"ModelConfig.{name} must be a positive int, got {v!r}")
        if not isinstance(self.ctx_tokens, int) or self.ctx_tokens < 0:
            raise ValueError(
                f"ModelConfig.ctx_tokens must be a non-negative int, got {self.ctx_tokens!r}")
        if self.max_params > 256:
            raise ValueError(
                f"ModelConfig.max_params must be <= 256, got {self.max_params}. The field is "
                "bounded (it widens the decoder's positional table and the param logits "
                "tensor), not left unbounded; 32 remains the default every prior run "
                "trained at."
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"ModelConfig.dropout must be in [0, 1), got {self.dropout}")

    # -- SSDCfg view -------------------------------------------------------

    def ssd_cfg(self) -> SSDCfg:
        return SSDCfg(**{name: getattr(self, name) for name in SSD_FIELD_NAMES})

    # -- SSDCfg-compatible derived quantities ------------------------------

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def head_dim(self) -> int:
        return self.d_inner // self.n_heads

    @property
    def d_conv_channels(self) -> int:
        return self.d_inner + 2 * self.d_state

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_ssd_cfg(cls, cfg: SSDCfg, **overrides) -> "ModelConfig":
        base = {name: getattr(cfg, name) for name in SSD_FIELD_NAMES}
        base.update(overrides)
        return cls(**base)

    def to_dict(self) -> dict:
        return asdict(self)

    # -- parameter count ---------------------------------------------------

    def n_params_estimate(self) -> int:
        """Analytic parameter count.  Must equal ``SylphaeModel.num_parameters()``."""
        d = self.d_model
        d_inner = self.expand * d
        ch = self.d_inner + 2 * self.d_state

        per_layer = (
            d                                             # norm1
            + d * (2 * d_inner + 2 * self.d_state + self.n_heads)   # in_proj (no bias)
            + ch * self.d_conv + ch                       # depthwise conv w + b
            + 3 * self.n_heads                            # A_log, D, dt_bias
            + d_inner                                     # gated norm weight
            + d_inner * d                                 # out_proj (no bias)
            + d                                           # norm2
            + 3 * d * self.d_ff                           # SwiGLU
        )
        backbone = self.vocab_size * d + self.n_layers * per_layer + d  # + norm_f

        ph, pe, V = self.param_dec_hidden, self.param_emb_dim, self.vocab_size
        heads = (
            (d * d + d)                                    # target score: project Linear(d,d)
            + (4 * d * (2 * d) + 2 * d)                    # target score: interact Linear(4d, 2d)
            + (2 * d + 1)                                  # target score: interact Linear(2d, 1)
            + 2 * (d * d + d) + 1                          # target score: MaxSim req/cand Linear(d,d) + gate
            + d * self.n_ops + self.n_ops                 # op head
            + (d + 1)                                      # confidence
            + (d + 1)                                      # memory-write gate
            + d * ph + ph                                  # decoder ctx projection (pooled -> hidden)
            + d * ph + ph                                  # decoder request-key projection
            + V * pe                                       # decoder token embedding
            + pe * ph + ph                                 # decoder token input projection
            + PARAM_DEC_MAX_POS * ph                       # decoder positional table
            # decoder layers: 2 x MultiheadAttention (4 h^2 + 4 h each) + FFN
            # (4 h^2 + 3 h) + 3 LayerNorms (2 h each)
            + PARAM_DEC_LAYERS * (12 * ph * ph + 17 * ph)
            + 2 * ph                                       # decoder final LayerNorm
            + ph * V + V                                   # decoder output projection
        )

        return backbone + heads


#: The model geometry every latency number in this project is quoted at.  It is a
#: *shape*, not a hyperparameter: the benchmarks measure what a real request costs
#: to serve, so they need the production geometry rather than whatever
#: ``ModelConfig()``'s dataclass defaults happen to be.  Declared here, next to the
#: class it configures, so the inference half can build the same model the
#: benchmarks did without importing a training-side module.
REFERENCE_CONFIG: dict[str, int] = {
    "d_model": 1024,
    "n_layers": 24,
    "d_state": 128,
    "n_heads": 16,
    "chunk_size": 64,
    "expand": 2,
    "vocab_size": 32768,
    "max_len": 2048,
    "d_ff": 1024,
    "n_candidates": 8,
    "max_params": 32,
}
