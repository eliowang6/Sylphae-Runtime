"""The Sylphae policy: SSD backbone + five heads + persistent latent memory.

Heads (DESIGN.md section 3, INTERFACES.md section 4):

===============  ==========================================================
target head      scalar score per candidate, read from the hidden state at
                 each candidate's ``<cand>`` token
op head          softmax over the 16-op vocabulary
param head       autoregressive decoder that cross-attends the request token
                 grid, ``P <= 32`` tokens
confidence head  sigmoid scalar -- the escalation trigger, not decoration
memory head      sigmoid scalar -- how much of this request's state persists
===============  ==========================================================

``MemState`` is the persistent latent memory.  It *is* the SSD recurrent state
(one (B, H, P, N) tensor per layer), which is why the design can claim a
decayed, gated, lossy memory for free.  The memory-write gate scales the update:

    new_state = old_state + gate * (raw_state - old_state)

so ``gate -> 1`` stores what this request produced and ``gate -> 0`` keeps the
previous memory.  Cold start (``mem=None``) is ``old_state = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Mamba2Backbone
from .collate import Batch
from .config import (
    PARAM_DEC_HEADS,
    PARAM_DEC_LAYERS,
    PARAM_DEC_MAX_POS,
    ModelConfig,
)
from .ssd import SSDCfg
from .tokenizer import BOS_ID, CAND_ID, EOS_ID, PAD_ID, SEP_ID

__all__ = ["PolicyOutput", "MemState", "ParamDecoder", "SylphaeModel"]

# Large but finite, so it survives fp16 and never produces NaN in a softmax.
_MASK_FILL = -1e4


# --------------------------------------------------------------------------
# persistent memory
# --------------------------------------------------------------------------


@dataclass
class MemState:
    """Per-layer SSD recurrent state.  ``ssm[i]`` has shape (B, H, P, N)."""

    ssm: list[torch.Tensor]

    # -- constructors ------------------------------------------------------

    @staticmethod
    def zeros(batch_size: int, cfg, device=None, dtype: torch.dtype = torch.float32) -> "MemState":
        """Cold-start memory for ``batch_size`` sequences.

        ``cfg`` may be an ``SSDCfg`` or a ``ModelConfig`` -- both expose
        ``n_layers``, ``n_heads``, ``head_dim`` and ``d_state``.
        """
        n_layers = int(cfg.n_layers)
        shape = (int(batch_size), int(cfg.n_heads), int(cfg.head_dim), int(cfg.d_state))
        return MemState([torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)])

    # -- protocol ----------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return int(self.ssm[0].shape[0]) if self.ssm else 0

    @property
    def n_layers(self) -> int:
        return len(self.ssm)

    @property
    def device(self):
        return self.ssm[0].device if self.ssm else None

    def detach(self) -> "MemState":
        return MemState([s.detach() for s in self.ssm])

    def to(self, device, dtype: torch.dtype | None = None) -> "MemState":
        return MemState(
            [s.to(device=device, dtype=dtype) if dtype is not None else s.to(device)
             for s in self.ssm]
        )

    def clone(self) -> "MemState":
        return MemState([s.clone() for s in self.ssm])

    def index_select(self, index: torch.Tensor) -> "MemState":
        """Reorder / subset the memory along the batch dimension."""
        return MemState([s.index_select(0, index) for s in self.ssm])

    def numel(self) -> int:
        return int(sum(s.numel() for s in self.ssm))

    def __len__(self) -> int:
        return len(self.ssm)


# --------------------------------------------------------------------------
# parameter decoder
# --------------------------------------------------------------------------


class _ParamDecoderLayer(nn.Module):
    """One pre-norm decoder layer: causal self-attention, request cross-attention,
    then a feed-forward block.  ``req_mask`` is a *keep* mask; it is inverted to
    the ``key_padding_mask`` ``MultiheadAttention`` expects inside ``forward``."""

    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden),
        )
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.ln3 = nn.LayerNorm(hidden)

    def forward(
        self,
        x: torch.Tensor,
        req: torch.Tensor,
        req_mask: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        n = self.ln1(x)
        h, _ = self.self_attn(n, n, n, attn_mask=causal_mask, need_weights=False)
        x = x + h
        h, _ = self.cross_attn(
            self.ln2(x), req, req,
            key_padding_mask=~req_mask,
            need_weights=False,
        )
        x = x + h
        return x + self.ff(self.ln3(x))


class ParamDecoder(nn.Module):
    """Autoregressive parameter decoder over the **request** token states.

    Why it is not a GRU over ``pooled`` (this class used to be one)
    ---------------------------------------------------------------
    Until this version the decoder was a ``GRUCell`` whose only conditioning was
    ``pooled``: one 384-d vector, the hidden state at the last non-pad position,
    i.e. *after* the request, the source window and every candidate.  A
    frozen-backbone probe measured what that costs (ROADMAP 12.15, iter30):

    * the gold value's first digit decodes linearly from ``pooled`` at **0.400**
      on held-out rows (chance 0.10), and a 256-unit MLP does **not** beat it
      (**0.375**; train 1.000) -- what is there is memorisation, not signal;
    * at the value's own request-token position the same probe reads **0.665**;
    * a linear per-position scorer finds the gold span at **top-1 0.722**;
    * on the rows it trained on, the GRU emitted the exact value **0/24**.

    The information is in the request and the decoder was architected not to read
    it.  This decoder cross-attends the request token grid instead, so the copy
    source is reachable by construction.  The output contract is unchanged --
    ``(B, P, vocab_size)`` -- so no ``Batch`` field, loss, or decode site moved.

    ``req_tokens``/``req_mask`` are the token grid ``_token_grids`` already builds
    for the target head.  They are optional so the streaming path (which never
    decodes parameters) can pass ``None`` and get a pooled-only decode.
    """

    def __init__(
        self,
        d_model: int,
        hidden: int,
        emb_dim: int,
        vocab_size: int,
        bos_id: int,
        pad_id: int,
        eos_id: int,
    ):
        super().__init__()
        self.hidden = hidden
        self.bos_id = bos_id
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.ctx_proj = nn.Linear(d_model, hidden)
        self.req_proj = nn.Linear(d_model, hidden)
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.inp = nn.Linear(emb_dim, hidden)
        self.pos = nn.Embedding(PARAM_DEC_MAX_POS, hidden)
        self.layers = nn.ModuleList(
            [_ParamDecoderLayer(hidden, PARAM_DEC_HEADS) for _ in range(PARAM_DEC_LAYERS)]
        )
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, vocab_size)

    def _initial(self, pooled: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.ctx_proj(pooled))

    @staticmethod
    def _causal_mask(P: int, device, dtype) -> torch.Tensor:
        m = torch.full((P, P), float("-inf"), device=device, dtype=dtype)
        return torch.triu(m, diagonal=1)

    def _run(
        self,
        pooled: torch.Tensor,
        prev: torch.Tensor,
        req_tokens: torch.Tensor | None,
        req_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        B, P = prev.shape
        if P > PARAM_DEC_MAX_POS:
            raise ValueError(
                f"ParamDecoder: {P} positions exceeds the positional table "
                f"({PARAM_DEC_MAX_POS}); lower max_params or raise PARAM_DEC_MAX_POS."
            )
        ar = torch.arange(P, device=prev.device)
        x = self.inp(self.emb(prev)) + self.pos(ar).unsqueeze(0)
        x = x + self._initial(pooled).unsqueeze(1)

        if req_tokens is not None and req_tokens.shape[1] > 0:
            req = self.req_proj(req_tokens)
            keep = (
                torch.ones(req.shape[:2], dtype=torch.bool, device=req.device)
                if req_mask is None
                else req_mask.to(torch.bool)
            )
            # An all-masked row makes the cross-attention softmax undefined.  Point
            # it at slot 0 -- a real token for every row that has a request, a zero
            # vector for the streaming/rng case -- so the row is neutral, not NaN.
            empty = ~keep.any(dim=1)
            if bool(empty.any()):
                keep = keep.clone()
                keep[empty, 0] = True
        else:
            req = pooled.new_zeros(B, 1, self.hidden)
            keep = torch.ones(B, 1, dtype=torch.bool, device=pooled.device)

        causal = self._causal_mask(P, x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, req, keep, causal)
        return self.out(self.norm(x))

    def forward(
        self,
        pooled: torch.Tensor,
        param_ids: torch.Tensor,
        req_tokens: torch.Tensor | None = None,
        req_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced logits, shape (B, P, vocab_size).

        Step 0 is fed ``<bos>``; step ``t`` is fed ``param_ids[:, t-1]``.
        """
        B, P = param_ids.shape
        prev = torch.cat(
            [param_ids.new_full((B, 1), self.bos_id), param_ids[:, :-1]], dim=1
        )
        return self._run(pooled, prev, req_tokens, req_mask)

    @torch.no_grad()
    def generate(
        self,
        pooled: torch.Tensor,
        max_len: int,
        req_tokens: torch.Tensor | None = None,
        req_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Greedy decode, shape (B, max_len).

        Each step re-runs the full prefix through the (causal) self-attention, so
        the emission is conditioned on the model's *own* previous tokens.  The
        GRU this replaced carried that recurrence in its hidden state; a
        transformer has to see the prefix to reproduce it.

        A row that has emitted its terminator is frozen on ``<pad>`` and the loop
        exits once every row has.  Two reasons, not one: the decode is *over* for
        that row, so continuing would append junk after the value, and ``max_len``
        is ``cfg.max_params`` (64) rather than the gold width, so an unconditional
        loop would pay 64 decoder passes to emit a two-token value.
        """
        B = pooled.shape[0]
        bos = pooled.new_full((B, 1), self.bos_id, dtype=torch.long)
        out = torch.full((B, max_len), self.pad_id, dtype=torch.long, device=pooled.device)
        done = torch.zeros(B, dtype=torch.bool, device=pooled.device)
        for t in range(max_len):
            if bool(done.all()):
                break
            prefix = torch.cat([bos, out[:, :t]], dim=1)
            logits = self._run(pooled, prefix, req_tokens, req_mask)
            nxt = logits[:, -1].argmax(dim=-1)
            nxt = torch.where(done, torch.full_like(nxt, self.pad_id), nxt)
            out[:, t] = nxt
            done = done | (nxt == self.eos_id) | (nxt == self.pad_id)
        return out


# --------------------------------------------------------------------------
# policy output
# --------------------------------------------------------------------------


@dataclass
class PolicyOutput:
    target_logits: torch.Tensor   # (B, C)
    op_logits: torch.Tensor       # (B, N_OPS)
    param_logits: torch.Tensor    # (B, P, vocab_size)
    confidence: torch.Tensor      # (B,) in [0, 1]
    mem_write: torch.Tensor       # (B,) in [0, 1]


# --------------------------------------------------------------------------
# target head
# --------------------------------------------------------------------------


class TargetScoreHead(nn.Module):
    """Score each candidate by comparing it against the request.

    A plain linear readout of the SSD state at ``</cand>`` can only fire on
    memorised request->path associations: the state vector carries the request
    and the candidate, but a single affine map cannot compute a *relation*
    between them.  Measured on iter18 the model therefore lands exactly on the
    lexical oracle (VAL hit@1 0.274 vs oracle 0.316) while under-fitting its
    own training set (TRAIN L_target 1.309 against ln(8)=2.079, hit@1 0.557).

    This head instead gathers two states -- ``h_req`` after ``<sep>`` and
    ``h_cand`` at ``</cand>`` -- and scores the pair through an MLP over the
    standard interaction features (concatenation, product, difference).  The
    product and difference terms are what make the comparison itself learnable
    rather than relying on the backbone to have pre-computed it.

    But two summary vectors still cannot compute a *set* relation.  Which tokens
    of the request occur in the candidate is invisible once each side is a
    single 384-dim state: on iter21 the head demonstrably reads content
    (hit@1 0.207 with snippets vs 0.093 on bare paths) yet its scores have
    Spearman 0.01 with lexical request<->candidate overlap -- it is matching
    something, but not overlap, and it sits below the lexical oracle (0.244 vs
    0.315).  ``max_sim`` is the fix: it scores each candidate against the
    request *token by token* -- max-pooling a learned similarity matrix over the
    candidate's tokens for each request token, then summing -- so the set
    intersection is computed explicitly rather than having to survive
    compression into one vector.  Learned projections mean it can match beyond
    surface lexicon, but the structure guarantees the overlap signal is
    representable at all.
    """

    def __init__(self, d_model: int, hidden: int | None = None, p_drop: float = 0.0):
        super().__init__()
        h = hidden or 2 * d_model
        self.project = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(p_drop),
        )
        # [h_req ; h_cand ; h_req*h_cand ; |h_req - h_cand|]
        self.interact = nn.Sequential(
            nn.Linear(4 * d_model, h), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(h, 1),
        )
        # token-level late interaction (ColBERT-style max-similarity)
        self.req_proj = nn.Linear(d_model, d_model)
        self.cand_proj = nn.Linear(d_model, d_model)
        # gate: how much the token-level evidence should move the score.
        # Initialised OPEN, not at 0.  The score gradient reaching the projections
        # is scaled by tanh(match_gate), so a gate parked at 0 zeroes their
        # gradients entirely: the channel can never learn to be useful and stays
        # dead for the whole run (MEASURED on iter23: tanh(gate)=0.0235 after 2000
        # from-scratch steps, yet the cold MaxSim channel already matched the
        # trained interact path on the hardest tier).  Warm-starting from a
        # pre-MaxSim checkpoint is impossible anyway (load_checkpoint is strict
        # and these parameters do not exist there), so the inert-at-init property
        # buys nothing and costs the channel.
        self.match_gate = nn.Parameter(torch.ones(()))

    @staticmethod
    def _max_sim(
        rq: torch.Tensor, rq_mask: torch.Tensor,
        cd: torch.Tensor, cd_mask: torch.Tensor,
    ) -> torch.Tensor:
        """ColBERT late interaction over padded token grids.

        ``rq`` (B, Lq, d) request tokens with ``rq_mask`` (B, Lq);
        ``cd`` (B, C, Lc, d) candidate tokens with ``cd_mask`` (B, C, Lc).
        Returns (B, C): for every request token the best-matching candidate
        token, averaged over request tokens -- an explicit set-intersection
        score in [-1, 1], where 0 means "no usable evidence".

        Inputs are L2-normalised by the caller, so the raw einsum *is* a cosine
        similarity and needs no further scaling.  Masked candidate tokens are
        pushed to -2, below any real cosine value, so they can never win a max;
        a request token whose candidate has no live tokens at all maps back to
        the neutral 0 rather than being penalised.
        """
        rq_mask = rq_mask.to(torch.bool)
        cd_mask = cd_mask.to(torch.bool)
        # (B, C, Lq, Lc): request token k vs candidate token m
        sim = torch.einsum("bqk,bcmk->bcqm", rq, cd)
        sim = sim.masked_fill(~cd_mask.unsqueeze(2), -2.0)
        best, _ = sim.max(dim=-1)                               # (B, C, Lq)
        # a candidate with no live tokens is neutral, not a negative match
        best = torch.where(best <= -1.5, torch.zeros_like(best), best)
        # a request token outside the request span contributes nothing
        best = torch.where(rq_mask.unsqueeze(1), best, torch.zeros_like(best))
        n_rq = rq_mask.sum(dim=-1).clamp(min=1)                 # (B,)
        return best.sum(dim=-1) / n_rq.unsqueeze(-1)

    def forward(
        self,
        h_req: torch.Tensor, h_cand: torch.Tensor,
        req_tokens: torch.Tensor | None = None, req_mask: torch.Tensor | None = None,
        cand_tokens: torch.Tensor | None = None, cand_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``h_req`` (B, d); ``h_cand`` (B, C, d) -> logits (B, C).

        The token grids are optional so callers without span bookkeeping
        (``stream_step`` sees only one token) still get the interaction score.
        """
        rq = self.project(h_req).unsqueeze(1)          # (B, 1, d)
        cd = self.project(h_cand)                       # (B, C, d)
        rq_b = rq.expand_as(cd)
        z = torch.cat([rq_b, cd, rq_b * cd, rq_b - cd], dim=-1)
        score = self.interact(z).squeeze(-1)
        if req_tokens is not None and cand_tokens is not None:
            ms = self._max_sim(
                F.normalize(self.req_proj(req_tokens), dim=-1),
                req_mask,
                F.normalize(self.cand_proj(cand_tokens), dim=-1),
                cand_mask,
            )
            # tanh keeps the learned gate bounded, so the new channel cannot
            # blow the logits up early in training and dominate the loss
            score = score + torch.tanh(self.match_gate) * ms
        return score


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


class SylphaeModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.ssd_cfg: SSDCfg = cfg.ssd_cfg()
        self.backbone = Mamba2Backbone(self.ssd_cfg, d_ff=cfg.d_ff, p_drop=cfg.dropout)
        d = cfg.d_model

        self.target_score = TargetScoreHead(d, p_drop=cfg.dropout)
        self.op_head = nn.Linear(d, cfg.n_ops, bias=True)
        self.conf_head = nn.Linear(d, 1, bias=True)
        self.mem_head = nn.Linear(d, 1, bias=True)
        self.dropout = nn.Dropout(cfg.dropout)

        # placeholders filled in policy.py: shared with the tokenizer
        self.param_decoder = ParamDecoder(
            d_model=d,
            hidden=cfg.param_dec_hidden,
            emb_dim=cfg.param_emb_dim,
            vocab_size=cfg.vocab_size,
            bos_id=BOS_ID,   # INTERFACES.md section 4: the frozen special-token order
            pad_id=PAD_ID,
            eos_id=EOS_ID,
        )

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        batch: Batch,
        mem: MemState | None = None,
        cache: list | None = None,
        mem_write_override: float | None = None,
    ) -> tuple[PolicyOutput, MemState]:
        """Run the policy.

        ``cache`` (from ``backbone.init_cache``) runs the backbone in streaming
        mode: exactly one new token per call, carrying the depthwise-conv history
        as well as the SSD state.  That is the single-token decode path and it
        reproduces a full forward exactly.  ``mem`` cannot be combined with
        ``cache`` -- the cache already carries the full state.

        ``mem_write_override`` forces the memory-write gate to a constant.  It is
        a testing/debug hook (``1.0`` makes the returned state the raw SSD state)
        and does not affect any production path.
        """
        input_ids = batch.input_ids
        B = int(input_ids.shape[0])
        if mem is not None and cache is not None:
            raise ValueError(
                "pass either mem (persistent, cross-request) or cache (exact "
                "single-token streaming) -- not both."
            )
        if mem is not None and mem.batch_size != B:
            raise ValueError(
                f"mem carries {mem.batch_size} sequences but the batch has {B}; "
                "index_select() the memory or fix the batch."
            )
        if mem is not None and mem.n_layers != self.ssd_cfg.n_layers:
            raise ValueError(
                f"mem has {mem.n_layers} layers but the model has {self.ssd_cfg.n_layers}"
            )

        if cache is not None:
            if mem is not None:
                raise ValueError(
                    "pass either mem (persistent, cross-request) or cache (exact "
                    "single-token streaming) -- not both."
                )
            out, _new_cache, mem_state = self.stream_step(
                batch, cache, mem_write_override=mem_write_override
            )
            return out, mem_state

        hidden, raw_states = self.backbone(
            input_ids,
            init_states=None if mem is None else mem.ssm,
            use_checkpoint=self.cfg.grad_checkpoint,
            return_ssm_states=True,
        )
        pooled = self._pool(hidden, batch.attention_mask)
        out = self._heads(batch, hidden, pooled)

        gate = self._gate(out.mem_write, mem_write_override)
        if mem is None:
            new_ssm = [gate * s for s in raw_states]
        else:
            new_ssm = [old + gate * (s - old) for old, s in zip(mem.ssm, raw_states)]
        return out, MemState(new_ssm)

    def stream_step(
        self,
        batch: Batch,
        cache: list,
        mem_write_override: float | None = None,
    ) -> tuple[PolicyOutput, list, MemState]:
        """Single-token decode.  One new token, carrying the block cache.

        Returns ``(output, new_cache, mem_state)``.  Feeding ``new_cache`` back
        reproduces a full forward exactly (see ``tests/test_policy.py``), which
        is the property the latency argument rests on.  ``mem_state`` is the
        gated persistent memory derived from the same states.
        """
        input_ids = batch.input_ids
        if input_ids.shape[1] != 1:
            raise ValueError(
                "stream_step expects exactly one new token, got "
                f"L={input_ids.shape[1]}"
            )
        hidden, new_cache = self.backbone(input_ids, cache=cache)
        pooled = self._pool(hidden, batch.attention_mask)
        out = self._heads(batch, hidden, pooled)

        raw_states = [c[1] for c in new_cache]
        gate = self._gate(out.mem_write, mem_write_override)
        mem_state = MemState([gate * s for s in raw_states])
        return out, new_cache, mem_state

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Hidden state at each sequence's last non-pad position."""
        B = hidden.shape[0]
        lengths = attention_mask.sum(dim=-1).clamp(min=1)
        last = (lengths - 1).to(torch.long)
        return hidden[torch.arange(B, device=hidden.device), last]

    @staticmethod
    def _gate(mem_write: torch.Tensor, override: float | None) -> torch.Tensor:
        gate = mem_write if override is None else torch.full_like(mem_write, float(override))
        return gate[:, None, None, None]

    @staticmethod
    def _token_grids(
        batch: Batch, hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Request and candidate token grids for the late-interaction channel.

        ``cand_index`` points at each candidate's *closing* ``</cand>``.  The
        span we want to compare is the content between ``<cand>`` and
        ``</cand>``, so the opening markers are recovered from the token ids
        rather than a new Batch field (the interface is frozen).

        Returns ``req_tokens`` (B, Lq, d), ``req_mask`` (B, Lq),
        ``cand_tokens`` (B, C, Lc, d), ``cand_mask`` (B, C, Lc), padded to the
        widest span per side.  All-zeros masks mean "no tokens" -- the caller
        treats that as "no evidence", not as a match.
        """
        B, L, d = hidden.shape
        C = int(batch.cand_index.shape[1])
        ids = batch.input_ids
        attn = batch.attention_mask.bool()

        # -- request span: everything between <bos> and the first <sep> --------
        sep = (ids == SEP_ID) & attn
        has_sep = sep.any(dim=-1)
        first_sep = sep.float().argmax(dim=-1)                     # (B,)
        req_len = torch.where(has_sep, first_sep.clamp(min=1), attn.sum(dim=-1))
        req_len = req_len.clamp(min=0, max=L)                      # (B,)
        # request tokens are positions 1 .. req_len-1 (position 0 is <bos>)
        Lq = int(req_len.max().clamp(min=1).item())
        ar = torch.arange(L, device=hidden.device).unsqueeze(0)    # (1, L)
        rq_pos = (ar >= 1) & (ar < req_len.unsqueeze(-1)) & attn    # (B, L)
        req_tokens = hidden * rq_pos.unsqueeze(-1).to(hidden.dtype)
        req_tokens = req_tokens[:, :Lq]
        req_mask = rq_pos[:, :Lq]

        # -- candidate spans: <cand> .. </cand>, per candidate ----------------
        cand_open = (ids == CAND_ID) & attn                        # (B, L)
        close = batch.cand_index.clamp(min=0, max=L - 1)           # (B, C)
        # walk backwards from each </cand> to its own <cand>: the nearest
        # opening marker at or before the closing position.  argmax on
        # (position * hit) returns the LAST hit since positions increase; a row
        # with no hit falls back to 0, which yields an empty content span.
        idx = torch.arange(L, device=hidden.device).view(1, L)
        opens = []
        for c in range(C):
            le = ar <= close[:, c].unsqueeze(-1)
            hit = cand_open & le                              # (B, L)
            opens.append((idx * hit.long()).argmax(dim=-1))   # (B,)
        open_pos = torch.stack(opens, dim=1)                       # (B, C)
        span_len = (close - open_pos + 1).clamp(min=0)             # (B, C)
        # content excludes the <cand> marker itself; keep </cand> out too
        content_len = (span_len - 2).clamp(min=0)                  # (B, C)
        Lc = int(content_len.max().clamp(min=1).item())
        cand_tokens = hidden.new_zeros(B, C, Lc, d)
        cand_mask = torch.zeros(B, C, Lc, dtype=torch.bool, device=hidden.device)
        ar_c = torch.arange(Lc, device=hidden.device)              # (Lc,)
        # offset of each token within its candidate's content span
        # position(p, c) = open_pos[:, c] + 1 + k  -> gather rows
        base = (open_pos + 1).unsqueeze(-1)                        # (B, C, 1)
        pos = base + ar_c.view(1, 1, Lc)                           # (B, C, Lc)
        valid = (ar_c.view(1, 1, Lc) < content_len.unsqueeze(-1))  # (B, C, Lc)
        pos_c = pos.clamp(min=0, max=L - 1)
        gathered = torch.gather(
            hidden, 1,
            pos_c.reshape(B, C * Lc, 1).expand(-1, -1, d),
        ).view(B, C, Lc, d)
        cand_tokens = torch.where(
            valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))
        cand_mask = valid
        return req_tokens, req_mask, cand_tokens, cand_mask

    def _heads(self, batch: Batch, hidden: torch.Tensor, pooled: torch.Tensor) -> PolicyOutput:
        pooled = self.dropout(pooled)
        B, L, d = hidden.shape
        C = int(batch.cand_index.shape[1])
        # The late-interaction channel compares the request span against each
        # candidate's content span, which only exists once the whole sequence is
        # in view.  stream_step feeds exactly one token per call, so it can never
        # see those spans; it degrades to the interact-only score by design (the
        # token grids are optional exactly for this case, see forward).  The
        # serving path scores candidates through the full forward, so nothing
        # consumes the streaming target_logits.
        if C > 0:
            gather_idx = batch.cand_index.unsqueeze(-1).expand(-1, -1, d)
            cand_h = torch.gather(hidden, 1, gather_idx)
            if L > 1:
                req_tokens, req_mask, cand_tokens, cand_mask = self._token_grids(
                    batch, hidden)
            else:
                req_tokens = req_mask = cand_tokens = cand_mask = None
        else:  # pragma: no cover - collate always emits at least one slot
            cand_h = hidden.new_zeros(B, 0, d)
            req_tokens = req_mask = cand_tokens = cand_mask = None

        # The request state: the SSD state right after ``<sep>``.  Found by
        # token id rather than a new Batch field so the interface stays frozen;
        # ``stream_step`` (single-token input) has no <sep> in view, so fall
        # back to the pooled state -- the request is already baked into the
        # recurrent state by then.
        sep = (batch.input_ids == SEP_ID)
        has_sep = sep.any(dim=-1)
        first_sep = sep.float().argmax(dim=-1)
        req_h = torch.where(
            has_sep.unsqueeze(-1),
            hidden[torch.arange(B, device=hidden.device), first_sep.clamp(max=L - 1)],
            pooled,
        )
        target_logits = self.target_score(
            req_h, cand_h, req_tokens, req_mask, cand_tokens, cand_mask,
        )
        target_logits = target_logits.masked_fill(~batch.cand_mask, _MASK_FILL)

        return PolicyOutput(
            target_logits=target_logits,
            op_logits=self.op_head(pooled),
            param_logits=self.param_decoder(pooled, batch.param_ids, req_tokens, req_mask),
            confidence=torch.sigmoid(self.conf_head(pooled).squeeze(-1)),
            mem_write=torch.sigmoid(self.mem_head(pooled).squeeze(-1)),
        )

    # -- introspection -----------------------------------------------------

    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    @torch.no_grad()
    def generate_params(self, batch: Batch, max_len: int | None = None) -> torch.Tensor:
        """Greedy parameter tokens, shape (B, P).  Used by the eval harness.

        ``P`` defaults to ``cfg.max_params`` -- the model's declared capacity --
        and **not** to ``batch.param_ids``' width.  The width of the gold encoding
        is a property of the answer, so decoding at that width hands the model the
        length of the string it is being scored on: measured on iter32, the same
        checkpoint emits the value's first token and stops at the inference width
        the agent actually uses (``op.props`` is empty there, so ``P`` collapses to
        1) while scoring 0.78 ``param_exact_match_value`` at the gold width.  One
        number was a measurement and the other was the answer's shape.

        The request token grid is rebuilt here so the emission is conditioned on
        the same tokens the teacher-forced path sees -- a decoder that reads the
        request must read it at decode time too, or train and inference disagree
        (which is exactly the class of defect this decoder replaced).
        """
        hidden = self.backbone(batch.input_ids)
        pooled = self._pool(hidden, batch.attention_mask)
        req_tokens = req_mask = None
        if hidden.shape[1] > 1:
            req_tokens, req_mask, _cand_tokens, _cand_mask = self._token_grids(batch, hidden)
        P = int(self.cfg.max_params) if max_len is None else int(max_len)
        return self.param_decoder.generate(pooled, P, req_tokens, req_mask)
