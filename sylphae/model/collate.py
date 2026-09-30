"""Batching for the Sylphae policy.

Sequence layout (INTERFACES.md section 4)::

    <bos> <request tokens> <sep>  <cand> <candidate 0> </cand>  <cand> <candidate 1> </cand> ...

Candidate ``c`` is *not* a second encoder -- it is a segment of the same
sequence, and ``cand_index[b, c]`` is the position of its closing ``</cand>``
(the model scores the candidate only after the candidate's own path has been
read into the SSD state -- see ``_encode_one``).
token.  ``target_logits`` is read out from exactly those positions.

``index_provider``
------------------
INTERFACES.md names the argument but does not fix its type.  This module takes
it to be a callable ``index_provider(example, k) -> list[Candidate]`` where each
candidate is duck-typed on ``.path`` / ``.symbols`` / ``.snippet``.  Passing
``None`` falls back to :func:`default_index_provider`, which builds the
candidate set from ``example["repo_files"]``.  See the report: this is an
interpretation, and the ``sylphae/repo`` workstream should confirm it.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "Batch",
    "collate",
    "synth_batch",
    "default_index_provider",
    "SimpleCandidate",
    "param_target_string",
    "param_answerable",
    "param_target_key",
    "param_target_rejection_reason",
    "substitute_value",
    "mem_write_labels",
    "conf_labels",
    "OP_KINDS",
    "OP_TO_ID",
    "N_OPS",
]


# --------------------------------------------------------------------------
# op vocabulary -- prefer the real module, mirror the frozen list otherwise
# --------------------------------------------------------------------------

_FROZEN_OP_KINDS = [
    "no_op",
    "replace_literal",
    "rename_symbol",
    "change_condition",
    "modify_call_args",
    "change_default",
    "add_import",
    "remove_import",
    "add_guard",
    "insert_statement",
    "delete_statement",
    "add_decorator",
    "change_return",
    "modify_string",
    "wrap_try_except",
    "freeform_patch",
]

try:  # pragma: no cover - depends on a sibling workstream landing
    from ..ops.vocab import OP_KINDS as _OP_KINDS_REAL
    from ..ops.vocab import N_OPS as _N_OPS_REAL

    OP_KINDS = list(_OP_KINDS_REAL)
    N_OPS = int(_N_OPS_REAL)
    OPS_VOCAB_SOURCE = "sylphae.ops.vocab"
except Exception:  # noqa: BLE001 - the sibling module may not exist yet
    OP_KINDS = list(_FROZEN_OP_KINDS)
    N_OPS = len(OP_KINDS)
    OPS_VOCAB_SOURCE = "mirror of INTERFACES.md section 1"

OP_TO_ID = {kind: i for i, kind in enumerate(OP_KINDS)}


# --------------------------------------------------------------------------
# Batch
# --------------------------------------------------------------------------


@dataclass
class Batch:
    """Every field a tensor.  Shapes are fixed by INTERFACES.md section 4."""

    input_ids: torch.Tensor       # (B, L) long
    attention_mask: torch.Tensor  # (B, L) long
    cand_index: torch.Tensor       # (B, C) long -- closing </cand> of each candidate
    cand_mask: torch.Tensor       # (B, C) bool
    op_labels: torch.Tensor       # (B,) long
    param_ids: torch.Tensor       # (B, P) long
    param_mask: torch.Tensor      # (B, P) bool
    conf_label: torch.Tensor      # (B,) float
    mem_write_label: torch.Tensor  # (B,) float

    def to(self, device, non_blocking: bool = False) -> "Batch":
        return Batch(
            **{
                f: (t.to(device, non_blocking=non_blocking) if torch.is_tensor(t) else t)
                for f, t in self.__dict__.items()
            }
        )

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])

    def __len__(self) -> int:
        return self.batch_size


# --------------------------------------------------------------------------
# candidate provisioning
# --------------------------------------------------------------------------


@dataclass
class SimpleCandidate:
    path: str
    symbols: list[str]
    snippet: str


def _rng_for(example: dict) -> random.Random:
    """Deterministic per-example RNG.  Never Python's ``hash()`` (randomised)."""
    key = str(example.get("ex_id", "")) + "|" + str(example.get("target_path", ""))
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


def default_index_provider(example: dict, k: int) -> list[SimpleCandidate]:
    """Target path plus deterministic distractors drawn from ``repo_files``.

    The order is shuffled deterministically so the target is not always first --
    otherwise the target head would learn a positional shortcut.  No snippet is
    attached, because only the target's source is available here and attaching
    it would leak the label.
    """
    target = example.get("target_path") or ""
    pool = [p for p in example.get("repo_files", []) if p and p != target]
    rng = _rng_for(example)
    rng.shuffle(pool)
    paths = ([target] if target else []) + pool[: max(0, k - 1)]
    rng.shuffle(paths)
    return [SimpleCandidate(path=p, symbols=[], snippet="") for p in paths]


def _get_candidates(example: dict, index_provider, k: int) -> list[Any]:
    if index_provider is None:
        return default_index_provider(example, k)
    if callable(index_provider):
        cands = index_provider(example, k)
    elif hasattr(index_provider, "candidates"):
        cands = index_provider.candidates(example.get("request", ""), k)
    else:
        raise TypeError(
            "index_provider must be a callable (example, k) -> list[Candidate] or an "
            f"object with .candidates(request, k); got {type(index_provider).__name__}"
        )
    out = []
    for c in list(cands)[:k]:
        if isinstance(c, dict):
            out.append(
                SimpleCandidate(
                    path=str(c.get("path", "")),
                    symbols=list(c.get("symbols", []) or []),
                    snippet=str(c.get("snippet", "") or ""),
                )
            )
        else:
            out.append(c)
    return out


def _shuffle_candidates(cands: list[Any], example: dict,
                        perm_rng: random.Random | None = None) -> list[Any]:
    """Deterministically permute candidates so position carries no label.

    Every retrieval provider returns candidates in its own *score order* --
    ``PathPoolProvider`` sorts by lexical relevance descending, ``RepoIndex``
    likewise.  Left as-is, the target file sits in an early slot for almost every
    example, because the provider's whole job is to rank it first.  The target
    head then learns "early slot is the answer" and reports a healthy
    ``target_hit@k`` that measures nothing but the provider's ordering.

    That was not a hypothesis.  Measured on the trained checkpoint against a real
    validation corpus: the gold occupied the provider's top-3 slots 0.796 of the
    time and the head scored hit@3 0.776 on that same ordering, but with the gold
    placed in each of the 8 slots by hand hit@1 fell from 0.83 (gold first) to
    0.00 (slot 4 onward) and the any-slot mean hit@3 was 0.384 against a random
    0.375.  The head was a positional shortcut, and ``target_hit@k`` could not see
    it because the metric and the shortcut were the same object.

    Shuffling here fixes the leak at the level where it is injected, so it holds
    for every provider.  ``default_index_provider`` already shuffled for exactly
    this reason -- "otherwise the target head would learn a positional shortcut"
    -- but only for the no-provider path, which training never takes.

    The default permutation is *deterministic per example* (seeded from
    ``ex_id``/``request``, never from Python's randomised ``hash()``), which is
    what makes evaluation reproducible -- but it is also a leak of a second
    order: over enough epochs the model can memorise "this request puts the gold
    in slot k" and stop reading the candidates at all.  Training therefore passes
    a live ``perm_rng`` so the permutation is redrawn every epoch, while
    evaluation keeps the deterministic path.
    """
    if len(cands) < 2:
        return list(cands)
    return [cands[i] for i in candidate_permutation(len(cands), example, perm_rng)]

def _permutation(n: int, example: dict, perm_rng: random.Random | None = None) -> list[int]:
    if perm_rng is not None:
        return perm_rng.sample(range(n), n)
    seed = int(
        hashlib.sha256(
            ("cand|" + str(example.get("ex_id", "")) + "|" + str(example.get("request", ""))).encode("utf-8")
        ).hexdigest()[:16],
        16,
    )
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


def candidate_permutation(n: int, example: dict, perm_rng: random.Random | None = None) -> list[int]:
    """The candidate reordering ``_encode_one`` applies, for callers that must
    align their own bookkeeping with the order the model actually saw.

    ``sylphae.eval.harness`` uses this so ``target_hit@k`` scores the model against
    the same candidate sequence it was shown, rather than against the provider's
    pre-shuffle ordering.  Without it the gold index and the logits index two
    different lists and the metric is meaningless.
    """
    if n < 2:
        return list(range(max(0, n)))
    return _permutation(n, example, perm_rng)


def _candidate_text(cand: Any) -> str:
    path = getattr(cand, "path", "") or ""
    symbols = getattr(cand, "symbols", None) or []
    snippet = getattr(cand, "snippet", "") or ""
    parts = [str(path)]
    if symbols:
        parts.append(" ".join(str(s) for s in symbols))
    if snippet:
        parts.append(str(snippet))
    return "\n".join(parts)


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# source context
# --------------------------------------------------------------------------

#: How many source lines of context to keep on each side of the edit span.
_CTX_LINE_PAD = 6


def _source_window_text(example: dict) -> str:
    """A bounded slice of ``old_source`` around the edit.

    The parameter head has to *produce* the new value text, and the op head has to
    recognise the shape of the edit, but neither can do either from a bare path:
    the new value is a local edit of the surrounding code.  Feeding the model the
    source region the change sits in turns parameter generation from a memorisation
    task (which it cannot generalise) into a copy-and-edit task (which it can).
    This is the "repo graph latent + snippet" input the design calls for.

    The window is centered on ``op.span`` (1-based inclusive in the ORIGINAL source)
    so the model is shown exactly the region the executor will replace.  For
    insertion kinds the span is ``(p, p)`` and the window is the lines around the
    insertion point.
    """
    old = example.get("old_source")
    if not isinstance(old, str) or not old:
        return ""
    op = example.get("op") or {}
    span = op.get("span")
    lines = old.split("\n")
    if isinstance(span, (list, tuple)) and len(span) == 2:
        try:
            a = max(1, int(span[0]))
            b = max(a, int(span[1]))
        except (TypeError, ValueError):
            a, b = 1, min(40, len(lines))
    else:
        a, b = 1, min(40, len(lines))
    lo = max(0, a - 1 - _CTX_LINE_PAD)
    hi = min(len(lines), b + _CTX_LINE_PAD)
    return "\n".join(lines[lo:hi])


#: Re-exported from :mod:`sylphae.ops.params`, which owns the single definition of
#: "what does the parameter head emit for this row" and "is that answerable".
#: They are re-exported here so the existing import sites
#: (``from sylphae.model.collate import param_answerable``) keep working, and --
#: the point of the arrangement -- the data-only validator
#: (``scripts/validate_synth_corpus.py``) can ask the same question without
#: paying for a torch import.  The runtime therefore has no edge into
#: ``sylphae.data``; both sides depend on ``sylphae.ops``, which depends on neither.
from ..ops.params import (  # noqa: E402
    _CHUNK_PROP_KEYS,      # noqa: F401 - re-exported; the eval harness mirrors these
    _PARAM_PROP_KEYS,      # noqa: F401 - re-exported
    _VALUE_PAIRS,          # noqa: F401 - re-exported
    param_answerable,
    param_target_key,
    param_target_rejection_reason,
    param_target_string,
    substitute_value,
)


def conf_labels(examples: list[dict]) -> list[float]:
    """1.0 when the example's op is structured *and* round-trips.

    ``detect_op`` guarantees the round-trip invariant for every non-freeform
    result (INTERFACES.md section 1, invariant 1), so ``kind != freeform_patch``
    is normally sufficient.  When ``sylphae.ops.apply`` is importable and the
    sources are present we verify it for real and treat a mismatch as 0.0.
    """
    apply_op = None
    try:  # pragma: no cover - depends on a sibling workstream landing
        from ..ops.apply import apply_op as _apply_op
        from ..ops.spec import OpSpec as _OpSpec

        apply_op = (_apply_op, _OpSpec)
    except Exception:  # noqa: BLE001
        apply_op = None

    out: list[float] = []
    for ex in examples:
        op = ex.get("op") or {}
        kind = op.get("kind")
        if kind is None or kind == "freeform_patch":
            out.append(0.0)
            continue
        old, new = ex.get("old_source"), ex.get("new_source")
        if apply_op is not None and isinstance(old, str) and isinstance(new, str):
            fn, spec_cls = apply_op
            try:
                ok = fn(old, spec_cls.from_dict(op)) == new
            except Exception:  # noqa: BLE001 - an op that cannot apply is not exact
                ok = False
            out.append(1.0 if ok else 0.0)
        else:
            out.append(1.0)
    return out


def mem_write_labels(examples: list[dict]) -> list[float]:
    """1.0 when this example's op kind occurs >= 2 times in the same repository.

    Pure function of the list it is given: no global state, no ``hash()``.
    """
    counts: dict[tuple[str, str], int] = {}
    for ex in examples:
        key = (str(ex.get("repo", "")), str((ex.get("op") or {}).get("kind", "")))
        counts[key] = counts.get(key, 0) + 1
    return [
        1.0
        if counts[(str(ex.get("repo", "")), str((ex.get("op") or {}).get("kind", "")))] >= 2
        else 0.0
        for ex in examples
    ]


# --------------------------------------------------------------------------
# collate
# --------------------------------------------------------------------------


def _encode_one(
    example: dict,
    tok,
    index_provider,
    k: int,
    max_len: int,
    ctx_tokens: int = 0,
    perm_rng: random.Random | None = None,
) -> tuple[list[int], list[int], list[bool], int]:
    """Build one example's ids plus its candidate bookkeeping.

    Returns the ids, the per-candidate score positions, the candidate mask, and
    the gold candidate column *in the order actually encoded* (-1 if the gold was
    not retrieved or was cut by the length budget).  The column is computed here
    rather than before training because it depends on the permutation applied
    here: a precomputed label read off the provider's order names a slot that does
    not hold the gold, which teaches the head nothing but the label marginal.

    When ``ctx_tokens > 0`` a bounded slice of the example's ``old_source`` is
    appended after the candidates, opened by a ``<mask>`` token (id 6, part of the
    frozen special-token set but previously unused).  It is deliberately *outside*
    the ``<cand>`` spans so it never becomes a target-head slot; it only widens what
    the pooled representation -- and therefore the op, parameter and confidence
    heads -- can see.  Because pooling reads the last non-pad position, the source
    tokens sit closest to the decoder's context vector.
    """
    if max_len < 8:
        raise ValueError(f"cfg.max_len must be >= 8 to fit a request and one candidate, got {max_len}")

    head_room = max_len - 4          # <bos> ... <sep> <cand> X </cand>
    req_cap = max(1, min(max_len // 2, head_room))
    req_ids = tok.encode(str(example.get("request", "")), req_cap)

    head = [tok.bos_id] + req_ids + [tok.sep_id]
    seq = list(head)
    remaining = max_len - len(seq)

    # Reserve budget for the source context before the candidates consume the rest,
    # so adding context cannot silently evict every candidate (the target head
    # still needs its slots).
    ctx_budget = 0
    if ctx_tokens > 0:
        ctx_budget = max(0, min(ctx_tokens, remaining // 2))
    cand_room = remaining - (ctx_budget + 1 if ctx_budget else 0)

    cands = _shuffle_candidates(_get_candidates(example, index_provider, k), example, perm_rng)
    n = len(cands)
    gold_path = str(example.get("target_path") or "")
    gold_col = -1
    cand_index: list[int] = []
    cand_mask: list[bool] = []
    if n and cand_room >= 3:
        keep = min(n, max(1, cand_room // 3))
        per = max(1, (cand_room // keep) - 2)
        for col, cand in enumerate(cands[:keep]):
            content = tok.encode(_candidate_text(cand), per)
            seg = [tok.cand_id] + content + [tok.endcand_id]
            if len(seq) + len(seg) > max_len - (ctx_budget + 1 if ctx_budget else 0):
                break
            if gold_path and gold_col < 0 and (str(getattr(cand, "path", "") or "") == gold_path):
                gold_col = col
            # Score at the CLOSING </cand>, not the opening <cand>: the SSD
            # recurrent state at the opening token has not yet read this
            # candidate's path, so a head reading that position cannot see what
            # it is scoring -- it can only predict the label marginal, which the
            # uniform candidate permutation makes ln(k).  Measured: L_target was
            # flat at 2.05 ~= ln(8) for all 2000 steps of iter13 while every
            # other loss term fell normally.  After the shift the head sees the
            # request and the candidate's own path.
            cand_index.append(len(seq) + len(seg) - 1)
            cand_mask.append(True)
            seq.extend(seg)

    if ctx_budget:
        ctx_ids = tok.encode(_source_window_text(example), ctx_budget)
        if ctx_ids and len(seq) + 1 + len(ctx_ids) <= max_len:
            seq.append(tok.mask_id)
            seq.extend(ctx_ids)

    return seq, cand_index, cand_mask, gold_col


def collate(
    examples: list[dict],
    tok,
    index_provider,
    cfg,
    perm_rng: random.Random | None = None,
    gold_out: list[int] | None = None,
) -> Batch:
    """Turn labelled examples into a :class:`Batch`.

    ``cfg`` may be an ``SSDCfg`` or a ``ModelConfig``; ``n_candidates`` and
    ``max_params`` are read off it when present, otherwise they default to 8 and
    32.

    ``perm_rng`` redraws the candidate order for every example.  Training passes
    one so the same example never sees the same candidate layout twice in a row,
    which is what stops the target head from memorising a slot; evaluation omits
    it and keeps the deterministic per-example permutation, so eval runs stay
    reproducible.

    ``gold_out`` receives the gold candidate column for each example in the order
    encoded.  It is the only way to keep the target label aligned with a random
    permutation: a label computed before collate names a slot in the provider's
    order, not the one the model sees.  Training passes a list and uses it as the
    step's target labels; evaluation ignores it (its metrics derive the label the
    same way, deterministically).
    """
    examples = list(examples)
    if not examples:
        raise ValueError("collate got an empty example list")

    max_len = int(cfg.max_len)
    k = int(getattr(cfg, "n_candidates", 8))
    p_max = int(getattr(cfg, "max_params", 32))
    ctx_tokens = int(getattr(cfg, "ctx_tokens", 0) or 0)

    seqs: list[list[int]] = []
    cand_idx: list[list[int]] = []
    cand_msk: list[list[bool]] = []
    for ex in examples:
        s, ci, cm, gold = _encode_one(ex, tok, index_provider, k, max_len, ctx_tokens, perm_rng)
        seqs.append(s)
        cand_idx.append(ci)
        cand_msk.append(cm)
        if gold_out is not None:
            gold_out.append(gold)

    C = max((len(ci) for ci in cand_idx), default=0)
    C = max(C, 1)  # the target head always has at least one slot

    L = max(len(s) for s in seqs)
    B = len(seqs)

    input_ids = torch.full((B, L), tok.pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, L), dtype=torch.long)
    cand_index = torch.zeros((B, C), dtype=torch.long)
    cand_mask = torch.zeros((B, C), dtype=torch.bool)

    for b, s in enumerate(seqs):
        input_ids[b, : len(s)] = torch.tensor(s, dtype=torch.long)
        attention_mask[b, : len(s)] = 1
        for c, pos in enumerate(cand_idx[b]):
            cand_index[b, c] = pos
            cand_mask[b, c] = True
        for c in range(len(cand_idx[b]), C):
            cand_mask[b, c] = False

    op_labels = torch.tensor(
        [OP_TO_ID.get(str((ex.get("op") or {}).get("kind", "")), OP_TO_ID["freeform_patch"])
         for ex in examples],
        dtype=torch.long,
    )

    param_strs = [param_target_string(ex) for ex in examples]
    #: Rows whose value is not in the request cannot be learned from the request
    #: (see :func:`param_answerable`).  Masking them out of ``L_param`` is what
    #: stops the parameter head being supervised on the unanswerable rows that
    #: previously taught it the old-source prior (12.11/12.12) -- on the current
    #: corpus that is 13,888 of 15,382 parameterised rows (90.3%).  The row is still
    #: fully supervised for the other heads -- the target, op, conf and mem labels
    #: are all sound -- only the parameter emission is declared unlearnable.
    #: ``False`` restores the old behaviour for an ablation.
    answerable_only = bool(getattr(cfg, "param_answerable_only", True))
    if answerable_only:
        for b, ex in enumerate(examples):
            if param_strs[b] and not param_answerable(ex):
                param_strs[b] = ""
    encoded = tok.encode_batch(param_strs, p_max)
    #: Every non-empty target is terminated with ``<eos>``, and the terminator is
    #: supervised (it sits inside ``param_mask``).  Without it "when to stop" has
    #: no gradient at all: ``encode_batch`` adds no specials, so a gold value of
    #: ``9000`` trained as ``['9','000']`` and the decoder emitted
    #: ``['9','000','0','9','000',...]`` forever at any width above 2 (MEASURED,
    #: ``scratch/probe_p1_truncation.py``: iter32 reproduces the value's first two
    #: tokens and then never stops).  One slot is reserved for the terminator so
    #: widening ``max_params`` cannot silently evict it.
    encoded = [
        (ids[: max(0, p_max - 1)] + [tok.eos_id]) if ids else []
        for ids in encoded
    ]
    P = max(1, max((len(e) for e in encoded), default=0))
    param_ids = torch.full((B, P), tok.pad_id, dtype=torch.long)
    param_mask = torch.zeros((B, P), dtype=torch.bool)
    for b, ids in enumerate(encoded):
        if ids:
            param_ids[b, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            param_mask[b, : len(ids)] = True

    conf = conf_labels(examples)
    # A precomputed label (computed over the whole training set) wins, so the
    # value does not depend on how the minibatch happens to be drawn.
    mem = mem_write_labels(examples)
    for b, ex in enumerate(examples):
        if ex.get("mem_write_label") is not None:
            mem[b] = float(ex["mem_write_label"])

    return Batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        cand_index=cand_index,
        cand_mask=cand_mask,
        op_labels=op_labels,
        param_ids=param_ids,
        param_mask=param_mask,
        conf_label=torch.tensor(conf, dtype=torch.float32),
        mem_write_label=torch.tensor(mem, dtype=torch.float32),
    )


def synth_batch(
    batch_size: int,
    seq_len: int,
    n_candidates: int,
    max_params: int,
    vocab_size: int,
    device,
    seed: int = 0,
) -> Batch:
    """A synthetic but shape-correct batch: the caller wants the *shapes*, not data.

    Lives here, next to :class:`Batch`, rather than in the training benchmark
    because that is what it is -- a way to get a correctly shaped tensor bundle
    without encoding anything.  Both sides need it and neither owns it: the
    benchmark measures compute, and ``torch.compile`` needs a representative
    batch to warm up on before it will specialise a graph.  Putting it in the
    benchmark would have made the inference half import a training-only module to
    compile itself.
    """
    g = torch.Generator().manual_seed(seed)
    cand_index = torch.zeros(batch_size, n_candidates, dtype=torch.long)
    for b in range(batch_size):
        cand_index[b] = torch.randint(0, seq_len, (n_candidates,), generator=g)
    return Batch(
        input_ids=torch.randint(7, vocab_size, (batch_size, seq_len), generator=g).to(device),
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
        cand_index=cand_index.to(device),
        cand_mask=torch.ones(batch_size, n_candidates, dtype=torch.bool, device=device),
        op_labels=torch.randint(0, 16, (batch_size,), generator=g).to(device),
        param_ids=torch.randint(7, vocab_size, (batch_size, max_params), generator=g).to(device),
        param_mask=torch.ones(batch_size, max_params, dtype=torch.bool, device=device),
        conf_label=torch.randint(0, 2, (batch_size,), generator=g).float().to(device),
        mem_write_label=torch.randint(0, 2, (batch_size,), generator=g).float().to(device),
    )
