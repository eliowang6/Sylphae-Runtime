"""Neural candidate ranking: the trained target head as an exploration tool.

The lexical and AST tools are the retrieval layer.  They rank by surface overlap
between the request and the path, which is why a request that names a *concept*
rather than an identifier falls through: the tokens shared between a Chinese
concept word and an English file path are the empty set, so every file scores 0.0
and the tie-break, not the request, picks the candidates.

This module is the model-path replacement for that ranking.  It scores a pool of
candidate paths with the trained target head, which was fit on (request,
candidates) -> gold pairs and therefore carries associations the lexical score
cannot express.  It is a *reranker*, not a retriever: it needs a pool to score,
and the pool still comes from the cheap tools.

Model loading is lazy and cached.  ``sylphae.agent.tools`` is imported by the light
CLI commands, and importing torch at module scope would cost the sub-second
budget the whole layer exists to protect (AGENTS.md section 2).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

#: Where a trained run lives when the caller does not say.  A run directory holds
#: ``config.json``, ``checkpoint.safetensors`` and ``tokenizer.json``; see
#: :func:`sylphae.model.io.load_weights`.
#:
#: There is no default *checkpoint* -- the runtime ships without weights, and the
#: lexical and AST tools are the supported zero-weight path.  What is
#: configurable here is only where to look, so that a user who trains one does not
#: have to thread ``run_dir`` through every call site.  ``SYLPHAE_RUN_DIR`` wins;
#: otherwise the first existing directory under ``runs/`` is used, and if there is
#: none the ranker reports itself unavailable, which is the same outcome as not
#: having a model at all.
_RUN_DIR_ENV = "SYLPHAE_RUN_DIR"
_RUNS_ROOT = "runs"
_MAX_POOL = 64
_BATCH = 8


def _default_run() -> str | None:
    """Best guess at a run directory, or ``None`` if there is no plausible one."""
    import os

    env = os.environ.get(_RUN_DIR_ENV)
    if env:
        return env
    root = Path(_RUNS_ROOT)
    if not root.is_dir():
        return None
    cands = sorted(d for d in root.iterdir()
                   if d.is_dir() and (d / "checkpoint.safetensors").is_file())
    return str(cands[-1]) if cands else None


def _entry_path(entry: Any) -> str:
    """A candidate is a bare path or a ``(path, symbols, snippet)`` triple."""
    if isinstance(entry, (tuple, list)):
        return str(entry[0])
    return str(entry)


def _entry_symbols(entry: Any) -> Sequence[str]:
    if isinstance(entry, (tuple, list)) and len(entry) > 1:
        syms = entry[1]
        return list(syms) if syms else []
    return []


def _entry_snippet(entry: Any) -> str:
    if isinstance(entry, (tuple, list)) and len(entry) > 2:
        return str(entry[2] or "")
    return ""


@dataclass(frozen=True)
class NeuralScore:
    """One candidate's neural score, in the caller's input order."""

    path: str
    score: float


class NeuralRanker:
    """Scores candidate paths with the target head of a trained run."""

    _instances: dict[str, "NeuralRanker"] = {}

    def __init__(self, model: Any, tok: Any, run_dir: Path):
        self.model = model
        self.tok = tok
        self.run_dir = Path(run_dir)
        self._dev = "cuda" if _torch_cuda_available() else "cpu"
        self.model.to(self._dev)
        self.model.eval()

    # -- construction -------------------------------------------------- #

    @classmethod
    def for_run(cls, run_dir: Path | str | None = None) -> "NeuralRanker":
        """Load (and cache) the ranker for ``run_dir``.

        ``run_dir`` defaults to :func:`_default_run`: ``$SYLPHAE_RUN_DIR``, else
        the newest run under ``runs/``.  A missing or unparseable checkpoint
        raises :class:`NeuralUnavailable` so callers can fall back to the lexical
        tools rather than crashing an exploration.
        """
        run_dir = Path(run_dir or _default_run() or (Path.cwd() / _RUNS_ROOT / "absent"))
        key = str(run_dir.resolve())
        cached = cls._instances.get(key)
        if cached is not None:
            return cached
        inst = cls._load(run_dir)
        cls._instances[key] = inst
        return inst

    @staticmethod
    def _load(run_dir: Path) -> "NeuralRanker":
        import torch  # noqa: PLC0415 - lazy by design

        from sylphae.model.collate import SimpleCandidate, collate, candidate_permutation
        from sylphae.model.config import ModelConfig
        from sylphae.model.io import load_weights
        from sylphae.model.policy import SylphaeModel
        from sylphae.model.tokenizer import CodeTokenizer
        from sylphae.ops.vocab import N_OPS

        cfg_path = run_dir / "config.json"
        if not cfg_path.is_file():
            raise NeuralUnavailable(f"no config.json under {run_dir}")
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        if "model_config" in raw:
            mc = dict(raw["model_config"])
        else:
            mc = {
                k: raw[k]
                for k in ("d_model", "n_layers", "d_state", "n_heads", "chunk_size",
                          "vocab_size", "max_len", "max_params", "ctx_tokens")
                if k in raw
            }
            mc["d_ff"] = raw.get("d_ff") or raw.get("d_model", 384)
        mc["n_ops"] = int(N_OPS)
        ckpt = run_dir / "checkpoint.safetensors"
        if not ckpt.is_file():
            raise NeuralUnavailable(f"no checkpoint.safetensors under {run_dir}")
        model = SylphaeModel(ModelConfig(**mc))
        load_weights(ckpt, model=model)
        model.eval()
        tok_path = Path("sylphae/model/artifacts/tokenizer.json")
        if not tok_path.is_file():
            raise NeuralUnavailable(f"no tokenizer artifact at {tok_path}")
        inst = NeuralRanker(model, CodeTokenizer.load(str(tok_path)), run_dir)
        # keep the imports reachable for callers that introspect the module
        inst._collate = collate  # type: ignore[attr-defined]
        inst._SimpleCandidate = SimpleCandidate  # type: ignore[attr-defined]
        inst._candidate_permutation = candidate_permutation  # type: ignore[attr-defined]
        inst._torch = torch  # type: ignore[attr-defined]
        return inst

    # -- scoring ------------------------------------------------------- #

    def score(self, request: str,
              candidates: Sequence[str | tuple[str, Sequence[str], str]]
              ) -> list[NeuralScore]:
        """Score ``candidates`` for ``request``, returned in the input order.

        ``collate`` permutes the candidate slots before encoding, so the head's
        ``target_logits`` index a permuted list; the scores are mapped back to the
        caller's order here, exactly as the evaluation harness does.  Returning
        them in the input order is what lets the explorer fuse them with the other
        tools' per-path terms.
        """
        if not candidates:
            return []
        k = int(getattr(self.model.cfg, "n_candidates", _BATCH))
        out: list[NeuralScore] = []
        for start in range(0, len(candidates), k):
            chunk = list(candidates[start:start + k])
            scores = self._score_chunk(request, chunk)
            out.extend(NeuralScore(path=p, score=s)
                       for p, s in zip((_entry_path(c) for c in chunk), scores))
        return out

    def _score_chunk(self, request: str,
                     chunk: list[str | tuple[str, Sequence[str], str]]) -> list[float]:
        torch = self._torch
        probe = {"request": request, "repo_files": []}
        cands = [self._SimpleCandidate(path=_entry_path(c),
                                       symbols=list(_entry_symbols(c)),
                                       snippet=_entry_snippet(c))
                 for c in chunk]
        batch = self._collate(
            [probe], self.tok,
            lambda e, kk: list(cands)[:kk],
            self.model.cfg,
        ).to(self._dev)
        with torch.no_grad():
            out, _ = self.model(batch)
        logits = out.target_logits[0].float().tolist()
        # slot j of the logits is chunk[perm[j]]: the permutation collate applied
        perm = self._candidate_permutation(len(chunk), probe)
        n = min(len(perm), len(logits))
        # entries may repeat a path (a padded pool can hold the same file twice),
        # so key by index, not by path -- a dict keyed on the entry would collapse
        # duplicates and return the wrong score for one of them.
        by_index = {perm[j]: logits[j] for j in range(n)}
        return [by_index.get(i, float("-inf")) for i in range(len(chunk))]


def _torch_cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch may be absent in minimal envs
        return False


class NeuralUnavailable(RuntimeError):
    """No usable trained checkpoint.  Callers should fall back to lexical tools."""


def neural_rank_search(
    request: str,
    pool: Sequence[str | tuple[str, Sequence[str], str]],
    run_dir: Path | str | None = None,
    max_results: int = 16,
) -> list[tuple[str, float]]:
    """Rank ``pool`` with the trained target head.

    Each entry is a bare path or a ``(path, symbols, snippet)`` triple.  The triple
    form matters: the head is trained on candidates carrying real file content, and
    scoring it on empty paths measures a path-only model that no longer exists.

    Returns ``(path, score)`` pairs sorted by score descending.  When no checkpoint
    is available this returns an empty list rather than raising, so an exploration
    that has no neural fallback degrades to the lexical tools instead of failing.
    """
    if not pool:
        return []
    try:
        ranker = NeuralRanker.for_run(run_dir)
    except NeuralUnavailable as exc:
        logger.debug("neural ranker unavailable (%s); falling back to lexical", exc)
        return []
    scored = ranker.score(request, list(pool)[:_MAX_POOL])
    scored.sort(key=lambda s: -s.score)
    return [(s.path, float(s.score)) for s in scored[:max_results]]
