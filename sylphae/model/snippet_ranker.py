"""Neural and feature-based relevance ranking for code snippets.

Scores and ranks candidate snippets by task relevance, accurately modeling:
1. Base definition vs. Default theme override precedence
2. Query-specific theme matching
3. Visual dominance & container filling occlusion
4. Unrequested alternative theme penalties
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Sequence

from sylphae.coprocessor.protocol import RelevantSnippet
from sylphae.coprocessor.snippet_extractor import RawCandidateSnippet

# 8 Feature dimensions:
# 0: semantic_relevance
# 1: is_base_definition
# 2: is_default_theme_override
# 3: query_specific_theme_match
# 4: unrequested_theme_penalty
# 5: visual_dominance_score
# 6: container_occlusion_penalty
# 7: minor_subelement_penalty
FEATURE_DIM = 8

# Calibrated reference linear weights (trained by scripts/train_snippet_ranker.py)
DEFAULT_WEIGHTS = [
    2.20,   # semantic_relevance
    1.40,   # is_base_definition
    2.60,   # is_default_theme_override
    3.80,   # query_specific_theme_match
    -2.50,  # unrequested_theme_penalty
    3.20,   # visual_dominance_score
    -1.50,  # container_occlusion_penalty
    -1.80,  # minor_subelement_penalty
]
DEFAULT_BIAS = -0.50


def extract_snippet_features(
    snippet: RawCandidateSnippet,
    query: str,
    all_snippets_in_scope: Sequence[RawCandidateSnippet] | None = None,
) -> list[float]:
    """Compute normalized 8-dimensional feature vector for a candidate snippet."""
    query_lower = query.lower()
    snip_lower = snippet.code_snippet.lower()
    sel_lower = snippet.selector_or_tag.lower()

    # Detect theme in query
    query_theme: str | None = None
    if any(k in query_lower for k in ("暗黑", "深色", "黑夜", "dark", "night")):
        query_theme = "dark"
    elif any(k in query_lower for k in ("明亮", "浅色", "白天", "light", "day")):
        query_theme = "light"
    elif any(k in query_lower for k in ("默认", "default")):
        query_theme = "default"
    else:
        m_t = re.search(r"([a-zA-Z0-9_\-]+)\s*(?:主题|theme)", query_lower)
        if m_t:
            query_theme = m_t.group(1).lower()

    # 0. Semantic relevance
    # Check concept matches
    query_terms = [w for w in re.split(r"[ ,，_.\-:/]+", query_lower) if len(w) >= 2]
    overlap_count = sum(1 for t in query_terms if t in snip_lower or t in sel_lower or t in snippet.file_path.lower())
    semantic_score = min(1.0, 0.4 + 0.3 * overlap_count) if query_terms else 0.5

    # 1. Base definition
    is_base = 1.0 if (snippet.is_container_root and not snippet.theme_name) else 0.0

    # 2. Default theme override
    is_default_theme = 1.0 if snippet.is_default_theme else 0.0

    # 3. Query specific theme match
    is_specific_match = 0.0
    if query_theme and snippet.theme_name:
        if query_theme in snippet.theme_name or snippet.theme_name in query_theme:
            is_specific_match = 1.0

    # 4. Unrequested theme penalty
    is_unrequested_theme = 0.0
    if snippet.theme_name and not snippet.is_default_theme:
        if not (query_theme and (query_theme in snippet.theme_name or snippet.theme_name in query_theme)):
            is_unrequested_theme = 1.0

    # 5. Visual dominance score (element fills container)
    vis_dom = 1.0 if snippet.is_visual_dominant else 0.0

    # 6. Container occlusion penalty
    # If this snippet is the base container, but another candidate in the same scope is visual_dominant
    has_dominant_child = False
    if all_snippets_in_scope:
        has_dominant_child = any(s.is_visual_dominant for s in all_snippets_in_scope if s != snippet)

    is_occluded = 1.0 if (snippet.is_container_root and has_dominant_child) else 0.0

    # 7. Minor subelement penalty
    is_minor = 1.0 if (snippet.kind == "nested_control" and not snippet.is_visual_dominant) else 0.0

    return [
        float(semantic_score),
        float(is_base),
        float(is_default_theme),
        float(is_specific_match),
        float(is_unrequested_theme),
        float(vis_dom),
        float(is_occluded),
        float(is_minor),
    ]


class SnippetRelevanceScorer:
    """Evaluates and ranks candidate snippets with calibrated probabilities."""

    def __init__(self, weights_path: Path | str | None = None):
        self.weights = list(DEFAULT_WEIGHTS)
        self.bias = float(DEFAULT_BIAS)

        if weights_path is None:
            default_art = Path(__file__).resolve().parent / "artifacts" / "snippet_ranker_weights.json"
            if default_art.is_file():
                weights_path = default_art

        if weights_path:
            p = Path(weights_path)
            if p.is_file():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    self.weights = data.get("weights", self.weights)
                    self.bias = data.get("bias", self.bias)
                except Exception:
                    pass

    def score_features(self, features: list[float]) -> float:
        """Compute sigmoid probability score from feature vector."""
        z = self.bias
        for w, f in zip(self.weights, features):
            z += w * f
        # Sigmoid activation with numerical bounds
        z = max(-15.0, min(15.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def rank_snippets(
        self,
        raw_candidates: list[RawCandidateSnippet],
        query: str,
        top_k: int = 5,
    ) -> list[RelevantSnippet]:
        """Rank candidates from highest to lowest importance probability."""
        if not raw_candidates:
            return []

        scored: list[tuple[float, RawCandidateSnippet, list[float]]] = []
        for cand in raw_candidates:
            feats = extract_snippet_features(cand, query, all_snippets_in_scope=raw_candidates)
            prob = self.score_features(feats)
            scored.append((prob, cand, feats))

        # Sort descending by importance probability
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[RelevantSnippet] = []
        for prob, cand, feats in scored[:top_k]:
            rationale = self._generate_rationale(cand, feats, query)
            results.append(RelevantSnippet(
                file_path=cand.file_path,
                start_line=cand.start_line,
                end_line=cand.end_line,
                code_snippet=cand.code_snippet,
                kind=cand.kind,
                importance=round(prob, 3),
                rationale=rationale,
                metadata={
                    "selector": cand.selector_or_tag,
                    "theme": cand.theme_name,
                    "is_visual_dominant": cand.is_visual_dominant,
                },
            ))

        return results

    def _generate_rationale(
        self,
        cand: RawCandidateSnippet,
        feats: list[float],
        query: str,
    ) -> str:
        """Generate human-understandable explanation for ranking probability."""
        if feats[3] > 0.5:  # query_specific_theme_match
            return f"Explicitly requested theme '{cand.theme_name}' override takes highest precedence"
        if feats[5] > 0.5:  # visual_dominance_score
            return f"Element visually dominates the container ({' '.join(cand.classes_or_props[:3])}), covering the base background"
        if feats[6] > 0.5:  # container_occlusion_penalty
            return "Base container background color is largely occluded by a dominant foreground element"
        if feats[2] > 0.5:  # is_default_theme_override
            return "Default theme override takes precedence over base component definition"
        if feats[4] > 0.5:  # unrequested_theme_penalty
            return f"Alternative inactive theme override ({cand.theme_name}) not requested in query"
        if feats[1] > 0.5:  # is_base_definition
            return "Base component container definition"
        if feats[7] > 0.5:  # minor_subelement_penalty
            return "Minor nested sub-element inside container"
        return "Relevant syntactic code block for query"
