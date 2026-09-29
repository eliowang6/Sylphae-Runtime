"""Effort control for Sylphae agent (DESIGN.md Section 10).

Effort is a policy parameter, not a prompt. It controls:
- Candidate search depth (how many candidates to retrieve and inspect)
- Enabled exploration tools (whether git history / deep AST search is included)
- Verification depth (syntax/compile only vs affected tests vs full test suite)
- Escalation threshold (minimum confidence required before immediate optimistic execution)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class EffortConfig:
    name: str
    n_candidates: int           # Candidate pool width (e.g. 3 for low, 8-12 for high)
    timeout_seconds: float      # Overall parallel exploration timeout
    include_git_history: bool   # Whether to query git log
    include_ripgrep: bool       # Whether to run ripgrep content search
    include_ast: bool           # Whether to run AST symbol search
    include_neural: bool        # Whether to rerank candidates with the trained target head
    verification_mode: str      # 'syntax_only', 'affected_tests', 'full_tests'
    min_confidence: float       # Threshold to execute immediately vs escalate
    max_patches: int            # Max candidate patches to explore (1 for low)

    def with_neural(self) -> "EffortConfig":
        """A copy of this effort with the neural reranker enabled.

        The head costs one forward pass per candidate batch, so it is opt-in: the
        light CLI commands stay heuristic-free but fast, and the callers that need
        the concept classes (``--require-context``) turn it on explicitly.
        """
        return dataclasses.replace(self, include_neural=True)


LowEffort = EffortConfig(
    name="low",
    n_candidates=3,
    timeout_seconds=0.5,
    include_git_history=False,
    include_ripgrep=True,
    include_ast=True,
    include_neural=False,
    verification_mode="syntax_only",
    min_confidence=0.60,
    max_patches=1,
)

HighEffort = EffortConfig(
    name="high",
    n_candidates=8,
    timeout_seconds=2.0,
    include_git_history=True,
    include_ripgrep=True,
    include_ast=True,
    include_neural=True,
    verification_mode="affected_tests",
    min_confidence=0.80,
    max_patches=3,
)
