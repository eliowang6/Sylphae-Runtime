"""Sylphae Agent: System-1 Reflex Agent runtime with parallel exploration and optimistic execution."""

from .effort import EffortConfig, HighEffort, LowEffort
from .explorer import CandidateResult, ParallelExplorer
from .tools import ast_search, git_history_search, path_lexical_search, ripgrep_search

__all__ = [
    "ReflexAgent",
    "ReflexResult",
    "ParallelExplorer",
    "CandidateResult",
    "EffortConfig",
    "LowEffort",
    "HighEffort",
    "ripgrep_search",
    "ast_search",
    "git_history_search",
    "path_lexical_search",
]


def __getattr__(name: str):
    if name in ("ReflexAgent", "ReflexResult"):
        from .reflex import ReflexAgent, ReflexResult
        return ReflexAgent if name == "ReflexAgent" else ReflexResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
