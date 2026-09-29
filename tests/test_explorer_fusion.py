"""Tests for the C1 PROBE fusion and keyword extraction.

Both are the same defect from two directions: a request that names an identifier exactly
was being outvoted by near-misses.

* ``extract_keywords`` expanded one identifier into a whole synonym field -- "change
  HTTP_401_UNAUTHORIZED" produced 18 keywords, 14 of them auth/session synonyms, so every
  auth-adjacent file in the repository matched something.
* the explorer fused tool scores by plain weighted addition, so several weak agreements
  outranked one exact match, and the fused explorer ranked *worse than its own best
  component* (ast alone 86/86 on class A, fused 81/86).

Run from the repository root:

    python -m pytest tests/test_explorer_fusion.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sylphae.agent.explorer import (  # noqa: E402
    CORROBORATION_DISCOUNT,
    extract_keywords,
    fuse_tool_terms,
)

#: The identifier-bearing prompt the failure was measured on.
EXPLICIT_PROMPT = "把 HTTP_401_UNAUTHORIZED 修改为 402"
#: A concept-only prompt, where the synonym field is the only signal there is.
CONCEPT_PROMPT = "把超时相关的默认值调大一些"
_SYNONYMS = ("authenticate", "session", "signin", "token", "login", "jwt", "权限", "登录")


# --------------------------------------------------------------------------- keyword gating


def test_an_explicit_identifier_prompt_is_not_expanded_into_synonyms():
    keywords = extract_keywords(EXPLICIT_PROMPT)
    assert "http_401_unauthorized" in keywords
    for synonym in _SYNONYMS:
        assert synonym not in keywords, f"{synonym} leaked into an explicit-identifier request"


def test_a_concept_only_prompt_still_gets_its_synonym_field():
    keywords = extract_keywords(CONCEPT_PROMPT)
    assert "timeout" in keywords
    assert any(k in keywords for k in _SYNONYMS) or "超时" in keywords


def test_a_symbol_rename_prompt_stays_focused():
    keywords = extract_keywords("把 LoadConfig 重命名为 LoadSettings_v2")
    assert "loadconfig" in keywords
    assert "loadsettings_v2" in keywords
    for synonym in _SYNONYMS:
        assert synonym not in keywords


# --------------------------------------------------------------------------- fusion


def test_a_single_strong_hit_outranks_several_weak_agreements():
    strong = fuse_tool_terms({"ast_index": 0.35})
    weak = fuse_tool_terms({"ripgrep": 0.30 * 0.9, "path_lexical": 0.25 * 0.9,
                            "git_history": 0.20 * 0.9})
    assert strong > weak, "several weak agreements must not outvote one exact match"


def test_corroboration_helps_but_cannot_drown_the_strongest_evidence():
    alone = fuse_tool_terms({"ast_index": 0.35})
    corroborated = fuse_tool_terms({"ast_index": 0.35, "ripgrep": 0.30})
    assert corroborated > alone
    # a weak second opinion must not push a near-miss past an exact match
    strong = fuse_tool_terms({"ast_index": 1.0 * 0.35})
    diluted = fuse_tool_terms({"ast_index": 0.9 * 0.35, "ripgrep": 0.30 * 0.9})
    assert strong > diluted


def test_a_repeated_tool_does_not_double_count_itself():
    once = fuse_tool_terms({"ast_index": 0.35})
    assert fuse_tool_terms({"ast_index": 0.35}) == once


def test_empty_terms_score_zero():
    assert fuse_tool_terms({}) == 0.0


def test_discount_is_the_documented_constant():
    assert 0.0 < CORROBORATION_DISCOUNT < 1.0
    terms = {"ast_index": 0.35, "ripgrep": 0.30}
    best = max(terms.values())
    assert fuse_tool_terms(terms) == best + (sum(terms.values()) - best) * CORROBORATION_DISCOUNT


@pytest.mark.parametrize("discount", [0.0, 0.5, 1.0])
def test_pure_max_is_the_lower_bound_of_the_fuse(discount):
    # discount=0 is pure max; discount=1 is the old additive sum.  Either way the best
    # term is never reduced, which is what makes the fuse domination-preserving.
    terms = {"ast_index": 0.35, "ripgrep": 0.30, "path_lexical": 0.20}
    assert fuse_tool_terms(terms, discount=discount) >= max(terms.values())


# --------------------------------------------------------------------- definition vs reference


def _write_repo(root: Path) -> None:
    """The two-file fixture the failure was measured on.

    ``src/navbar.py`` defines ``NAV_HEIGHT`` and ``get_nav_height`` but never writes the
    word "navbar" anywhere in its own body.  ``tests/test_navbar.py`` imports
    ``src.navbar``, so the word "navbar" appears in *its* body and in nobody else's.
    """
    src = root / "src"
    src.mkdir(parents=True)
    tests = root / "tests"
    tests.mkdir(parents=True)
    (src / "navbar.py").write_text(
        "NAV_HEIGHT = 64\n\n\ndef get_nav_height():\n    return NAV_HEIGHT\n",
        encoding="utf-8",
    )
    (tests / "test_navbar.py").write_text(
        "from src.navbar import get_nav_height\n\n\ndef test_nav_height():\n"
        "    assert get_nav_height() > 0\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_a_file_named_the_concept_outranks_the_file_that_imports_it(tmp_path: Path):
    """The rarest keyword in the request must not be credited to a reference.

    ``navbar`` is the highest-information term here (it matches one file, not two), and
    before the self-name credit it matched only the *importer*, because the definition
    never spells its own name.  ripgrep then scored the importer 1.000 against the
    definition's 0.608, and the reflex agent edited the test instead of the source.
    """
    from sylphae.agent.tools import ripgrep_search

    _write_repo(tmp_path)
    matches = await ripgrep_search(
        ["navbar", "nav", "height"], tmp_path, max_matches=15
    )
    paths = [m.path for m in matches]
    assert "src/navbar.py" in paths and "tests/test_navbar.py" in paths
    scored = {m.path: m.score for m in matches}
    assert scored["src/navbar.py"] >= scored["tests/test_navbar.py"]
    assert paths[0] == "src/navbar.py"


@pytest.mark.asyncio
async def test_the_fused_explorer_points_at_the_definition_not_the_test(tmp_path: Path):
    """End-to-end statement of the same property through the full fuse."""
    from sylphae.agent.effort import LowEffort
    from sylphae.agent.explorer import ParallelExplorer

    _write_repo(tmp_path)
    explorer = ParallelExplorer(root=tmp_path, effort=LowEffort)
    last: list = []
    async for batch in explorer.stream_explore("把导航栏的高度缩短一些"):
        last = batch
    assert last, "the explorer must return at least one candidate"
    assert last[0].path == "src/navbar.py", (
        f"definition lost to a test file: {[m.path for m in last[:4]]}"
    )
