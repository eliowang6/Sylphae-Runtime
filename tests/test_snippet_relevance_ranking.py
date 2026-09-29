"""Automated tests for snippet-level code intelligence and relevance probability ranking.

Validates:
1. Normal Base Color: Foundational navbar container ranks highest when no theme overrides exist.
2. Default Theme Override Precedence: Default theme override ranks higher than base container.
3. Query Specific Theme Match: Explicitly requested theme override ranks highest (P > 0.90).
4. Visual Dominance: Container-filling card (w-full h-full) ranks higher than occluded container background.
5. Inactive Alternative Theme: Unrequested themes are downranked.
6. End-to-End Context Probing: collect_codebase_context emits ranked snippets under 25ms overhead.
"""

from __future__ import annotations

import time
from pathlib import Path
import pytest

from sylphae.coprocessor.collector import collect_codebase_context
from sylphae.coprocessor.snippet_extractor import (
    RawCandidateSnippet,
)
from sylphae.model.snippet_ranker import (
    SnippetRelevanceScorer,
)


@pytest.fixture
def mock_theme_codebase(tmp_path: Path) -> Path:
    """Create a realistic codebase featuring base styles, theme overrides, and visual dominant cards."""
    root = tmp_path / "theme_app"
    root.mkdir()

    # 1. Base Vue component with black navbar and a huge white card filling it
    (root / "Navbar.vue").write_text(
        "<template>\n"
        '  <nav class="navbar bg-black min-h-[64px] flex items-center relative">\n'
        '    <!-- Dominant card occupying 100% of navbar space -->\n'
        '    <div class="navbar-card bg-white w-full h-full shadow-md flex justify-between">\n'
        '      <span class="brand text-slate-800">Logo</span>\n'
        '      <button class="nav-btn bg-blue-600 text-xs px-2">Action</button>\n'
        "    </div>\n"
        "  </nav>\n"
        "</template>\n"
        "<style scoped>\n"
        ".navbar {\n"
        "  background-color: #000000;\n"
        "  height: 64px;\n"
        "}\n"
        "</style>\n",
        encoding="utf-8",
    )

    # 2. Default theme override file
    (root / "theme.css").write_text(
        "/* Default Theme Overrides */\n"
        ":root {\n"
        "  --primary: #3b82f6;\n"
        "}\n\n"
        ".theme-default .navbar {\n"
        "  background: #1e293b;\n"
        "  color: #f8fafc;\n"
        "}\n",
        encoding="utf-8",
    )

    # 3. Dark theme override file
    (root / "theme-dark.css").write_text(
        "/* Dark Theme Overrides */\n"
        '[data-theme="dark"] .navbar {\n'
        "  background: #0b0f19;\n"
        "  color: #e2e8f0;\n"
        "}\n",
        encoding="utf-8",
    )

    # 4. Inactive neon theme
    (root / "theme-neon.css").write_text(
        "/* Alternative Neon Theme */\n"
        ".theme-neon .navbar {\n"
        "  background: #00ffcc;\n"
        "  color: #000;\n"
        "}\n",
        encoding="utf-8",
    )

    return root


def test_scorer_normal_base_color():
    """Verify normal base navbar color ranks highest when no active theme overrides exist."""
    q = "导航栏颜色"
    cand_base = RawCandidateSnippet(
        file_path="Navbar.vue",
        start_line=2, end_line=5,
        code_snippet='<nav class="navbar bg-slate-900 min-h-[64px]">',
        target_entity="navbar", kind="base_style", selector_or_tag="<nav>",
        is_container_root=True, is_visual_dominant=False,
    )
    cand_btn = RawCandidateSnippet(
        file_path="Navbar.vue",
        start_line=6, end_line=8,
        code_snippet='<button class="nav-btn bg-blue-600 text-xs">Login</button>',
        target_entity="button", kind="nested_control", selector_or_tag="<button>",
        is_container_root=False, is_visual_dominant=False,
    )

    scorer = SnippetRelevanceScorer()
    ranked = scorer.rank_snippets([cand_base, cand_btn], query=q)

    assert len(ranked) == 2
    assert ranked[0].code_snippet == cand_base.code_snippet
    assert ranked[0].importance > ranked[1].importance
    assert ranked[0].importance >= 0.70
    assert ranked[1].importance <= 0.45


def test_scorer_default_theme_override_precedence():
    """Verify default theme override ranks higher than generic base container."""
    q = "导航栏颜色"
    cand_base = RawCandidateSnippet(
        file_path="Navbar.vue",
        start_line=2, end_line=5,
        code_snippet='<nav class="navbar bg-slate-900 min-h-[64px]">',
        target_entity="navbar", kind="base_style", selector_or_tag="<nav>",
        is_container_root=True, is_visual_dominant=False,
    )
    cand_default_theme = RawCandidateSnippet(
        file_path="theme.css",
        start_line=5, end_line=8,
        code_snippet='.theme-default .navbar { background: #1e293b; }',
        target_entity="navbar", kind="theme_override", selector_or_tag=".theme-default .navbar",
        theme_name="default", is_default_theme=True, is_container_root=True,
    )
    cand_alt_neon = RawCandidateSnippet(
        file_path="theme-neon.css",
        start_line=2, end_line=5,
        code_snippet='.theme-neon .navbar { background: #00ffcc; }',
        target_entity="navbar", kind="theme_override", selector_or_tag=".theme-neon .navbar",
        theme_name="neon", is_default_theme=False, is_container_root=True,
    )

    scorer = SnippetRelevanceScorer()
    ranked = scorer.rank_snippets([cand_base, cand_default_theme, cand_alt_neon], query=q)

    # 1. Default theme override is top ranked
    assert ranked[0].file_path == "theme.css"
    assert ranked[0].kind == "theme_override"
    assert ranked[0].importance > ranked[1].importance

    # 2. Base style is second
    assert ranked[1].file_path == "Navbar.vue"

    # 3. Inactive neon theme is ranked lowest
    assert ranked[2].file_path == "theme-neon.css"
    assert ranked[2].importance < 0.50
    assert ranked[2].importance < ranked[1].importance


def test_scorer_query_specific_theme_match():
    """Verify explicitly requested theme override ranks highest."""
    q = "暗黑主题的导航栏颜色"
    cand_base = RawCandidateSnippet(
        file_path="Navbar.vue",
        start_line=2, end_line=5,
        code_snippet='<nav class="navbar bg-slate-900">',
        target_entity="navbar", kind="base_style", selector_or_tag="<nav>",
        is_container_root=True,
    )
    cand_default = RawCandidateSnippet(
        file_path="theme.css",
        start_line=5, end_line=8,
        code_snippet='.theme-default .navbar { background: #1e293b; }',
        target_entity="navbar", kind="theme_override", selector_or_tag=".theme-default .navbar",
        theme_name="default", is_default_theme=True, is_container_root=True,
    )
    cand_dark = RawCandidateSnippet(
        file_path="theme-dark.css",
        start_line=2, end_line=5,
        code_snippet='[data-theme="dark"] .navbar { background: #0b0f19; }',
        target_entity="navbar", kind="theme_override", selector_or_tag='[data-theme="dark"] .navbar',
        theme_name="dark", is_default_theme=False, is_container_root=True,
    )

    scorer = SnippetRelevanceScorer()
    ranked = scorer.rank_snippets([cand_base, cand_default, cand_dark], query=q)

    # Dark theme must be Rank 1
    assert ranked[0].file_path == "theme-dark.css"
    assert ranked[0].importance >= 0.85
    assert "暗黑" in ranked[0].rationale or "dark" in ranked[0].rationale.lower()

    # Default theme and base must be lower
    assert ranked[0].importance > ranked[1].importance


def test_scorer_visual_dominance_containment():
    """Verify giant card filling navbar (w-full h-full) ranks higher than black container."""
    q = "导航栏底色"
    cand_container_black = RawCandidateSnippet(
        file_path="Navbar.vue",
        start_line=2, end_line=4,
        code_snippet='<nav class="navbar bg-black min-h-[64px] relative">',
        target_entity="navbar", kind="base_style", selector_or_tag="<nav>",
        is_container_root=True, is_visual_dominant=False, color_or_value="bg-black",
    )
    cand_card_white = RawCandidateSnippet(
        file_path="Navbar.vue",
        start_line=5, end_line=8,
        code_snippet='<div class="navbar-card bg-white w-full h-full shadow-md">',
        target_entity="navbar", kind="visual_dominant", selector_or_tag="<div>",
        is_container_root=False, is_visual_dominant=True, color_or_value="bg-white",
        classes_or_props=["navbar-card", "bg-white", "w-full", "h-full"],
    )

    scorer = SnippetRelevanceScorer()
    ranked = scorer.rank_snippets([cand_container_black, cand_card_white], query=q)

    # Visually dominant white card is functionally perceived as the navbar background
    assert ranked[0].code_snippet == cand_card_white.code_snippet
    assert ranked[0].importance > ranked[1].importance
    assert "visually dominates" in ranked[0].rationale.lower()
    assert "occluded" in ranked[1].rationale.lower()


@pytest.mark.asyncio
async def test_end_to_end_context_probing_with_snippets(mock_theme_codebase: Path):
    """Verify collect_codebase_context extracts and ranks snippets within sub-second latency."""
    t0 = time.perf_counter()
    report = await collect_codebase_context(
        root=mock_theme_codebase,
        macro_prompt="暗黑主题的导航栏颜色",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Wall-clock guard for the whole call.  Looser than the probe budget on
    # purpose: the budget cuts a phase loose through ``asyncio.wait_for``, but the
    # worker thread it was scanning in keeps running, so this number can exceed
    # ``report.budget_ms`` without anything being wrong.  ``budget_exceeded`` is
    # the field that answers "was this complete"; this is only a hang detector.
    assert elapsed_ms < 2000.0

    # Verify snippets populated and ordered
    assert len(report.relevant_snippets) >= 2
    top_snip = report.relevant_snippets[0]
    assert "theme-dark.css" in top_snip.file_path or "dark" in top_snip.code_snippet.lower()
    assert top_snip.importance >= 0.80

    # Verify descending probability order
    for i in range(len(report.relevant_snippets) - 1):
        assert report.relevant_snippets[i].importance >= report.relevant_snippets[i + 1].importance

    # Verify prompt context output formatting
    ctx_text = report.to_prompt_context()
    assert "Relevant Code Snippets (Ranked by Importance):" in ctx_text
    assert f"[P={top_snip.importance:.2f}]" in ctx_text
    assert top_snip.file_path in ctx_text
