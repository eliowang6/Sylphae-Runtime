"""Tests for parallel exploration and async tool execution."""

from pathlib import Path

import pytest

from sylphae.agent.effort import LowEffort
from sylphae.agent.explorer import ParallelExplorer, extract_keywords
from sylphae.agent.tools import ast_search, ripgrep_search
from sylphae.repo.index import RepoIndex


@pytest.fixture
def sample_tree(tmp_path: Path) -> Path:
    """Create a mock repository tree with UI components and utils."""
    comp_dir = tmp_path / "src" / "components"
    comp_dir.mkdir(parents=True)
    (comp_dir / "Navbar.py").write_text(
        "NAV_HEIGHT = 64\n"
        "NAV_PADDING = 16\n\n"
        "class Navbar:\n"
        "    def __init__(self, height=NAV_HEIGHT):\n"
        "        self.height = height\n",
        encoding="utf-8",
    )
    (comp_dir / "Header.py").write_text(
        "class Header:\n"
        "    def render(self):\n"
        "        return 'Header'\n",
        encoding="utf-8",
    )
    utils_dir = tmp_path / "src" / "utils"
    utils_dir.mkdir(parents=True)
    (utils_dir / "config.py").write_text(
        "TIMEOUT_SECONDS = 30\n"
        "MAX_RETRIES = 3\n",
        encoding="utf-8",
    )
    return tmp_path


def test_extract_keywords():
    kw = extract_keywords("把导航栏的高度缩短一些")
    assert "navbar" in kw
    assert "height" in kw
    assert "导航栏" in kw


@pytest.mark.asyncio
async def test_ripgrep_search_finds_keyword(sample_tree: Path):
    matches = await ripgrep_search(["NAV_HEIGHT", "Navbar"], sample_tree)
    assert len(matches) >= 1
    paths = [m.path for m in matches]
    assert any("Navbar.py" in p for p in paths)


@pytest.fixture
def ranking_tree(tmp_path: Path) -> Path:
    """A tree where the whole identifier and its subtokens live in different files.

    ``extract_keywords`` adds an identifier *and* its subtokens, so a request naming
    ``DEFAULT_TIMEOUT`` produces ``default_timeout``, ``default`` and ``timeout``.  Every
    other file here contains only a subtoken, and there are more of them than the cap, so
    returning ``settings.py`` is a statement about ranking rather than about the tree
    being small.

    ``uv.lock`` carries the *full* identifier, which makes it the strongest match by
    content while being the weakest candidate for a code request.  It exists to pin down
    which of those two facts wins.
    """
    (tmp_path / "settings.py").write_text("DEFAULT_TIMEOUT = 30\n", encoding="utf-8")
    for name in ("a.py", "b.py", "c.py", "d.py", "e.py"):
        (tmp_path / name).write_text(f"timeout = None  # {name}\n", encoding="utf-8")
    (tmp_path / "f.py").write_text("default = 1\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text('name = "DEFAULT_TIMEOUT"\n', encoding="utf-8")
    return tmp_path


@pytest.mark.asyncio
async def test_content_search_ranks_the_full_identifier_over_its_subtokens(
    ranking_tree: Path,
):
    """The defect this pins down, MEASURED on starlette before it was fixed.

    ``ripgrep_search`` computed a relevance score per match and then returned
    ``order[:max_matches]``, where ``order`` is *first-seen filesystem walk order* -- the
    score only ever chose which line anchored a file, never which files were returned.
    On starlette the gold file for a request naming a constant sat at median rank 56, and
    with the production cap of 15 matches it was found 0 times out of 40.
    """
    matches = await ripgrep_search(
        extract_keywords("把 DEFAULT_TIMEOUT 修改为 60"), ranking_tree, max_matches=3
    )
    paths = [m.path for m in matches]
    assert paths, "the identifier is in the tree"
    assert paths[0] == "settings.py", paths
    assert "uv.lock" not in paths, paths


@pytest.mark.asyncio
async def test_ast_search_finds_symbol(sample_tree: Path):
    index = RepoIndex.build(sample_tree)
    matches = await ast_search(["Navbar", "height"], index)
    assert len(matches) >= 1
    assert any("Navbar.py" in m.path for m in matches)


@pytest.mark.asyncio
async def test_parallel_explorer_stream(sample_tree: Path):
    index = RepoIndex.build(sample_tree)
    explorer = ParallelExplorer(sample_tree, repo_index=index, effort=LowEffort)

    intermediate_updates = []
    async for cands in explorer.stream_explore("把导航栏的高度缩短一些"):
        intermediate_updates.append(cands)

    assert len(intermediate_updates) >= 1
    final_cands = intermediate_updates[-1]
    assert len(final_cands) >= 1
    assert any("Navbar.py" in c.path for c in final_cands)
    # The top candidate should have contributions from multiple tools
    top = final_cands[0]
    assert "Navbar.py" in top.path
    assert len(top.contributing_tools) >= 1


@pytest.mark.asyncio
async def test_parallel_explorer_early_lock(sample_tree: Path):
    index = RepoIndex.build(sample_tree)
    explorer = ParallelExplorer(sample_tree, repo_index=index, effort=LowEffort)
    results = await explorer.explore("把导航栏的高度缩短一些", early_lock_threshold=0.50)
    assert len(results) >= 1
    assert any("Navbar.py" in c.path for c in results)
