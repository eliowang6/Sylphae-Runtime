"""End-to-end tests for the ReflexAgent."""

from pathlib import Path

import pytest

from sylphae.agent.effort import LowEffort
from sylphae.agent.reflex import ReflexAgent


@pytest.fixture
def repo_with_tests(tmp_path: Path) -> Path:
    """Create a mock repository with a target file and a test file."""
    src = tmp_path / "src"
    src.mkdir(parents=True)
    tests = tmp_path / "tests"
    tests.mkdir(parents=True)

    navbar_code = (
        "NAV_HEIGHT = 64\n\n"
        "def get_nav_height():\n"
        "    return NAV_HEIGHT\n"
    )
    (src / "navbar.py").write_text(navbar_code, encoding="utf-8")

    test_code = (
        "from src.navbar import get_nav_height\n\n"
        "def test_nav_height():\n"
        "    assert get_nav_height() > 0\n"
    )
    (tests / "test_navbar.py").write_text(test_code, encoding="utf-8")
    return tmp_path


@pytest.mark.asyncio
async def test_reflex_agent_optimistic_dry_run(repo_with_tests: Path):
    agent = ReflexAgent(root=repo_with_tests, effort=LowEffort)
    res = await agent.run("把导航栏的高度缩短一些", dry_run=True, verify_in_background=False)

    assert "navbar.py" in res.target_path
    assert res.op_kind == "replace_literal"
    assert res.immediate_patch_ok is True
    assert res.new_source is not None
    # 64 should be reduced to 32
    assert "NAV_HEIGHT = 32" in res.new_source

    # Disk was NOT modified in dry-run
    original_on_disk = (repo_with_tests / "src" / "navbar.py").read_text(encoding="utf-8")
    assert "NAV_HEIGHT = 64" in original_on_disk


@pytest.mark.asyncio
async def test_reflex_agent_apply_and_verify(repo_with_tests: Path):
    agent = ReflexAgent(root=repo_with_tests, effort=LowEffort)
    res = await agent.run("把导航栏的高度缩短一些", dry_run=False, verify_in_background=False)

    assert res.immediate_patch_ok is True
    # Disk WAS modified
    modified_on_disk = (repo_with_tests / "src" / "navbar.py").read_text(encoding="utf-8")
    assert "NAV_HEIGHT = 32" in modified_on_disk

    # Verification should pass
    assert res.verification_report is not None
    assert res.verification_report.compiles is True
    assert res.verification_report.tests_passed >= 1
