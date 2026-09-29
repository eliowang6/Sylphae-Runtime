"""Security tests for coprocessor execution and path containment."""

import sys
import subprocess
from pathlib import Path
import pytest

from sylphae.coprocessor.executor import execute_act_plan
from sylphae.coprocessor.protocol import ActPlan, PlanStep


@pytest.mark.asyncio
async def test_execute_plan_blocks_relative_escape(tmp_path: Path):
    """Verify relative traversal (../escaped.py) is rejected and not written."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("print('hello')\n", encoding="utf-8")

    plan = ActPlan(
        plan_id="sec_test_rel",
        macro_intent="Malicious escape attempt",
        steps=[
            PlanStep(
                step_id="s_rel",
                action="add_function",
                description="def pwn(): pass",
                target_hint="../escaped_relative.py",
                code_snippet="def pwn(): pass\n",
            ),
        ],
    )

    report = await execute_act_plan(repo, plan)
    assert report.all_passed is False
    assert len(report.step_results) == 1
    res = report.step_results[0]
    assert res.applied is False
    assert (tmp_path / "escaped_relative.py").exists() is False


@pytest.mark.asyncio
async def test_execute_plan_blocks_absolute_escape(tmp_path: Path):
    """Verify absolute path escaping root is rejected and not written."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("print('hello')\n", encoding="utf-8")

    outside_file = tmp_path / "outside" / "escaped_absolute.sql"
    plan = ActPlan(
        plan_id="sec_test_abs",
        macro_intent="Malicious absolute escape",
        steps=[
            PlanStep(
                step_id="s_abs",
                action="create_table",
                description="create audit_logs table",
                target_hint=str(outside_file),
                code_snippet="CREATE TABLE audit_logs (id INT);\n",
            ),
        ],
    )

    report = await execute_act_plan(repo, plan)
    assert report.all_passed is False
    res = report.step_results[0]
    assert res.applied is False
    assert outside_file.exists() is False


@pytest.mark.asyncio
async def test_execute_plan_allows_safe_nested_creation(tmp_path: Path):
    """Verify valid relative path inside root is successfully executed."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("print('hello')\n", encoding="utf-8")

    plan = ActPlan(
        plan_id="sec_test_safe",
        macro_intent="Safe creation inside repo",
        steps=[
            PlanStep(
                step_id="s_safe",
                action="add_function",
                description="def helper(): pass",
                target_hint="pkg/sub/helper.py",
                code_snippet="def helper(): pass\n",
            ),
        ],
    )

    report = await execute_act_plan(repo, plan)
    assert report.all_passed is True
    res = report.step_results[0]
    assert res.applied is True
    assert (repo / "pkg" / "sub" / "helper.py").exists() is True
    assert "def helper(): pass" in (repo / "pkg" / "sub" / "helper.py").read_text(encoding="utf-8")


def test_cli_execute_plan_security_exit_code(tmp_path: Path):
    """Verify CLI --execute-plan exits with non-zero exit code when plan has escape."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("print('hello')\n", encoding="utf-8")

    plan_content = (
        "plan_id: sec_cli\n"
        "macro_intent: escape test\n"
        "steps:\n"
        "  - step_id: s1\n"
        "    action: add_function\n"
        "    description: escape\n"
        "    target_hint: ../escaped_cli.py\n"
    )
    plan_file = repo / "plan.yaml"
    plan_file.write_text(plan_content, encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "sylphae.agent.cli",
        "--execute-plan",
        str(plan_file),
        "--repo",
        str(repo),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    assert (tmp_path / "escaped_cli.py").exists() is False
