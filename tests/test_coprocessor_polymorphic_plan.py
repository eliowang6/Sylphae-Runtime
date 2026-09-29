"""Comprehensive tests for Dual-Mode Action Ingestion & Polymorphic Plan Execution.

Validates:
1. Semi-atomic directives emitted by Frontier LLMs:
   - Flat JSON dictionary: {"schema.sql": "+create ...", "server.py": "+route ..."}
   - Markdown task checklist: "- [ ] schema.sql: [CREATE] ...\n- [ ] server.py: [ROUTE] ..."
   - String array / micro-DSL directives.
2. Fault-tolerant natural language expressions from human developers:
   - Single-clause and multi-clause intent decomposition without regex hardcoding.
   - Physical codebase probe grounding (distinguishing new tables from field additions on existing tables).
3. Polyglot fine-grained operations:
   - Incremental add_field (ALTER TABLE in SQL, struct fields in Go, models in Prisma).
   - Code snippet deletion (delete_block).
   - Route synthesis with dynamic methods and paths.
4. CLI inline and stdin execution.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import pytest

from sylphae.coprocessor.executor import execute_act_plan
from sylphae.coprocessor.grounding import (
    ai_ground_natural_language,
    split_clauses_semantic,
)
from sylphae.coprocessor.protocol import (
    ActPlan,
    parse_act_plan,
)


@pytest.fixture
def mock_polyglot_repo(tmp_path: Path) -> Path:
    """Create a realistic polyglot repo with SQL, Python server, and Vue UI component."""
    root = tmp_path / "polyglot_service"
    root.mkdir()

    # 1. Existing database schema
    (root / "schema.sql").write_text(
        "CREATE TABLE items (\n"
        "    id INTEGER PRIMARY KEY,\n"
        "    name TEXT NOT NULL,\n"
        "    price REAL\n"
        ");\n",
        encoding="utf-8",
    )

    # 2. Existing FastAPI server
    (root / "server.py").write_text(
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n\n"
        '@app.get("/items")\n'
        "async def list_items():\n"
        '    return [{"id": 1, "name": "widget"}]\n',
        encoding="utf-8",
    )

    # 3. Existing auth file with legacy function
    (root / "auth.py").write_text(
        '"""Authentication helpers."""\n\n'
        "def legacy_token_check(token: str) -> bool:\n"
        "    return token == 'legacy_secret'\n\n"
        "def verify_user(uid: int) -> bool:\n"
        "    return uid > 0\n",
        encoding="utf-8",
    )

    # 4. Existing UI component
    (root / "Header.vue").write_text(
        "<template>\n"
        '  <header class="nav-bar">\n'
        '    <button class="bg-blue-500 text-sm">Logout</button>\n'
        "  </header>\n"
        "</template>\n",
        encoding="utf-8",
    )

    return root


def test_parse_act_plan_flat_dict():
    """Verify flat dictionary from Frontier LLM parses into canonical ActPlan."""
    flat_input = {
        "schema.sql": "+create users(user_id, password)",
        "server.py": "+route POST /auth/logout",
    }
    plan = parse_act_plan(flat_input, macro_intent="Add user auth")
    assert isinstance(plan, ActPlan)
    assert len(plan.steps) == 2
    assert plan.steps[0].action == "create_table"
    assert plan.steps[0].target_hint == "schema.sql"
    assert "users" in plan.steps[0].description

    assert plan.steps[1].action == "insert_statement"
    assert plan.steps[1].target_hint == "server.py"
    assert "/auth/logout" in plan.steps[1].description


def test_parse_act_plan_markdown_checklist():
    """Verify Markdown task checklist parses into canonical ActPlan."""
    md_text = """
    # Implementation Steps
    - [ ] schema.sql: [CREATE] users(user_id, password)
    - [ ] server.py: [ROUTE] POST /auth/logout
    - [ ] auth.py: [DELETE] legacy_token_check
    """
    plan = parse_act_plan(md_text)
    assert len(plan.steps) == 3
    assert plan.steps[0].action == "create_table"
    assert plan.steps[0].target_hint == "schema.sql"

    assert plan.steps[1].action == "insert_statement"
    assert plan.steps[1].target_hint == "server.py"

    assert plan.steps[2].action == "delete_block"
    assert plan.steps[2].target_hint == "auth.py"
    assert "legacy_token_check" in plan.steps[2].description


def test_parse_act_plan_micro_dsl_lines():
    """Verify string array and micro-DSL lines."""
    lines = [
        "schema.sql: +create sessions(token, user_id)",
        "server.py: +route GET /healthz",
    ]
    plan = parse_act_plan(lines)
    assert len(plan.steps) == 2
    assert plan.steps[0].action == "create_table"
    assert plan.steps[1].action == "insert_statement"


def test_semantic_clause_splitting():
    """Verify semantic decomposition splits compound clauses without breaking inner tokens."""
    prompt = "把退出按钮调深一点，并且在用户表加上失效时间；然后新增一个注销路由"
    clauses = split_clauses_semantic(prompt)
    assert len(clauses) >= 3
    assert any("退出按钮" in c for c in clauses)
    assert any("用户表" in c for c in clauses)
    assert any("注销路由" in c for c in clauses)


@pytest.mark.asyncio
async def test_ai_grounding_natural_language_distinguishes_table_vs_field(mock_polyglot_repo: Path):
    """Verify AI grounding distinguishes adding field to existing table vs creating new table."""
    # 1. Existing table 'items' -> should ground to 'add_field'
    field_prompt = "在items表增加expire_at字段"
    steps_field = await ai_ground_natural_language(field_prompt, root=mock_polyglot_repo)
    assert len(steps_field) == 1
    assert steps_field[0].action == "add_field"
    assert "schema.sql" in steps_field[0].target_hint

    # 2. Non-existing table 'users' -> should ground to 'create_table'
    new_tbl_prompt = "新增一个用户表包含user_id和password"
    steps_tbl = await ai_ground_natural_language(new_tbl_prompt, root=mock_polyglot_repo)
    assert len(steps_tbl) == 1
    assert steps_tbl[0].action == "create_table"


@pytest.mark.asyncio
async def test_execute_polymorphic_flat_dict(mock_polyglot_repo: Path):
    """Verify end-to-end execution of a flat dict plan across 3 files."""
    flat_plan = {
        "schema.sql": "+create users(user_id, password)",
        "server.py": "+route POST /auth/logout",
        "auth.py": "-delete legacy_token_check",
    }

    report = await execute_act_plan(mock_polyglot_repo, flat_plan)
    assert report.all_passed is True
    assert len(report.step_results) == 3

    # Check schema.sql
    schema = (mock_polyglot_repo / "schema.sql").read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS users" in schema

    # Check server.py
    server = (mock_polyglot_repo / "server.py").read_text(encoding="utf-8")
    assert '@app.post("/auth/logout")' in server
    assert "async def logout" in server

    # Check auth.py (legacy_token_check deleted, verify_user preserved)
    auth = (mock_polyglot_repo / "auth.py").read_text(encoding="utf-8")
    assert "legacy_token_check" not in auth
    assert "verify_user" in auth


@pytest.mark.asyncio
async def test_execute_incremental_add_field(mock_polyglot_repo: Path):
    """Verify incremental add_field generates valid SQL ALTER TABLE."""
    plan = {
        "schema.sql": "+field items.expire_at:timestamp",
    }
    report = await execute_act_plan(mock_polyglot_repo, plan)
    assert report.all_passed is True
    schema = (mock_polyglot_repo / "schema.sql").read_text(encoding="utf-8")
    assert "ALTER TABLE items ADD COLUMN expire_at TIMESTAMP;" in schema


@pytest.mark.asyncio
async def test_execute_polyglot_prisma_and_go(tmp_path: Path):
    """Verify add_field and create_table on Prisma and Go codebases."""
    root = tmp_path / "polyglot_models"
    root.mkdir()

    # 1. Prisma schema
    (root / "schema.prisma").write_text(
        "model User {\n"
        "  id           Int      @id\n"
        "  email        String\n"
        "}\n",
        encoding="utf-8",
    )

    # 2. Go struct
    (root / "models.go").write_text(
        "package models\n\n"
        "type User struct {\n"
        '\tId           int    `json:"id"`\n'
        "}\n",
        encoding="utf-8",
    )

    plan = {
        "schema.prisma": "+field user.avatar_url:string",
        "models.go": "+field user.avatar_url:string",
    }

    report = await execute_act_plan(root, plan)
    assert report.all_passed is True

    prisma = (root / "schema.prisma").read_text(encoding="utf-8")
    assert "avatar_url" in prisma

    go_code = (root / "models.go").read_text(encoding="utf-8")
    assert "AvatarUrl" in go_code


def test_cli_polymorphic_execution_inline_and_stdin(mock_polyglot_repo: Path):
    """Verify CLI --execute-plan accepts inline string and stdin piping."""
    # 1. Inline string directive
    inline_cmd = [
        sys.executable, "-m", "sylphae.agent.cli",
        "--execute-plan", "schema.sql: +create audit_logs(id, action)",
        "--repo", str(mock_polyglot_repo),
        "--json",
    ]
    res_inline = subprocess.run(inline_cmd, capture_output=True, text=True, check=True)
    out_inline = json.loads(res_inline.stdout)
    assert out_inline["all_passed"] is True
    assert "audit_logs" in (mock_polyglot_repo / "schema.sql").read_text(encoding="utf-8")

    # 2. Stdin pipe flat dict
    pipe_data = json.dumps({
        "server.py": "+route GET /health",
    })
    pipe_cmd = [
        sys.executable, "-m", "sylphae.agent.cli",
        "--execute-plan", "-",
        "--repo", str(mock_polyglot_repo),
        "--json",
    ]
    res_pipe = subprocess.run(pipe_cmd, input=pipe_data, capture_output=True, text=True, check=True)
    out_pipe = json.loads(res_pipe.stdout)
    assert out_pipe["all_passed"] is True
    assert '@app.get("/health")' in (mock_polyglot_repo / "server.py").read_text(encoding="utf-8")
