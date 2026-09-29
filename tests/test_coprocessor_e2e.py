"""End-to-end tests for the Frontier LLM <-> Fast Act Coprocessor system.

Validates:
1. Fast codebase intelligence collection emitting ActContextReport inside its
   probe budget (what that budget covers is defined in
   :mod:`sylphae.coprocessor.collector`).
2. Natural language TableSpec parsing and multi-dialect schema generation (SQLite, Postgres, SQLAlchemy, Prisma, Go, Rust).
3. Idiomatic code synthesis across Python, TypeScript, and Go.
4. Concurrently executing multi-step ActPlans (e.g. macro '新增一个用户认证功能').
5. Polyglot execution with full syntax verification and error isolation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import pytest

from sylphae.coprocessor.collector import collect_codebase_context
from sylphae.coprocessor.executor import execute_act_plan
from sylphae.coprocessor.generator import (
    parse_table_description,
    render_table_schema,
    synthesize_code_snippet,
)
from sylphae.coprocessor.protocol import (
    ActContextReport,
    ActPlan,
    DiscoveredEndpoint,
    PlanStep,
)


@pytest.fixture
def mock_codebase(tmp_path: Path) -> Path:
    """Create a realistic mock web service codebase."""
    root = tmp_path / "mock_service"
    root.mkdir()

    # 1. Existing schema file
    (root / "schema.sql").write_text(
        "CREATE TABLE items (\n"
        "    id INTEGER PRIMARY KEY,\n"
        "    name TEXT NOT NULL,\n"
        "    price REAL\n"
        ");\n",
        encoding="utf-8",
    )

    # 2. Existing server file
    (root / "server.py").write_text(
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n\n"
        '@app.get("/items")\n'
        "async def list_items():\n"
        '    return [{"id": 1, "name": "widget"}]\n',
        encoding="utf-8",
    )

    # 3. Existing auth utility file
    (root / "auth.py").write_text(
        '"""Authentication utilities."""\n\n'
        "def dummy_check(token: str) -> bool:\n"
        "    return token == 'secret'\n",
        encoding="utf-8",
    )

    return root


@pytest.mark.asyncio
async def test_collector_probe_intelligence(mock_codebase: Path):
    """Test fast context probing discovers schemas, endpoints, and frameworks."""
    report = await collect_codebase_context(
        root=mock_codebase,
        macro_prompt="新增一个用户认证功能",
    )

    # Performance requirement: sub-500ms (typically < 100ms)
    assert report.probe_latency_ms < 500.0, f"Probe took {report.probe_latency_ms:.1f}ms, expected < 500ms"

    # Schema discovery
    assert len(report.discovered_schemas) >= 1
    table_names = [s.name for s in report.discovered_schemas]
    assert "items" in table_names
    items_schema = next(s for s in report.discovered_schemas if s.name == "items")
    assert "id" in items_schema.fields
    assert "name" in items_schema.fields

    # Endpoint discovery
    assert len(report.discovered_endpoints) >= 1
    routes = [ep.route for ep in report.discovered_endpoints]
    assert "/items" in routes

    # Framework & dialect detection
    assert "FastAPI" in report.active_frameworks
    assert any("sql" in d for d in report.detected_dialects)

    # Prompt context rendering
    ctx_text = report.to_prompt_context()
    assert "Codebase Context Probe" in ctx_text
    assert "FastAPI" in ctx_text
    assert "items in schema.sql" in ctx_text


def test_generator_table_spec_parsing_and_dialects():
    """Test parsing NL descriptions into TableSpec and rendering to various dialects."""
    # 1. Chinese SQLite specification
    spec_cn = parse_table_description("新增一个sqlite用户表，拥有user_id，password项")
    assert spec_cn.name == "users" or "user" in spec_cn.name
    assert spec_cn.dialect == "sqlite"
    col_names = [c.name for c in spec_cn.columns]
    assert "user_id" in col_names
    assert "password" in col_names
    assert spec_cn.columns[0].primary_key is True

    # Render SQLite
    sql = render_table_schema(spec_cn, "sqlite")
    assert "CREATE TABLE IF NOT EXISTS" in sql
    assert "user_id" in sql
    assert "PRIMARY KEY" in sql

    # Render PostgreSQL
    sql_pg = render_table_schema(spec_cn, "postgres")
    assert "CREATE TABLE IF NOT EXISTS" in sql_pg
    assert "PRIMARY KEY" in sql_pg

    # Render SQLAlchemy
    sa_code = render_table_schema(spec_cn, "sqlalchemy")
    assert "class Users(Base):" in sa_code
    assert "Column(" in sa_code
    assert "primary_key=True" in sa_code

    # Render Prisma
    prisma_code = render_table_schema(spec_cn, "prisma")
    assert "model Users {" in prisma_code
    assert "@id" in prisma_code

    # Render Go struct
    go_code = render_table_schema(spec_cn, "go")
    assert "type Users struct {" in go_code
    assert "gorm:\"primaryKey\"" in go_code

    # Render Rust struct
    rust_code = render_table_schema(spec_cn, "rust")
    assert "pub struct Users {" in rust_code
    assert "pub user_id: String" in rust_code


def test_generator_code_synthesis():
    """Test language-idiomatic helper synthesis."""
    # Python hash functions
    py_code = synthesize_code_snippet("安全密码哈希与验证函数 hash_password verify_password", target_lang="python")
    assert "def hash_password" in py_code
    assert "def verify_password" in py_code
    assert "hashlib.sha256" in py_code

    # Go hash functions
    go_code = synthesize_code_snippet("hash password and verify", target_lang="go")
    assert "func HashPassword" in go_code
    assert "sha256.Sum256" in go_code

    # TypeScript hash functions
    ts_code = synthesize_code_snippet("hash password function", target_lang="typescript")
    assert "export function hashPassword" in ts_code

    # Python route
    route_py = synthesize_code_snippet("新增用户登录认证路由 /auth/login", target_lang="python")
    assert '@app.post("/auth/login")' in route_py
    assert "async def login" in route_py


@pytest.mark.asyncio
async def test_coprocessor_auth_e2e_macro_plan(mock_codebase: Path):
    """End-to-end execution of user authentication macro plan across 3 files."""
    plan = ActPlan(
        plan_id="plan_auth_001",
        macro_intent="新增一个用户认证功能",
        steps=[
            PlanStep(
                step_id="step_table",
                action="create_table",
                description="新增一个sqlite用户表，拥有user_id，password项",
                target_hint="schema.sql",
            ),
            PlanStep(
                step_id="step_auth_fn",
                action="add_function",
                description="实现安全密码哈希与验证函数 hash_password, verify_password",
                target_hint="auth.py",
            ),
            PlanStep(
                step_id="step_route",
                action="insert_statement",
                description="新增用户登录认证路由 /auth/login",
                target_hint="server.py",
            ),
        ],
    )

    report = await execute_act_plan(mock_codebase, plan)

    # 1. Execution report assertions
    assert report.all_passed is True, f"Plan execution failed: {[r.error for r in report.step_results if r.error]}"
    assert len(report.step_results) == 3

    for step_res in report.step_results:
        assert step_res.applied is True
        assert step_res.syntax_ok is True
        assert step_res.error is None

    # 2. Verify schema.sql updated
    schema_content = (mock_codebase / "schema.sql").read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS users" in schema_content
    assert "user_id" in schema_content
    assert "password" in schema_content

    # 3. Verify auth.py updated and syntactically valid
    auth_content = (mock_codebase / "auth.py").read_text(encoding="utf-8")
    assert "def hash_password" in auth_content
    assert "def verify_password" in auth_content

    # 4. Verify server.py updated and syntactically valid
    server_content = (mock_codebase / "server.py").read_text(encoding="utf-8")
    assert '@app.post("/auth/login")' in server_content
    assert "async def login" in server_content


@pytest.mark.asyncio
async def test_coprocessor_polyglot_plan_execution(tmp_path: Path):
    """Verify coprocessor executing a polyglot plan on a Go repository."""
    go_root = tmp_path / "go_service"
    go_root.mkdir()
    (go_root / "main.go").write_text(
        "package main\n\n"
        "import \"fmt\"\n\n"
        "func main() {\n"
        "\tfmt.Println(\"Hello\")\n"
        "}\n",
        encoding="utf-8",
    )

    plan = ActPlan(
        plan_id="plan_go_002",
        macro_intent="Add user session model and handler",
        steps=[
            PlanStep(
                step_id="step_model",
                action="create_table",
                description="add user table with user_id, session_token",
                target_hint="models.go",
            ),
            PlanStep(
                step_id="step_handler",
                action="add_function",
                description="add login route handler",
                target_hint="main.go",
            ),
        ],
    )

    report = await execute_act_plan(go_root, plan)
    assert report.all_passed is True
    assert (go_root / "models.go").exists()
    models_code = (go_root / "models.go").read_text(encoding="utf-8")
    assert "type Users struct" in models_code

    main_code = (go_root / "main.go").read_text(encoding="utf-8")
    assert "LoginHandler" in main_code


@pytest.mark.asyncio
async def test_coprocessor_error_isolation(mock_codebase: Path):
    """Verify syntax errors in one step are isolated without crashing the executor."""
    plan = ActPlan(
        plan_id="plan_error_isolation",
        macro_intent="Faulty plan step",
        steps=[
            PlanStep(
                step_id="step_valid",
                action="add_function",
                description="def valid_fn(): return 42\n",
                code_snippet="def valid_fn() -> int:\n    return 42\n",
                target_hint="auth.py",
            ),
            PlanStep(
                step_id="step_broken",
                action="add_function",
                description="broken function",
                code_snippet="def broken_syntax( { : invalid python syntax\n",
                target_hint="auth.py",
            ),
        ],
    )

    report = await execute_act_plan(mock_codebase, plan)
    assert report.all_passed is False
    res_valid = next(r for r in report.step_results if r.step_id == "step_valid")
    res_broken = next(r for r in report.step_results if r.step_id == "step_broken")

    assert res_valid.applied is True
    assert res_broken.syntax_ok is False
    assert res_broken.error is not None
    assert "Syntax" in res_broken.error


def test_coprocessor_protocol_serialization_roundtrip():
    """Verify all protocol models serialize to valid JSON-compatible dicts."""
    import json
    from sylphae.coprocessor.protocol import DiscoveredEndpoint, DiscoveredSchema

    rep = ActContextReport(
        repo_root="/test",
        macro_prompt="Test prompt",
        discovered_schemas=[DiscoveredSchema(name="user", path="schema.sql", dialect="sqlite", fields=["id"])],
        discovered_endpoints=[DiscoveredEndpoint(route="/test", method="GET", handler_name="test_h", source_file="main.py")],
        detected_dialects=["sqlite"],
        active_frameworks=["FastAPI"],
        candidate_target_files=["main.py"],
        probe_latency_ms=45.2,
    )

    d = rep.to_dict()
    assert isinstance(d, dict)
    encoded = json.dumps(d)
    assert "/test" in encoded

    plan = ActPlan(
        plan_id="test_p",
        macro_intent="Intent",
        steps=[PlanStep(step_id="s1", action="create_table", description="desc")],
    )
    plan_d = plan.to_dict()
    assert plan_d["plan_id"] == "test_p"
    assert json.dumps(plan_d)


def test_coprocessor_cli_integration(tmp_path: Path):
    """Verify CLI commands --probe-context and --execute-plan via subprocess."""
    import json
    import subprocess
    import sys

    # 1. Setup small repo
    repo = tmp_path / "cli_repo"
    repo.mkdir()
    (repo / "schema.sql").write_text("CREATE TABLE items (id INT);\n", encoding="utf-8")

    # 2. Test --probe-context --json
    res_probe = subprocess.run(
        [sys.executable, "-m", "sylphae.agent.cli", "--probe-context", "查表", "--repo", str(repo), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    probe_data = json.loads(res_probe.stdout)
    assert "discovered_schemas" in probe_data
    assert len(probe_data["discovered_schemas"]) >= 1

    # 3. Test --execute-plan --json
    plan_file = repo / "plan.json"
    plan_file.write_text(
        json.dumps({
            "plan_id": "cli_test_01",
            "macro_intent": "Add table",
            "steps": [
                {
                    "step_id": "s1",
                    "action": "create_table",
                    "description": "新增用户表，拥有user_id，password项",
                    "target_hint": "schema.sql"
                }
            ]
        }),
        encoding="utf-8",
    )

    res_exec = subprocess.run(
        [sys.executable, "-m", "sylphae.agent.cli", "--execute-plan", str(plan_file), "--repo", str(repo), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    exec_data = json.loads(res_exec.stdout)
    assert exec_data["all_passed"] is True
    assert len(exec_data["step_results"]) == 1
    assert "users" in (repo / "schema.sql").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# probe budget
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_budgeted_returns_none_only_when_the_budget_runs_out():
    """``None`` means "cut short"; ``[]`` means "ran and found nothing"."""
    from sylphae.coprocessor.collector import _budgeted

    async def _fast():
        return ["found"]

    async def _empty():
        return []

    async def _slow():
        await asyncio.sleep(30)

    assert await _budgeted(_fast(), 1.0) == ["found"]
    assert await _budgeted(_empty(), 1.0) == []
    assert await _budgeted(_slow(), 0.05) is None


@pytest.mark.asyncio
async def test_probe_reports_the_budget_it_ran_under(mock_codebase: Path):
    report = await collect_codebase_context(
        root=mock_codebase, macro_prompt="新增一个用户认证功能", timeout_seconds=5.0
    )
    assert report.budget_ms == 5000.0
    assert report.budget_exceeded is False


@pytest.mark.asyncio
async def test_a_probe_that_overruns_is_dropped_and_says_so(
    mock_codebase: Path, monkeypatch: pytest.MonkeyPatch
):
    """A budget that fires must say so, instead of looking like an empty codebase.

    ``collect_codebase_context`` took a ``timeout_seconds`` and never used it, so
    a probe cut short and a codebase with nothing to find produced the same
    report.  Only the facts that do not depend on the clock are asserted here;
    which neighbouring phase survives a cut is
    ``test_the_budget_cuts_one_phase_without_cutting_its_neighbours``.
    """
    from sylphae.coprocessor import collector as collector_mod

    async def _never_finishes(*_args, **_kwargs):
        await asyncio.sleep(30)

    monkeypatch.setattr(collector_mod, "_probe_schemas", _never_finishes)

    report = await collect_codebase_context(
        root=mock_codebase, macro_prompt="新增一个用户认证功能", timeout_seconds=0.5
    )

    assert report.budget_exceeded is True
    assert report.discovered_schemas == []
    assert report.budget_ms == 500.0
    # And the drop is visible to whoever reads the context block.
    assert "INCOMPLETE" in report.to_prompt_context()


@pytest.mark.asyncio
async def test_the_budget_cuts_one_phase_without_cutting_its_neighbours(
    mock_codebase: Path, monkeypatch: pytest.MonkeyPatch
):
    """One phase running out of time is not a reason for another to lose its work.

    The neighbours here are fakes that do no I/O, so this stays a statement about
    the budget rather than about this machine's speed.  It used to run the real
    neighbours and assert they survived a 0.5s budget: that held when idle and
    failed under load, because ``asyncio.to_thread`` pays thread creation on
    first use and nothing bounds that cost by the probe budget.
    """
    from sylphae.coprocessor import collector as collector_mod

    async def _never_finishes(*_args, **_kwargs):
        await asyncio.sleep(30)

    async def _endpoints(*_args, **_kwargs):
        return (
            [
                DiscoveredEndpoint(
                    route="/items",
                    method="GET",
                    handler_name="list_items",
                    source_file="server.py",
                )
            ],
            ["FastAPI"],
        )

    async def _candidates(*_args, **_kwargs):
        return ["server.py"]

    monkeypatch.setattr(collector_mod, "_probe_schemas", _never_finishes)
    monkeypatch.setattr(collector_mod, "_probe_endpoints", _endpoints)
    monkeypatch.setattr(collector_mod, "_probe_target_candidates", _candidates)

    report = await collect_codebase_context(
        root=mock_codebase, macro_prompt="新增一个用户认证功能", timeout_seconds=1.0
    )

    assert report.budget_exceeded is True
    assert report.discovered_schemas == []
    # The phases that did finish keep their results.
    assert [ep.route for ep in report.discovered_endpoints] == ["/items"]
    assert "FastAPI" in report.active_frameworks
    assert report.candidate_target_files == ["server.py"]


@pytest.mark.asyncio
async def test_an_exhausted_budget_skips_the_snippet_phase(
    mock_codebase: Path, monkeypatch: pytest.MonkeyPatch
):
    from sylphae.coprocessor import collector as collector_mod

    called = False

    async def _spy(*args, **kwargs):
        nonlocal called
        called = True
        return []

    async def _slow_candidates(*_args, **_kwargs):
        await asyncio.sleep(30)

    monkeypatch.setattr(collector_mod, "_probe_relevant_snippets", _spy)
    monkeypatch.setattr(collector_mod, "_probe_target_candidates", _slow_candidates)

    report = await collect_codebase_context(
        root=mock_codebase, macro_prompt="新增一个用户认证功能", timeout_seconds=0.2
    )

    assert called is False, "the snippet phase must not start once the budget is gone"
    assert report.relevant_snippets == []
    assert report.budget_exceeded is True


@pytest.mark.asyncio
async def test_the_target_is_met_when_the_content_search_is_not_spawned(
    mock_codebase: Path, monkeypatch: pytest.MonkeyPatch
):
    """The sub-150ms target describes the probe minus the optional ``rg`` spawn.

    Spawning ``rg`` costs ~300ms on Windows -- twice the target on its own -- so
    the path that does not spawn it is the one the target is about.
    ``sylphae.coprocessor.collector`` documents the gap; this pins down the number
    for the path that meets it.
    """
    from sylphae.agent import tools as tools_mod
    from sylphae.coprocessor.collector import TARGET_PROBE_MS

    monkeypatch.setattr(tools_mod, "get_ripgrep_binary", lambda: None)

    report = await collect_codebase_context(
        root=mock_codebase, macro_prompt="新增一个用户认证功能", timeout_seconds=5.0
    )
    assert report.probe_latency_ms < TARGET_PROBE_MS, (
        f"probe took {report.probe_latency_ms:.1f}ms, target is {TARGET_PROBE_MS:.0f}ms"
    )
    assert report.candidate_target_files, "the lexical pass must still produce candidates"


@pytest.mark.asyncio
async def test_a_tight_budget_keeps_the_lexical_candidates(mock_codebase: Path):
    """The enrichment may be dropped; the answer it was enriching may not be."""
    report = await collect_codebase_context(
        root=mock_codebase, macro_prompt="新增一个用户认证功能", timeout_seconds=0.15
    )
    assert report.candidate_target_files, "lexical candidates must survive the drop"
    assert "auth.py" in report.candidate_target_files
    # A hang detector, not a target: killing a spawned ``rg`` costs wall time the
    # budget cannot reclaim (see the module docstring on non-cancellable work).
    assert report.probe_latency_ms < 2000.0



# --------------------------------------------------------------------------------------
# Edit fidelity: a small edit must change the region it names and nothing else
# --------------------------------------------------------------------------------------
#
# These assert on raw bytes, not on a re-read string: ``Path.read_text`` uses universal
# newlines, so re-reading would hide the very change under test.  MEASURED 2026-09-23
# before the fix, through ``execute_act_plan``: an LF file came back with every line
# rewritten to CRLF, a latin-1 0xe9 came back as ef bf bd, and the step reported
# ``applied=True syntax_ok=True error=None`` in both cases.


def _lf_file(root: Path, name: str, body: str) -> Path:
    """Write ``body`` with LF endings regardless of platform, so the fixture is fixed."""
    path = root / name
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
    return path


def _one_step(name: str, **step_kwargs) -> ActPlan:
    return ActPlan(
        plan_id="fidelity",
        macro_intent="a small edit",
        steps=[PlanStep(step_id="s1", target_hint=name, **step_kwargs)],
    )


@pytest.mark.asyncio
async def test_executor_does_not_rewrite_an_lf_file_as_crlf(tmp_path: Path):
    """A one-line edit on a Unix file must not touch the other lines."""
    body = "import os\n\n\ndef main():\n    return 0\n"
    target = _lf_file(tmp_path, "app.py", body)
    before = target.read_bytes()

    report = await execute_act_plan(
        tmp_path,
        _one_step(
            "app.py",
            action="add_function",
            description="def helper(): return 1",
            code_snippet="def helper():\n    return 1\n",
        ),
    )

    after = target.read_bytes()
    assert report.step_results[0].applied, report.step_results[0].error
    assert after.startswith(before), "the untouched region must be byte-identical"
    assert b"\r\n" not in after, "an LF file must not be rewritten with CRLF"


@pytest.mark.asyncio
async def test_executor_does_not_mix_endings_in_a_crlf_file(tmp_path: Path):
    """The generated block is LF; spliced in unchanged it would leave two conventions."""
    target = tmp_path / "app.py"
    target.write_bytes(b"import os\r\n\r\n\r\ndef main():\r\n    return 0\r\n")

    report = await execute_act_plan(
        tmp_path,
        _one_step(
            "app.py",
            action="add_function",
            description="def helper(): return 1",
            code_snippet="def helper():\n    return 1\n",
        ),
    )

    after = target.read_bytes()
    assert report.step_results[0].applied, report.step_results[0].error
    assert after.count(b"\r\n") == after.count(b"\n"), (
        "every line break in a CRLF file must be CRLF, including the appended ones"
    )


@pytest.mark.asyncio
async def test_executor_keeps_a_byte_that_is_not_valid_utf8(tmp_path: Path):
    """``errors="replace"`` used to write U+FFFD over the original byte, irreversibly."""
    target = tmp_path / "notes.py"
    target.write_bytes(b"# caf\xe9 latte\n\n\ndef main():\n    return 0\n")

    report = await execute_act_plan(
        tmp_path,
        _one_step(
            "notes.py",
            action="add_function",
            description="def helper(): return 1",
            code_snippet="def helper():\n    return 1\n",
        ),
    )

    after = target.read_bytes()
    assert report.step_results[0].applied, report.step_results[0].error
    assert b"\xe9" in after, "the original byte must survive"
    assert b"\xef\xbf\xbd" not in after, "U+FFFD is not a replacement for the byte"
    # The compile stage decodes with surrogateescape, so it must compile the *bytes*
    # rather than the decoded string -- a lone surrogate is not compilable.
    assert report.step_results[0].syntax_ok, report.step_results[0].error


@pytest.mark.asyncio
async def test_executor_replace_matches_a_crlf_file_with_an_lf_snippet(tmp_path: Path):
    """The snippet comes from a plan, which is written with LF; the file is not."""
    target = tmp_path / "app.py"
    target.write_bytes(b"MAX = 10\r\nMIN = 1\r\n")

    report = await execute_act_plan(
        tmp_path,
        _one_step(
            "app.py",
            action="replace_literal",
            description="raise the ceiling",
            code_snippet="MAX = 10 -> MAX = 40",
        ),
    )

    after = target.read_bytes()
    assert report.step_results[0].applied, report.step_results[0].error
    assert after == b"MAX = 40\r\nMIN = 1\r\n", (
        "the replacement must land and the untouched line must keep its CRLF"
    )
