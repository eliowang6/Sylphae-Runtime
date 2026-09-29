"""Comprehensive tests for Service Runner, runtime grounding, and early crash diagnosis."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from sylphae.coprocessor.executor import execute_act_plan
from sylphae.coprocessor.grounding import ground_natural_language_sync
from sylphae.coprocessor.protocol import ActPlan, PlanStep, parse_act_plan
from sylphae.coprocessor.service_runner import (
    discover_candidate_interpreters,
    discover_service_targets,
    is_port_listening,
    launch_and_verify_service,
    probe_interpreter_compatibility,
    select_best_runtime,
)


def test_discover_service_targets_fastapi(tmp_path: Path):
    server_dir = tmp_path / "server"
    server_dir.mkdir()
    main_py = server_dir / "main.py"
    main_py.write_text(
        "from fastapi import FastAPI\nimport uvicorn\napp = FastAPI()\nif __name__ == '__main__':\n    uvicorn.run(app, host='127.0.0.1', port=8000)\n",
        encoding="utf-8",
    )
    req_txt = server_dir / "requirements.txt"
    req_txt.write_text("fastapi>=0.115.0\nuvicorn>=0.34.0\n", encoding="utf-8")

    targets = discover_service_targets(tmp_path)
    assert len(targets) == 1
    t = targets[0]
    assert t.name == "backend"
    assert t.framework == "fastapi"
    assert t.default_port == 8000
    assert "fastapi" in t.required_modules
    assert "uvicorn" in t.required_modules
    assert t.entrypoint == main_py
    assert t.working_dir == server_dir


def test_discover_service_targets_flask(tmp_path: Path):
    app_py = tmp_path / "app.py"
    app_py.write_text(
        "from flask import Flask\napp = Flask(__name__)\nif __name__ == '__main__':\n    app.run(port=5050)\n",
        encoding="utf-8",
    )
    targets = discover_service_targets(tmp_path)
    assert len(targets) == 1
    t = targets[0]
    assert t.framework == "flask"
    assert t.default_port == 5050
    assert "flask" in t.required_modules


def test_discover_candidate_interpreters(tmp_path: Path):
    # Create mock venv
    mock_venv = tmp_path / ".venv" / "Scripts"
    mock_venv.mkdir(parents=True)
    mock_py = mock_venv / ("python.exe" if os.name == "nt" else "python")
    mock_py.write_text("", encoding="utf-8")

    cands = discover_candidate_interpreters(tmp_path)
    cand_strs = [str(c).lower() for c in cands]
    assert str(sys.executable).lower() in cand_strs
    if os.name == "nt":
        assert str(mock_py).lower() in cand_strs


@pytest.mark.asyncio
async def test_probe_interpreter_compatibility():
    current_py = Path(sys.executable)
    # Standard library modules should pass
    ok, lat, err = await probe_interpreter_compatibility(current_py, ["os", "sys", "json"])
    assert ok is True
    assert lat > 0.0
    assert err == ""

    # Non-existent module should fail cleanly with diagnostic
    ok_fail, lat_fail, err_fail = await probe_interpreter_compatibility(
        current_py, ["non_existent_module_xyz_998877"]
    )
    assert ok_fail is False
    assert "No module named" in err_fail


@pytest.mark.asyncio
async def test_select_best_runtime():
    current_py = Path(sys.executable)
    best_py, diag = await select_best_runtime([current_py], ["os", "pathlib"])
    assert best_py == current_py
    assert len(diag) == 1
    assert diag[0]["compatible"] is True


def test_port_listening_check():
    # Pick an ephemeral port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.listen(1)
        assert is_port_listening(port) is True

    # After closing, should not be listening
    assert is_port_listening(port) is False


def test_grounding_service_commands():
    # 1. Exact user directive
    steps = ground_natural_language_sync("启动后端测试服务器")
    assert len(steps) == 1
    assert steps[0].action == "start_service"
    assert steps[0].target_hint == "backend"

    # 2. English directive
    steps_en = ground_natural_language_sync("start backend dev server")
    assert len(steps_en) == 1
    assert steps_en[0].action == "start_service"

    # 3. DSL symbol +start backend
    plan = parse_act_plan("+start backend")
    assert len(plan.steps) == 1
    assert plan.steps[0].action == "start_service"

    # 4. Bracket symbol [START]
    plan_bracket = parse_act_plan("[START] backend")
    assert len(plan_bracket.steps) == 1
    assert plan_bracket.steps[0].action == "start_service"


@pytest.mark.asyncio
async def test_early_crash_detection(tmp_path: Path):
    """Verify that a script terminating prematurely is diagnosed immediately with its error logs."""
    server_dir = tmp_path / "server"
    server_dir.mkdir()
    main_py = server_dir / "main.py"
    # Script that writes a fatal traceback to stderr and exits with 1
    main_py.write_text(
        "import sys\nsys.stderr.write('CRITICAL_DATABASE_INIT_FAILURE: Connection refused at localhost:5432\\n')\nsys.exit(1)\n",
        encoding="utf-8",
    )

    rep = await launch_and_verify_service(tmp_path, target_hint="backend", wait_timeout_s=2.0)
    assert rep.status == "failed"
    assert rep.error_summary is not None
    assert "CRITICAL_DATABASE_INIT_FAILURE" in rep.error_summary
    assert rep.diagnostics.get("exit_code") == 1


@pytest.mark.asyncio
async def test_launch_and_verify_mock_http_service(tmp_path: Path):
    """Verify end-to-end service launch, port polling, and already_running detection."""
    # Find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]

    server_dir = tmp_path / "server"
    server_dir.mkdir()
    main_py = server_dir / "main.py"

    # Minimal HTTP server using stdlib (works with any python)
    script_content = f"""
import http.server
import socketserver
import sys

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'EIDOLON_TEST_OK')

with socketserver.TCPServer(('127.0.0.1', {free_port}), Handler) as httpd:
    httpd.serve_forever()
"""
    main_py.write_text(script_content, encoding="utf-8")

    # 1. Launch service
    rep1 = await launch_and_verify_service(tmp_path, target_hint="backend", wait_timeout_s=4.0)
    try:
        assert rep1.status == "running"
        assert rep1.port == free_port
        assert rep1.pid is not None
        assert rep1.url == f"http://127.0.0.1:{free_port}"

        # 2. Pre-flight check on second call: should report already_running
        rep2 = await launch_and_verify_service(tmp_path, target_hint="backend", wait_timeout_s=1.0)
        assert rep2.status == "already_running"
        assert rep2.port == free_port

        # 3. Test ActPlan execution integration
        plan = ActPlan(
            plan_id="test_plan",
            macro_intent="测试服务器",
            steps=[PlanStep(step_id="step_1", action="start_service", description="后端", target_hint="backend")],
        )
        exec_rep = await execute_act_plan(tmp_path, plan)
        assert exec_rep.all_passed is True
        assert len(exec_rep.service_reports) == 1
        assert exec_rep.service_reports[0].status == "already_running"

    finally:
        # Cleanup spawned test process
        if rep1.pid:
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(rep1.pid)], capture_output=True)
                else:
                    os.kill(rep1.pid, signal.SIGKILL)
            except Exception:
                pass


@pytest.mark.asyncio
async def test_a_silent_listener_is_not_reported_as_running(tmp_path: Path):
    """An open socket is not a working service.

    This was a measured false positive in the C3 gate: a process that bound its port and
    then looped on ``sleep`` came back ``running``, because the post-spawn loop declared
    readiness on ``is_port_listening`` alone.  ``probe_service_health`` existed and was
    correct but was only consulted on the pre-existing-service path; the freshly spawned
    one -- the case where a half-initialised server is most likely -- got the weaker
    check.  It must now come back ``failed`` with the condition named and the pid given,
    because "running" is a promise the caller acts on.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]

    server_dir = tmp_path / "server"
    server_dir.mkdir()
    (server_dir / "main.py").write_text(
        "import socket, time\n"
        "s = socket.socket()\n"
        "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        f"s.bind(('127.0.0.1', {free_port}))\n"
        "s.listen(8)\n"
        "while True:\n"
        "    time.sleep(3600)\n",
        encoding="utf-8",
    )

    rep = await launch_and_verify_service(tmp_path, wait_timeout_s=2.0)
    try:
        assert rep.status == "failed", (
            f"a silent listener must be diagnosed, got {rep.status!r}"
        )
        assert rep.pid is not None
        assert str(free_port) in (rep.error_summary or "")
        assert "never answered an HTTP probe" in (rep.error_summary or "")
        assert rep.diagnostics.get("port_open") is True
        assert rep.diagnostics.get("http_verified") is False
    finally:
        if rep.pid:
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(rep.pid)],
                                   capture_output=True)
                else:
                    os.kill(rep.pid, signal.SIGKILL)
            except Exception:
                pass


@pytest.mark.asyncio
async def test_a_healthy_service_is_still_reported_as_running(tmp_path: Path):
    """The health check must not turn a working service into a false negative.

    Regression guard for the fix above: the probe runs after the port opens, so a server
    that answers on the first request still has to reach ``running``, and quickly.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]

    server_dir = tmp_path / "server"
    server_dir.mkdir()
    (server_dir / "main.py").write_text(
        "import http.server, socketserver\n"
        "class H(http.server.SimpleHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200)\n"
        "        self.end_headers()\n"
        "        self.wfile.write(b'OK')\n"
        f"with socketserver.TCPServer(('127.0.0.1', {free_port}), H) as httpd:\n"
        "    httpd.serve_forever()\n",
        encoding="utf-8",
    )

    rep = await launch_and_verify_service(tmp_path, wait_timeout_s=6.0)
    try:
        assert rep.status == "running"
        assert rep.diagnostics.get("http_verified") is True
    finally:
        if rep.pid:
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(rep.pid)],
                                   capture_output=True)
                else:
                    os.kill(rep.pid, signal.SIGKILL)
            except Exception:
                pass
