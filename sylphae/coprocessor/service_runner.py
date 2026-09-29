"""Autonomous Service Runner & Runtime Environment Grounding.

Compresses multi-turn LLM trial-and-error (port checks, log scraping,
ModuleNotFoundError, finding virtualenvs, checking dev.bat, managing stale PIDs)
into a single, high-speed, autonomous action for '启动后端测试服务器' (+start backend).
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sylphae.coprocessor.protocol import ServiceLaunchReport


@dataclass
class ServiceTarget:
    """Discovered candidate server or service definition within repository."""
    name: str  # 'backend', 'server', 'client', 'imagegen'
    entrypoint: Path  # e.g. /srv/api/server/main.py
    working_dir: Path  # e.g. /srv/api/server
    framework: str  # 'fastapi', 'flask', 'django', 'express', 'generic_python'
    default_port: int  # 8000, 5000, 3000, etc.
    required_modules: list[str] = field(default_factory=list)
    launch_command: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    log_file: Path | None = None
    err_file: Path | None = None


def discover_service_targets(root: Path, target_hint: str = "backend") -> list[ServiceTarget]:
    """Inspect repository AST and directory structure to discover service targets.

    ``target_hint`` is a label, not a filter: it is what the caller says it is looking
    for, and it is carried into the report so a failure can name what was being sought.
    Discovery itself is driven only by what is on disk -- Python entrypoints first, then
    Node -- so a wrong or missing hint cannot hide a service that exists.  It reads as
    though it steered the search, which is worth knowing before changing anything here.
    """
    root = Path(root).resolve()
    targets: list[ServiceTarget] = []

    # Priority candidates for backend
    candidate_rel_paths = [
        "server/main.py",
        "backend/main.py",
        "server/app.py",
        "backend/app.py",
        "app/main.py",
        "main.py",
        "app.py",
        "run.py",
        "manage.py",
    ]

    for rel in candidate_rel_paths:
        ep = root / rel
        if ep.is_file():
            target = _inspect_python_entrypoint(root, ep)
            if target:
                targets.append(target)
                break

    # If nothing found, search all .py files in shallow depth (< 3)
    if not targets:
        for py_file in root.glob("*/*.py"):
            if py_file.name in ("main.py", "app.py", "run.py", "server.py"):
                target = _inspect_python_entrypoint(root, py_file)
                if target:
                    targets.append(target)
                    break

    # If still nothing, check Node / Express
    if not targets:
        for js_rel in ("server/index.js", "server.js", "index.js", "app.js"):
            js_ep = root / js_rel
            if js_ep.is_file():
                targets.append(
                    ServiceTarget(
                        name="backend",
                        entrypoint=js_ep,
                        working_dir=js_ep.parent,
                        framework="express",
                        default_port=3000,
                        required_modules=[],
                        launch_command=["node", js_ep.name],
                    )
                )
                break

    return targets


def _inspect_python_entrypoint(root: Path, entrypoint: Path) -> ServiceTarget:
    """Inspect a Python entrypoint to determine framework, required modules, and port."""
    workdir = entrypoint.parent
    text = entrypoint.read_text(encoding="utf-8", errors="replace")

    framework = "generic_python"
    req_modules: list[str] = []
    port = 8000

    # 1. Detect Framework
    if "fastapi" in text.lower() or "uvicorn" in text.lower():
        framework = "fastapi"
        req_modules = ["fastapi", "uvicorn"]
        port = 8000
    elif "flask" in text.lower():
        framework = "flask"
        req_modules = ["flask"]
        port = 5000
    elif "django" in text.lower():
        framework = "django"
        req_modules = ["django"]
        port = 8000

    # 2. Check requirements.txt or pyproject.toml in workdir or root
    req_file = workdir / "requirements.txt"
    if not req_file.exists():
        req_file = root / "requirements.txt"
    if req_file.exists():
        req_text = req_file.read_text(encoding="utf-8", errors="replace").lower()
        if "fastapi" in req_text and "fastapi" not in req_modules:
            req_modules.append("fastapi")
        if "uvicorn" in req_text and "uvicorn" not in req_modules:
            req_modules.append("uvicorn")
        if "flask" in req_text and "flask" not in req_modules:
            req_modules.append("flask")

    # 3. Detect configured port from code or .env
    env_file = workdir / ".env"
    if not env_file.exists():
        env_file = root / ".env"
    if env_file.exists():
        env_text = env_file.read_text(encoding="utf-8", errors="replace")
        m_port = re.search(r"^\s*PORT\s*=\s*(\d+)", env_text, re.MULTILINE | re.IGNORECASE)
        if m_port:
            port = int(m_port.group(1))

    # Also search code for port=..., PORT = ..., or ("127.0.0.1", <port>)
    m_code_port = re.search(r'\bPORT\s*=\s*(\d{2,5})|["\']?port["\']?\s*[:=]\s*(\d{2,5})', text, re.IGNORECASE)
    if m_code_port:
        port = int(m_code_port.group(1) or m_code_port.group(2))
    else:
        m_bind = re.search(r'\(\s*["\'](?:127\.0\.0\.1|0\.0\.0\.0|localhost)["\']\s*,\s*(\d{2,5})\s*\)', text)
        if m_bind:
            port = int(m_bind.group(1))

    # Log files
    log_file = workdir / "dev-server.log"
    err_file = workdir / "dev-server-err.log"

    return ServiceTarget(
        name="backend",
        entrypoint=entrypoint,
        working_dir=workdir,
        framework=framework,
        default_port=port,
        required_modules=req_modules,
        launch_command=["python", entrypoint.name],
        env_vars={
            "PYTHONUNBUFFERED": "1",
            "DEBUG": "true",
        },
        log_file=log_file,
        err_file=err_file,
    )


def discover_candidate_interpreters(root: Path) -> list[Path]:
    """Find all candidate Python interpreters across virtual environments, PATH, and system."""
    root = Path(root).resolve()
    candidates: list[Path] = []

    def _add_cand(p: Path | str | None):
        if not p:
            return
        cand_path = Path(p).resolve()
        if cand_path.is_file() and cand_path not in candidates:
            # Check executable extension on Windows
            if os.name == "nt" and cand_path.suffix.lower() != ".exe":
                return
            candidates.append(cand_path)

    # 1. Active virtualenvs inside repository and subfolders
    venv_patterns = [
        root / ".venv" / "Scripts" / "python.exe",
        root / ".venv" / "bin" / "python",
        root / "server" / ".venv" / "Scripts" / "python.exe",
        root / "server" / ".venv" / "bin" / "python",
        root / "venv" / "Scripts" / "python.exe",
        root / "venv" / "bin" / "python",
        root / "env" / "Scripts" / "python.exe",
        root / "env" / "bin" / "python",
    ]
    for vp in venv_patterns:
        _add_cand(vp)

    # 2. sys.executable (current running python)
    _add_cand(sys.executable)

    # 3. PATH pythons
    which_py = shutil.which("python")
    if which_py:
        _add_cand(which_py)
    which_py3 = shutil.which("python3")
    if which_py3:
        _add_cand(which_py3)

    # 4. Windows py launcher (py -0p)
    if os.name == "nt" and shutil.which("py"):
        try:
            res = subprocess.run(
                ["py", "-0p"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=0.3,
            )
            for line in res.stdout.splitlines():
                # Line format: " -V:3.13 *        /usr/bin/python3.13"
                parts = line.strip().split()
                if len(parts) >= 2:
                    potential_path = parts[-1]
                    _add_cand(potential_path)
        except Exception:
            pass

    # 5. Standard Windows user paths (AppData / Local / Programs / Python)
    if os.name == "nt":
        appdata_local = os.environ.get("LOCALAPPDATA")
        if appdata_local:
            py_progs = Path(appdata_local) / "Programs" / "Python"
            if py_progs.is_dir():
                for sub in py_progs.iterdir():
                    _add_cand(sub / "python.exe")

        # UV python cache
        appdata_roaming = os.environ.get("APPDATA")
        if appdata_roaming:
            uv_py_dir = Path(appdata_roaming) / "uv" / "python"
            if uv_py_dir.is_dir():
                for sub in uv_py_dir.glob("*/python.exe"):
                    _add_cand(sub)

    return candidates


async def probe_interpreter_compatibility(
    candidate: Path,
    required_modules: list[str],
    workdir: Path | None = None,
    timeout_s: float = 5.0,
) -> tuple[bool, float, str]:
    """Test candidate interpreter compatibility for required modules in parallel.

    Returns (compatible, latency_ms, error_message).

    ``timeout_s`` is a *budget for a healthy interpreter*, not a latency expectation: on
    Windows the first ``create_subprocess_exec`` in a process is a cold start measured at
    ~1.1s (warm spawns are ~0.1s), so the 0.6s it used to be turned a perfectly good
    interpreter into "Timeout after 0.6s" -- a false incompatible verdict.  Selection still
    ranks candidates by the *measured* latency, so widening this budget does not make the
    ranking any less selective.
    """
    if not required_modules:
        return True, 0.0, ""

    mod_imports = "; ".join(f"import {m}" for m in required_modules)
    test_code = f"{mod_imports}; print('COMPAT_OK')"

    t0 = time.perf_counter()
    try:
        proc = await asyncio.create_subprocess_exec(
            str(candidate),
            "-c",
            test_code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workdir) if workdir else None,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return False, (time.perf_counter() - t0) * 1000.0, f"Timeout after {timeout_s}s"

        latency_ms = (time.perf_counter() - t0) * 1000.0
        out_str = stdout.decode("utf-8", errors="replace").strip()
        err_str = stderr.decode("utf-8", errors="replace").strip()

        if proc.returncode == 0 and "COMPAT_OK" in out_str:
            return True, latency_ms, ""

        first_err = err_str.splitlines()[-1] if err_str else f"Returncode {proc.returncode}"
        return False, latency_ms, first_err

    except Exception as exc:
        return False, (time.perf_counter() - t0) * 1000.0, str(exc)


async def select_best_runtime(
    candidates: list[Path],
    required_modules: list[str],
    workdir: Path | None = None,
) -> tuple[Path | None, list[dict[str, Any]]]:
    """Probe all candidates concurrently and select the lowest-latency compatible interpreter."""
    if not candidates:
        return None, []

    tasks = [
        probe_interpreter_compatibility(c, required_modules, workdir=workdir)
        for c in candidates
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    diagnostics: list[dict[str, Any]] = []
    best_candidate: Path | None = None
    best_latency = float("inf")

    for cand, (ok, lat, err) in zip(candidates, results):
        diag = {
            "path": str(cand),
            "compatible": ok,
            "latency_ms": round(lat, 1),
            "error": err if not ok else None,
        }
        diagnostics.append(diag)
        if ok and lat < best_latency:
            best_latency = lat
            best_candidate = cand

    return best_candidate, diagnostics


def is_port_listening(port: int, host: str = "127.0.0.1") -> bool:
    """Check if TCP port is listening using socket connection (<10ms)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.15)
        res = s.connect_ex((host, port))
        return res == 0


def find_pid_by_port(port: int) -> int | None:
    """Find owning PID for a listening port on Windows or Unix."""
    if os.name == "nt":
        try:
            res = subprocess.run(
                ["netstat", "-aon"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=0.5,
            )
            port_str = f":{port}"
            for line in res.stdout.splitlines():
                if port_str in line and "LISTENING" in line:
                    parts = line.strip().split()
                    if parts:
                        pid = int(parts[-1])
                        return pid
        except Exception:
            pass
    else:
        try:
            res = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=0.5,
            )
            pids = res.stdout.strip().split()
            if pids:
                return int(pids[0])
        except Exception:
            pass
    return None


def probe_service_health(port: int, timeout: float = 0.5) -> bool:
    """Test HTTP health probe on common endpoints (/docs, /health, /)."""
    urls = [
        f"http://127.0.0.1:{port}/health",
        f"http://127.0.0.1:{port}/docs",
        f"http://127.0.0.1:{port}/",
    ]
    for u in urls:
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Fae-HealthCheck/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status in (200, 404, 301, 302):
                    return True
        except urllib.error.HTTPError as he:
            if he.code in (200, 404, 301, 302):
                return True
        except Exception:
            continue
    return False


async def launch_and_verify_service(
    root: Path,
    target_hint: str = "backend",
    wait_timeout_s: float = 3.5,
) -> ServiceLaunchReport:
    """Autonomously discover, probe, launch, and verify a service in a single shot."""
    t_start = time.perf_counter()
    root = Path(root).resolve()

    # 1. Discover target
    targets = discover_service_targets(root, target_hint=target_hint)
    if not targets:
        return ServiceLaunchReport(
            status="unresolvable",
            service_name=target_hint,
            latency_ms=(time.perf_counter() - t_start) * 1000.0,
            error_summary=f"No service entrypoint (e.g. server/main.py, app.py) found in {root}",
        )

    target = targets[0]
    port = target.default_port

    # 2. Port Pre-flight Check: is it already running?
    if is_port_listening(port):
        existing_pid = find_pid_by_port(port)
        is_healthy = probe_service_health(port)
        if is_healthy:
            return ServiceLaunchReport(
                status="already_running",
                service_name=target.name,
                pid=existing_pid,
                port=port,
                url=f"http://127.0.0.1:{port}",
                working_dir=str(target.working_dir),
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
                diagnostics={"healthy": True, "pre_existing": True},
            )

    # 3. Discover candidate interpreters & parallel compatibility probe
    candidates = discover_candidate_interpreters(root)
    best_py, diag = await select_best_runtime(
        candidates,
        target.required_modules,
        workdir=target.working_dir,
    )

    if not best_py:
        # Honest reporting when no runtime satisfies required dependencies
        return ServiceLaunchReport(
            status="unresolvable",
            service_name=target.name,
            working_dir=str(target.working_dir),
            latency_ms=(time.perf_counter() - t_start) * 1000.0,
            error_summary=(
                f"Missing dependencies: {', '.join(target.required_modules)}. "
                f"None of the {len(candidates)} candidate Python environments satisfied requirements."
            ),
            diagnostics={
                "candidate_environments_probed": diag,
                "suggested_action": f"uv pip install {' '.join(target.required_modules)} or activate project venv",
            },
        )

    # 4. Prepare launch command & environment
    env = os.environ.copy()
    for k, v in target.env_vars.items():
        env[k] = v

    cmd = [str(best_py), target.entrypoint.name]

    log_f = target.log_file or (target.working_dir / "dev-server.log")
    err_f = target.err_file or (target.working_dir / "dev-server-err.log")

    log_out = open(log_f, "w", encoding="utf-8")
    log_err = open(err_f, "w", encoding="utf-8")

    # 5. Spawn background daemon process
    try:
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            proc = subprocess.Popen(
                cmd,
                cwd=str(target.working_dir),
                env=env,
                stdout=log_out,
                stderr=log_err,
                creationflags=creationflags,
            )
        else:
            proc = subprocess.Popen(
                cmd,
                cwd=str(target.working_dir),
                env=env,
                stdout=log_out,
                stderr=log_err,
                start_new_session=True,
            )
    except Exception as spawn_exc:
        log_out.close()
        log_err.close()
        return ServiceLaunchReport(
            status="failed",
            service_name=target.name,
            runtime_path=str(best_py),
            working_dir=str(target.working_dir),
            latency_ms=(time.perf_counter() - t_start) * 1000.0,
            error_summary=f"Failed to spawn server process: {spawn_exc}",
        )

    # 6. Early Supervision & Port Polling Loop
    deadline = time.perf_counter() + wait_timeout_s
    server_ready = False
    # An open socket is not a working service.  MEASURED by the C3 gate: a process that
    # bound its port and then looped on ``sleep`` came back as ``running``, because this
    # loop declared readiness on ``is_port_listening`` alone and never asked the service
    # anything.  ``probe_service_health`` existed and was correct, but was only used on
    # the pre-flight path for a *pre-existing* service -- the freshly spawned one, the
    # case where a half-initialised server is most likely, got the weaker check.
    port_opened = False

    while time.perf_counter() < deadline:
        # Check if process terminated prematurely (early crash)
        ret = proc.poll()
        if ret is not None:
            # Crashed early! Read stderr immediately
            log_out.close()
            log_err.close()
            err_content = err_f.read_text(encoding="utf-8", errors="replace") if err_f.exists() else ""
            out_content = log_f.read_text(encoding="utf-8", errors="replace") if log_f.exists() else ""
            combined_err = (err_content + "\n" + out_content).strip()
            last_lines = "\n".join(combined_err.splitlines()[-6:]) if combined_err else f"Process exited with code {ret}"

            return ServiceLaunchReport(
                status="failed",
                service_name=target.name,
                pid=proc.pid,
                runtime_path=str(best_py),
                working_dir=str(target.working_dir),
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
                log_file=str(err_f),
                error_summary=f"Server process terminated prematurely (exit code {ret}):\n{last_lines}",
                diagnostics={"exit_code": ret, "last_logs": last_lines},
            )

        # Check if port opened
        if is_port_listening(port):
            port_opened = True
            # Declare readiness only when the service actually answers.  Keep polling
            # while it does not: a server that is still initialising is expected to open
            # its socket before it is ready to serve.
            if probe_service_health(port):
                server_ready = True
                break

        await asyncio.sleep(0.12)

    log_out.close()
    log_err.close()
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    if server_ready:
        return ServiceLaunchReport(
            status="running",
            service_name=target.name,
            pid=proc.pid,
            port=port,
            url=f"http://127.0.0.1:{port}",
            runtime_path=str(best_py),
            working_dir=str(target.working_dir),
            latency_ms=elapsed_ms,
            log_file=str(log_f),
            diagnostics={
                "framework": target.framework,
                "entrypoint": str(target.entrypoint),
                "probed_environments": len(candidates),
                "http_verified": True,
            },
        )
    elif port_opened:
        # The port is open and the process is alive, but nothing ever answered an HTTP
        # request.  Reporting that as ``running`` would be a false positive -- the caller
        # would hand a URL to something that cannot serve -- so it is a diagnosed failure
        # instead, with the pid so the caller can reap it.
        return ServiceLaunchReport(
            status="failed",
            service_name=target.name,
            pid=proc.pid,
            port=port,
            url=f"http://127.0.0.1:{port}",
            runtime_path=str(best_py),
            working_dir=str(target.working_dir),
            latency_ms=elapsed_ms,
            log_file=str(err_f),
            error_summary=(
                f"Port {port} opened but the service never answered an HTTP probe "
                f"within {wait_timeout_s}s (pid {proc.pid}); it is listening but not serving"
            ),
            diagnostics={
                "port_open": True,
                "http_verified": False,
                "process_alive": proc.poll() is None,
            },
        )
    else:
        # Port didn't open within deadline, but process is still alive
        return ServiceLaunchReport(
            status="starting",
            service_name=target.name,
            pid=proc.pid,
            port=port,
            url=f"http://127.0.0.1:{port}",
            runtime_path=str(best_py),
            working_dir=str(target.working_dir),
            latency_ms=elapsed_ms,
            log_file=str(log_f),
            error_summary=f"Process {proc.pid} started but port {port} not yet responding within {wait_timeout_s}s",
            diagnostics={"process_alive": True},
        )
