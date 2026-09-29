"""Detection of runnable service entrypoints from repository contents.

This answers exactly one question: **can this repository be started at all?** It
never answers "does the user want it started?" — those two were conflated in the
routing layer, and the conflation is what let a UI bug report about a floating
period reach the process launcher.

The result is tri-state. ``None`` means "not probed" (no repository was supplied)
and is distinct from ``False`` meaning "probed, and there is no entrypoint". A
caller that treats ``None`` as ``False`` will veto service launches for every
text-only invocation, which is why the distinction is carried all the way to the
prior update in :mod:`sylphae.model.semantic_grounder`.

Evidence is layered, cheapest first: known config filenames, then filenames that
name a server outright, then a bounded content check for the names that are just as
often a script or a library root.
"""

from __future__ import annotations

import json
from pathlib import Path

#: Config files that identify a runnable project on their own. Compared against the
#: basename, so both ``vite.config.ts`` and ``client/vite.config.ts`` match.
STRONG_MARKERS: frozenset[str] = frozenset(
    {
        # Container orchestration / process managers
        "docker-compose.yml",
        "docker-compose.yaml",
        "compose.yml",
        "compose.yaml",
        "procfile",
        # Backend framework entrypoints
        "manage.py",  # Django
        "wsgi.py",
        "asgi.py",
        # Frontend / full-stack build tooling
        "vite.config.ts",
        "vite.config.js",
        "vite.config.mjs",
        "next.config.js",
        "next.config.mjs",
        "next.config.ts",
        "nuxt.config.ts",
        "nuxt.config.js",
        "angular.json",
        "svelte.config.js",
        "svelte.config.ts",
        "astro.config.mjs",
        "remix.config.js",
    }
)

#: ``package.json`` script names that mean "this project can be launched".
LAUNCH_SCRIPTS: tuple[str, ...] = ("start", "dev", "serve", "develop")

#: Basenames that name a server outright, so the filename alone is the evidence.
#: Deliberately exact basenames rather than a substring test: the previous
#: implementation matched ``"server" in path``, which is true for ``serverConfig.ts``,
#: ``ServerStatus.tsx`` and every vendored ``src/server/`` directory, making the
#: ``start_service`` evidence signal effectively always-on.
DEFINITE_ENTRY_BASENAMES: frozenset[str] = frozenset(
    {
        "server.py",
        "server.js",
        "server.ts",
        "server.mjs",
        "server.rb",
        "wsgi.py",
        "asgi.py",
        "manage.py",
        "main.go",
        "main.rs",
    }
)

#: Basenames that *may* be a service entrypoint but just as often are a plain script
#: or a library root, so the body has to show a server binding too.
AMBIGUOUS_ENTRY_BASENAMES: frozenset[str] = frozenset(
    {
        "main.py",
        "app.py",
        "application.py",
        "run.py",
        "api.py",
        "index.js",
        "index.ts",
        "app.js",
        "app.ts",
    }
)

#: Substrings proving a module actually binds a server, per ecosystem.
SERVER_BODY_HINTS: tuple[str, ...] = (
    "fastapi(",
    "flask(",
    "uvicorn.run",
    "app.run(",
    "sanic(",
    "aiohttp.web",
    "http.listenandserve",  # Go
    "axum::serve",  # Rust
    "actix_web",
    "warp::serve",
    "express(",
    "fastify(",
    "createserver(",
    "app.listen(",
)

#: Build-tooling configs that only ever drive a front-end dev server.
FRONTEND_MARKERS: frozenset[str] = frozenset(
    {
        "vite.config.ts",
        "vite.config.js",
        "vite.config.mjs",
        "next.config.js",
        "next.config.mjs",
        "next.config.ts",
        "nuxt.config.ts",
        "nuxt.config.js",
        "angular.json",
        "svelte.config.js",
        "svelte.config.ts",
        "astro.config.mjs",
        "remix.config.js",
    }
)

#: Configs that only ever drive a back-end service.
BACKEND_MARKERS: frozenset[str] = frozenset(
    {
        "manage.py",  # Django
        "wsgi.py",
        "asgi.py",
    }
)

#: Cap on how many candidate modules get a content check, so probing stays fast on
#: repositories with many same-named files.
MAX_CONTENT_CHECKS: int = 6

#: Bytes read from a candidate module.
MODULE_HEAD_BYTES: int = 16384

#: ``package.json`` is small, but guard against a pathological one.
PACKAGE_JSON_MAX_BYTES: int = 512 * 1024


def _read_text(path: Path, limit: int) -> str:
    """Read at most ``limit`` bytes, tolerating any encoding or I/O failure."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(limit)
    except OSError:
        return ""


def _basename(rel_path: str) -> str:
    """Lower-cased basename of a repository-relative posix path."""
    return rel_path.rsplit("/", 1)[-1].lower()


def _package_json_launchable(root: Path, rel_path: str) -> bool:
    """True when ``package.json`` declares a launchable script."""
    raw = _read_text(root / rel_path, PACKAGE_JSON_MAX_BYTES)
    if not raw:
        return False
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return False
    if not isinstance(data, dict):
        return False
    scripts = data.get("scripts")
    if not isinstance(scripts, dict):
        return False
    return any(name in scripts for name in LAUNCH_SCRIPTS)


def _module_binds_server(root: Path, rel_path: str) -> bool:
    """True when a candidate module's body shows a server binding."""
    body = _read_text(root / rel_path, MODULE_HEAD_BYTES).lower()
    if not body:
        return False
    return any(hint in body for hint in SERVER_BODY_HINTS)


def detect_service_entrypoint(
    root: Path | None,
    proj_files: list[str],
) -> bool | None:
    """Report whether ``root`` can be started.

    Returns ``True`` when a runnable entrypoint was found, ``False`` when the
    repository was probed and none exists, and ``None`` when there was nothing to
    probe (no repository, or an empty file list). The tri-state matters: callers use
    ``False`` to veto service launches, and treating "unknown" as "no" would silently
    disable that action for every text-only caller.
    """
    if root is None or not proj_files:
        return None

    basenames = {_basename(p) for p in proj_files}
    if basenames & STRONG_MARKERS or basenames & DEFINITE_ENTRY_BASENAMES:
        return True

    for rel_path in proj_files:
        if _basename(rel_path) == "package.json" and _package_json_launchable(root, rel_path):
            return True

    checked = 0
    for rel_path in proj_files:
        if _basename(rel_path) not in AMBIGUOUS_ENTRY_BASENAMES:
            continue
        if checked >= MAX_CONTENT_CHECKS:
            break
        checked += 1
        if _module_binds_server(root, rel_path):
            return True

    return False


def detect_service_flavor(
    root: Path | None,
    proj_files: list[str],
) -> str | None:
    """Report which kind of service this repository runs.

    Returns ``"frontend"``, ``"backend"``, or ``None`` when neither can be
    established from the repository. ``None`` lets the caller keep its own default
    rather than guessing from the wording of the request, which is what the removed
    keyword table (``"前端"``/``"client"``/``"vite"``) used to do.
    """
    if root is None or not proj_files:
        return None

    basenames = {_basename(p) for p in proj_files}
    if basenames & FRONTEND_MARKERS:
        return "frontend"
    if basenames & BACKEND_MARKERS:
        return "backend"

    checked = 0
    for rel_path in proj_files:
        if _basename(rel_path) not in AMBIGUOUS_ENTRY_BASENAMES:
            continue
        if checked >= MAX_CONTENT_CHECKS:
            break
        checked += 1
        if _module_binds_server(root, rel_path):
            return "backend"

    return None
