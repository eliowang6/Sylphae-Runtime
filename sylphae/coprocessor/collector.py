"""Parallel codebase intelligence collector.

Pre-collects and synthesizes rich context for Frontier LLMs:
- Discovers existing database models and SQL schemas
- Discovers existing API routes and controllers
- Detects active frameworks, ORMs, and dialects
- Suggests candidate files for target modifications

What the latency budget covers
------------------------------

The ``<150ms`` target (:data:`TARGET_PROBE_MS`) is the **probe's own wall time** --
file discovery, the three parallel probes, and snippet ranking, i.e. exactly the
number reported in :attr:`ActContextReport.probe_latency_ms`.  It deliberately
does *not* cover interpreter start-up or module imports.  Measured on the
development machine, ``python -c pass`` alone takes ~1.2s and importing this
module's dependencies another ~110ms, so an end-to-end ``python -m
sylphae.agent.cli -p ...`` invocation cannot reach 150ms no matter what this file
does.  Quoting the budget as end-to-end latency was the error; the probe keeps to
it.

:attr:`ActContextReport.budget_ms` and
:attr:`ActContextReport.budget_exceeded` report whether the probe actually stayed
within the budget it was given, so a caller never has to guess whether a thin
result means "nothing matched" or "we ran out of time".

One caveat the budget cannot fix: each probe does its scanning in a worker thread
(``asyncio.to_thread``), and a thread cannot be cancelled.  Cutting a phase short
stops the *caller* waiting, not the work -- so ``probe_latency_ms`` may exceed
``budget_ms`` on a cold or contended machine even though every phase was dropped
on time.  The budget bounds latency, not CPU spent.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

from sylphae.agent.explorer import extract_keywords
from sylphae.coprocessor.protocol import (
    ActContextReport,
    DiscoveredEndpoint,
    DiscoveredSchema,
    RelevantSnippet,
)
from sylphae.repo.polyglot import iter_project_files

#: The documented target for the probe's own wall time, in milliseconds.
#:
#: Measured, this is met by everything in the probe except the optional content
#: search: on Windows one ``rg`` invocation costs ~300ms just to spawn (measured
#: 284-318ms across repeats, against a binary bundled inside another product's
#: install directory), which is twice this budget before any searching happens.
#: Because that pass is triggered whenever the path-lexical pass returns fewer
#: than five candidates, the *smallest* repositories are the ones that pay it --
#: a 3-file fixture probes at ~290ms while a 119-file repository probes at ~84ms.
#: The spawn is documented as the known gap rather than papered over with a
#: throughput estimate: the enrichment runs under the caller's budget and is
#: dropped rather than waited for, and
#: :attr:`ActContextReport.budget_exceeded` says when that happened.
TARGET_PROBE_MS: float = 150.0

#: Default enforcement budget.  Deliberately looser than :data:`TARGET_PROBE_MS`:
#: the target is what the probe should achieve, this is the point past which
#: returning partial context beats making the caller wait.  A caller that needs
#: the target honoured passes ``timeout_seconds=0.15`` and gets lexical
#: candidates only -- deterministically, because the content search cannot fit.
DEFAULT_PROBE_BUDGET_S: float = 1.0


async def _budgeted(coro, seconds: float | None):
    """Await ``coro`` under a time budget; return ``None`` if it runs out.

    ``None`` means "cut short", which is distinct from a probe that ran and found
    nothing (``[]``).  Collapsing the two is how a truncated probe comes to look
    like an empty codebase, so the distinction is carried all the way to
    :attr:`ActContextReport.budget_exceeded`.

    ``seconds=None`` means unbounded, for callers that have no budget to give.
    """
    if seconds is None:
        return await coro
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        return None


class _CandidateSlot:
    """Lexical candidates, published *before* the optional enrichment runs.

    The phase budget can fire at any moment once ``rg`` is spawning, and a timeout
    on the enclosing task would otherwise discard the lexical answer that had
    already been computed.  Carrying it in a slot lets the budget drop the
    enrichment without dropping the result it was enriching.
    """

    __slots__ = ("paths",)

    def __init__(self) -> None:
        self.paths: list[str] = []


async def _optional(coro, seconds: float | None):
    """Budgeted *enrichment*: ``None`` on timeout or on failure.

    For steps whose result improves the report but is not the report -- the
    content search through ``rg``.  A probe must not fail because an optional pass
    did, nor hang because one did.  Kept separate from :func:`_budgeted` so that
    budget bugs in the mandatory phases still surface instead of being swallowed.
    """
    try:
        return await _budgeted(coro, seconds)
    except Exception:
        return None


async def collect_codebase_context(
    root: Path,
    macro_prompt: str,
    timeout_seconds: float = DEFAULT_PROBE_BUDGET_S,
) -> ActContextReport:
    """Collect dense codebase context, staying inside ``timeout_seconds``.

    Each probe runs under the budget and a phase that overruns is dropped rather
    than waited for; the report then carries ``budget_exceeded=True`` so the
    caller can tell a partial answer from a complete one.  Compute the remaining
    budget for the snippet phase from the clock, so four phases cannot each spend
    the full allowance.
    """
    t0 = time.perf_counter()
    root = Path(root).resolve()
    proj_files = iter_project_files(root)

    # Launch parallel probes sharing the indexed file list
    task_schema = asyncio.create_task(_probe_schemas(root, proj_files))
    task_endpoints = asyncio.create_task(_probe_endpoints(root, proj_files))
    candidate_slot = _CandidateSlot()
    task_candidates = asyncio.create_task(
        _probe_target_candidates(
            root,
            macro_prompt,
            proj_files,
            budget_seconds=timeout_seconds,
            slot=candidate_slot,
        )
    )

    schema_res, endpoint_res, candidate_res = await asyncio.gather(
        _budgeted(task_schema, timeout_seconds),
        _budgeted(task_endpoints, timeout_seconds),
        _budgeted(task_candidates, timeout_seconds),
    )
    schemas, dialects = schema_res if schema_res is not None else ([], [])
    endpoints, frameworks = endpoint_res if endpoint_res is not None else ([], [])
    # A cut-short candidates phase still produced its lexical answer; take it,
    # rather than reporting "no candidates" for a search that did run.
    candidates = (
        candidate_res if candidate_res is not None else list(candidate_slot.paths)
    )

    # Extract & rank fine-grained code snippets based on candidates & query
    remaining = timeout_seconds - (time.perf_counter() - t0)
    if remaining > 0:
        snippet_res = await _budgeted(
            _probe_relevant_snippets(root, macro_prompt, candidates, proj_files), remaining
        )
    else:
        snippet_res = None
    relevant_snippets = snippet_res if snippet_res is not None else []

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    cut_short = any(
        r is None for r in (schema_res, endpoint_res, candidate_res, snippet_res)
    )

    return ActContextReport(
        repo_root=str(root),
        macro_prompt=macro_prompt,
        discovered_schemas=schemas,
        discovered_endpoints=endpoints,
        detected_dialects=dialects,
        active_frameworks=frameworks,
        candidate_target_files=candidates,
        relevant_snippets=relevant_snippets,
        probe_latency_ms=elapsed_ms,
        budget_ms=max(0.0, timeout_seconds * 1000.0),
        budget_exceeded=cut_short,
    )


async def _probe_relevant_snippets(
    root: Path,
    macro_prompt: str,
    candidate_files: list[str],
    proj_files: list[str],
) -> list[RelevantSnippet]:
    """Extract and rank candidate code snippets with calibrated task relevance probabilities."""
    from sylphae.coprocessor.snippet_extractor import extract_candidate_snippets
    from sylphae.coprocessor.ranker import SnippetRelevanceScorer

    search_files = list(candidate_files)
    query_l = macro_prompt.lower()
    if any(k in query_l for k in ("颜色", "样式", "theme", "color", "style", "css", "nav", "导航", "底色", "背景")):
        for f in proj_files:
            f_l = f.lower()
            if any(k in f_l for k in ("theme", "style", "nav", "header", "app.css", "main.css", "global")) and f not in search_files:
                search_files.append(f)
                if len(search_files) >= 10:
                    break

    try:
        raw_snippets = await asyncio.to_thread(
            extract_candidate_snippets,
            root=root,
            candidate_files=search_files,
            query=macro_prompt,
            max_candidates=12,
        )
        scorer = SnippetRelevanceScorer()
        ranked = scorer.rank_snippets(raw_snippets, query=macro_prompt, top_k=6)
        return ranked
    except Exception:
        return []


async def _probe_schemas(
    root: Path,
    proj_files: list[str],
) -> tuple[list[DiscoveredSchema], list[str]]:
    """Probe existing database tables, models, and migrations."""
    schemas: list[DiscoveredSchema] = []
    dialects: set[str] = set()

    def _sync_scan():
        for rel in proj_files:
            rel_l = rel.lower()
            if any(k in rel_l for k in ("schema", "model", "db", "migration", "database", "store", "table", "entity")):
                full = root / rel
                try:
                    st = full.stat()
                    if st.st_size > 256 * 1024 or st.st_size == 0:
                        continue
                    text = full.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue

                # 1. SQL CREATE TABLE
                for m in re.finditer(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_]+)\s*\((.*?)\);", text, re.I | re.DOTALL):
                    tbl_name = m.group(1)
                    cols_body = m.group(2)
                    col_names = [c.strip().split()[0] for c in cols_body.split(",") if c.strip() and not c.strip().upper().startswith(("PRIMARY", "FOREIGN", "KEY", "CONSTRAINT"))]
                    schemas.append(DiscoveredSchema(
                        name=tbl_name,
                        path=rel,
                        dialect="sqlite" if "sqlite" in rel_l or "sqlite" in text.lower() else "sql",
                        fields=col_names[:6],
                    ))
                    dialects.add("sqlite" if "sqlite" in rel_l or "sqlite" in text.lower() else "sql")
                    if len(schemas) >= 8:
                        return

                # 2. Python SQLAlchemy / SQLite / Django models
                if rel.endswith(".py"):
                    if "sqlite3" in text:
                        dialects.add("sqlite")
                    for m in re.finditer(r"class\s+([A-Za-z0-9_]+)\s*\([^)]*(?:Base|Model|Document)[^)]*\):", text):
                        cls_name = m.group(1)
                        schemas.append(DiscoveredSchema(
                            name=cls_name,
                            path=rel,
                            dialect="sqlalchemy" if "Column" in text else "python_model",
                        ))
                        dialects.add("sqlalchemy" if "Column" in text else "python_model")
                        if len(schemas) >= 8:
                            return

                # 3. Prisma Schema
                if rel.endswith(".prisma"):
                    dialects.add("prisma")
                    for m in re.finditer(r"model\s+([A-Za-z0-9_]+)\s*\{", text):
                        schemas.append(DiscoveredSchema(name=m.group(1), path=rel, dialect="prisma"))
                        if len(schemas) >= 8:
                            return

                # 4. Go Structs / GORM
                if rel.endswith(".go"):
                    for m in re.finditer(r"type\s+([A-Za-z0-9_]+)\s+struct\s*\{", text):
                        cls_name = m.group(1)
                        if any(kw in cls_name.lower() for kw in ("user", "auth", "account", "order", "item", "record", "config")):
                            schemas.append(DiscoveredSchema(name=cls_name, path=rel, dialect="go_struct"))
                            dialects.add("go_struct")
                            if len(schemas) >= 8:
                                return

                # 5. Rust Structs / Diesel
                if rel.endswith(".rs"):
                    for m in re.finditer(r"pub\s+struct\s+([A-Za-z0-9_]+)\s*\{", text):
                        cls_name = m.group(1)
                        if any(kw in cls_name.lower() for kw in ("user", "auth", "account", "config", "state")):
                            schemas.append(DiscoveredSchema(name=cls_name, path=rel, dialect="rust_struct"))
                            dialects.add("rust_struct")
                            if len(schemas) >= 8:
                                return

    await asyncio.to_thread(_sync_scan)
    return schemas, sorted(dialects)


async def _probe_endpoints(
    root: Path,
    proj_files: list[str],
) -> tuple[list[DiscoveredEndpoint], list[str]]:
    """Probe existing API routes and controllers."""
    endpoints: list[DiscoveredEndpoint] = []
    frameworks: set[str] = set()

    def _sync_scan():
        for rel in proj_files:
            rel_l = rel.lower()
            base = Path(rel_l).name
            is_cand = (
                any(k in base for k in ("route", "handler", "controller", "server", "main", "api"))
                or base in ("app.py", "app.ts", "app.js", "app.go")
                or any(f"/{k}/" in rel_l or rel_l.startswith(f"{k}/") for k in ("routes", "api", "handlers", "controllers", "endpoints"))
            )
            if not is_cand:
                continue

            full = root / rel
            try:
                st = full.stat()
                if st.st_size > 256 * 1024 or st.st_size == 0:
                    continue
                text = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            # Python FastAPI / Flask
            if rel.endswith(".py"):
                if "fastapi" in text:
                    frameworks.add("FastAPI")
                elif "flask" in text:
                    frameworks.add("Flask")
                for m in re.finditer(r'@(?:app|router)\.(get|post|put|delete)\s*\(\s*["\']([^"\']+)["\']\s*\)\s*(?:async\s+)?def\s+([A-Za-z0-9_]+)', text, re.I):
                    endpoints.append(DiscoveredEndpoint(
                        method=m.group(1).upper(),
                        route=m.group(2),
                        handler_name=m.group(3),
                        source_file=rel,
                    ))
                    if len(endpoints) >= 8:
                        return

            # Go net/http / Gin
            if rel.endswith(".go"):
                frameworks.add("Go http")
                for m in re.finditer(r'http\.HandleFunc\s*\(\s*["\']([^"\']+)["\']\s*,\s*([A-Za-z0-9_\.]+)', text):
                    endpoints.append(DiscoveredEndpoint(
                        method="ALL",
                        route=m.group(1),
                        handler_name=m.group(2),
                        source_file=rel,
                    ))
                    if len(endpoints) >= 8:
                        return

            # Next.js / Express
            if rel.endswith((".ts", ".tsx", ".js")):
                if "next" in text or "react" in text:
                    frameworks.add("Next.js")
                elif "express" in text:
                    frameworks.add("Express")
                for m in re.finditer(r'export\s+async\s+function\s+(GET|POST|PUT|DELETE)\b', text):
                    endpoints.append(DiscoveredEndpoint(
                        method=m.group(1),
                        route=rel.replace("\\", "/"),
                        handler_name=m.group(1),
                        source_file=rel,
                    ))
                    if len(endpoints) >= 8:
                        return

    await asyncio.to_thread(_sync_scan)
    return endpoints, sorted(frameworks)


async def _probe_target_candidates(
    root: Path,
    macro_prompt: str,
    proj_files: list[str] | None = None,
    budget_seconds: float | None = None,
    slot: "_CandidateSlot | None" = None,
) -> list[str]:
    """Identify top candidate files for modifications based on macro prompt.

    Two passes: a path-lexical match, which is in-process and costs milliseconds,
    and an optional content search through ``rg``.  The second one is not cheap --
    measured on Windows, one ``rg`` invocation costs ~300ms to spawn, which is
    twice the whole probe target on its own -- so it runs under ``budget_seconds``
    and is abandoned rather than waited for.  The lexical candidates are returned
    either way: the content pass is an enrichment, not the answer, and ``slot``
    carries them out even if the enclosing phase budget fires first.

    ``budget_seconds=None`` means unbounded, so a caller with no deadline keeps
    exactly the old behaviour.
    """
    from sylphae.agent.tools import get_ripgrep_binary, path_lexical_search, ripgrep_search
    keywords = extract_keywords(macro_prompt)
    if not keywords:
        return []

    if proj_files is None:
        proj_files = iter_project_files(root)

    # 1. Ultra-fast path lexical match (~10-30ms)
    lex_matches = await path_lexical_search(keywords, proj_files, k=5)
    candidates = [m.path for m in lex_matches]
    if slot is not None:
        slot.paths = list(candidates)

    # 2. If ripgrep is available and we need more candidates, fast content search.
    #    Skipped outright when the lexical pass already returned every project
    #    file: a content search can only ever surface files, and there are none
    #    left for it to surface.  That check costs one comparison and saves the
    #    process spawn on exactly the small repositories that would pay it.
    rg_bin = get_ripgrep_binary()
    content_search_can_help = len(candidates) < min(5, len(proj_files))
    if rg_bin and content_search_can_help:
        content_matches = await _optional(
            ripgrep_search(keywords[:3], root, max_matches=5), budget_seconds
        )
        for m in content_matches or []:
            if m.path not in candidates:
                candidates.append(m.path)
            if len(candidates) >= 5:
                break

    return candidates[:5]
