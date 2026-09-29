"""AI Semantic Grounding and Natural Language Decomposition Layer.

Translates freeform, conversational human natural language queries into canonical
PlanSteps by combining AI intent classification with factual codebase probe evidence.
Eliminates brittle regex hardcoding and ensures multi-language compatibility.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from sylphae.agent.explorer import extract_keywords
from sylphae.agent.tools import path_lexical_search
from sylphae.coprocessor.protocol import PlanStep
from sylphae.coprocessor.service_detect import (
    detect_service_entrypoint,
    detect_service_flavor,
)
from sylphae.repo.polyglot import iter_project_files


def split_clauses_semantic(text: str) -> list[str]:
    """Split a compound natural language request into discrete semantic intent clauses."""
    # Split on line breaks, semicolons, conjunctions, and commas outside parentheses
    delimiters = r"(?:[\r\n;；]+|[,，](?![^(]*\))\s*(?:另外|顺便|同时|并且|然后|接着|还有|以及|\balso\b|\band\s+then\b|\bfurthermore\b)?\s*)"
    raw_parts = re.split(delimiters, text.strip(), flags=re.IGNORECASE)
    cleaned = [p.strip() for p in raw_parts if p and len(p.strip()) >= 3]
    return cleaned if cleaned else [text.strip()]


async def ai_ground_natural_language(
    query: str,
    root: Path | None = None,
) -> list[PlanStep]:
    """Decompose freeform natural language query into grounded PlanSteps using physical evidence."""
    clauses = split_clauses_semantic(query)
    steps: list[PlanStep] = []

    root_path = Path(root).resolve() if root else None
    proj_files = iter_project_files(root_path) if root_path and root_path.exists() else []

    # Fast probe existing schema context if root available
    existing_tables: dict[str, str] = {}  # table_name -> file_path
    if root_path and proj_files:
        from sylphae.coprocessor.collector import _probe_schemas
        schemas, _ = await _probe_schemas(root_path, proj_files)
        for s in schemas:
            existing_tables[s.name.lower()] = s.path

    # Probe service feasibility once per call, not once per clause.  ``None`` means
    # "no repository to probe" and must not be read as "no service here" — see
    # sylphae.coprocessor.service_detect for why that distinction is load-bearing.
    service_entrypoint = detect_service_entrypoint(root_path, proj_files)
    service_flavor = detect_service_flavor(root_path, proj_files)

    for i, clause in enumerate(clauses, 1):
        step = await _ground_single_clause(
            clause_idx=i,
            clause=clause,
            root=root_path,
            proj_files=proj_files,
            existing_tables=existing_tables,
            service_entrypoint=service_entrypoint,
            service_flavor=service_flavor,
        )
        steps.append(step)

    return steps


def ground_natural_language_sync(
    query: str,
    root: Path | None = None,
) -> list[PlanStep]:
    """Synchronous entry point for natural language grounding."""
    try:
        loop = asyncio.get_running_loop()
        # If in active event loop, run in thread or create task
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            fut = executor.submit(lambda: asyncio.run(ai_ground_natural_language(query, root=root)))
            return fut.result()
    except RuntimeError:
        return asyncio.run(ai_ground_natural_language(query, root=root))


async def _ground_single_clause(
    clause_idx: int,
    clause: str,
    root: Path | None,
    proj_files: list[str],
    existing_tables: dict[str, str],
    service_entrypoint: bool | None = None,
    service_flavor: str | None = None,
) -> PlanStep:
    """Ground a single natural language clause to target file and semantic action using AI SemanticGrounder."""
    from sylphae.model.semantic_grounder import get_semantic_grounder

    step_id = f"step_{clause_idx}"
    grounder = get_semantic_grounder()
    keywords = extract_keywords(clause)

    # 1. Resolve candidate target files from physical codebase evidence
    target_hint = ""
    if proj_files:
        lex_matches = await path_lexical_search(keywords, proj_files, k=3)
        if lex_matches:
            target_hint = lex_matches[0].path

    # 2. Extract semantic domain entity (e.g. users, orders, products, configs)
    semantic_ent = grounder.extract_semantic_entity(clause)
    matched_table_path = existing_tables.get(semantic_ent.normalized_name.lower())

    # 3. AI Semantic Intent Classification with Bayesian Physical Conditioning
    physical_ctx = {
        "existing_tables": existing_tables,
        # Tri-state on purpose: True/False are findings, None is "not probed".
        "has_service_target": service_entrypoint,
    }
    pred = grounder.predict_action(clause, physical_context=physical_ctx)
    action = pred.action

    # 4. Synthesize Physical Target Hint based on Grounded Action & Codebase Evidence
    if action == "start_service":
        # Which service to start is a fact about the repository, not about the wording
        # of the request.  This used to scan the clause for the literals
        # "前端"/"frontend"/"client"/"vite" — a keyword table AGENTS.md §2.1 rules out,
        # and one that answered the wrong question anyway: the phrasing says what the
        # user *mentioned*, not what is actually there to run.
        #
        # ``service_flavor`` carries that answer.  It is ``None`` when neither kind of
        # service could be established, and the fallback label ``"backend"`` then stays
        # in the report -- harmless, because discovery reads the disk rather than this
        # string, but it should not be read as a finding.
        return PlanStep(
            step_id=step_id,
            action=action,
            description=clause,
            target_hint=service_flavor or "backend",
        )

    if action in ("create_table", "add_field"):
        if matched_table_path:
            target_hint = matched_table_path
        elif not target_hint or not any(target_hint.lower().endswith(ext) for ext in (".sql", ".prisma", ".py", ".go", ".rs")):
            # Find best schema / model file in repository
            for f in proj_files:
                if any(s in f.lower() for s in ("schema.sql", "models.py", "schema.prisma", "db.go", "schema.rs", "database.py")):
                    target_hint = f
                    break
            if not target_hint:
                target_hint = "schema.sql"

        return PlanStep(
            step_id=step_id,
            action=action,
            description=clause,
            target_hint=target_hint,
        )

    if action == "replace_literal":
        # Target UI components or styles
        if not target_hint or not any(target_hint.lower().endswith(ext) for ext in (".vue", ".tsx", ".jsx", ".css", ".scss", ".html")):
            for f in proj_files:
                if any(f.lower().endswith(ext) for ext in (".vue", ".tsx", ".jsx", ".css", ".scss")):
                    target_hint = f
                    break
        return PlanStep(
            step_id=step_id,
            action=action,
            description=clause,
            target_hint=target_hint,
        )

    if action == "insert_statement":
        # Target API routers or handlers
        if not target_hint or not any(target_hint.lower().endswith(ext) for ext in (".py", ".go", ".ts", ".js")):
            for f in proj_files:
                if any(k in f.lower() for k in ("router", "route", "api", "server", "main.py")):
                    target_hint = f
                    break
        return PlanStep(
            step_id=step_id,
            action=action,
            description=clause,
            target_hint=target_hint,
        )

    # Fallback to functional addition or general code edit
    return PlanStep(
        step_id=step_id,
        action=action,
        description=clause,
        target_hint=target_hint,
    )
