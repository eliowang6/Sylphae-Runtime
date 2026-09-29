"""Parallel Act Plan Executor for Frontier LLM Coprocessor workflows.

Accepts structured ActPlans from LLMs, executes concurrent multi-file code modifications,
synthesizes schemas/models, and verifies syntax in parallel.
"""

from __future__ import annotations

import asyncio
import ast
import re
import time
from pathlib import Path

from sylphae.agent.explorer import ParallelExplorer
from sylphae.coprocessor.generator import (
    parse_field_spec,
    parse_table_description,
    render_field_addition,
    render_table_schema,
    synthesize_code_snippet,
)
from sylphae.coprocessor.protocol import (
    ActExecutionReport,
    ActPlan,
    PlanStep,
    StepExecutionResult,
)
from sylphae.exec.heal import index_repo, verify_and_heal
from sylphae.exec.patch import _resolve_inside, apply_to_tree
from sylphae.exec.textio import line_ending, read_source, with_ending, write_source
from sylphae.repo.polyglot import iter_project_files


def _is_test_path(rel: str) -> bool:
    """A test file by name convention, so the step runs the tests its edit can touch."""
    norm = rel.replace("\\", "/")
    last = norm.rsplit("/", 1)[-1]
    return last.startswith("test_") and last.endswith(".py")


async def execute_act_plan(
    root: Path,
    plan: ActPlan | str | dict | list | Path,
) -> ActExecutionReport:
    """Parallelly execute all steps in an ActPlan across the repository.
    
    Accepts canonical ActPlan, flat dicts, Markdown checklists, micro-DSL, or natural language.
    """
    t0 = time.perf_counter()
    root = Path(root).resolve()

    if not isinstance(plan, ActPlan):
        from sylphae.coprocessor.protocol import parse_act_plan
        plan = parse_act_plan(plan, root=root)

    # Launch all plan steps concurrently
    tasks = [
        asyncio.create_task(_execute_single_step(root, step))
        for step in plan.steps
    ]

    results = await asyncio.gather(*tasks, return_exceptions=False)
    total_ms = (time.perf_counter() - t0) * 1000.0

    service_reports = [r.service_report for r in results if r.service_report is not None]
    all_passed = all(r.applied and r.syntax_ok for r in results)

    return ActExecutionReport(
        plan_id=plan.plan_id,
        step_results=results,
        service_reports=service_reports,
        all_passed=all_passed,
        total_latency_ms=total_ms,
    )


def _struct_pattern(ext: str, table_name: str) -> str | None:
    """Regex matching the struct/model body a field is inserted into, or None.

    The name is matched **case-insensitively against the whole declared name**: the request
    may say ``users`` while the file declares ``model Users``, or say ``RunnerConfig`` while
    the file declares the same thing -- and ``"RunnerConfig".capitalize()`` is
    ``"Runnerconfig"``, which silently matched nothing.  The whole-word guard stops
    ``Config`` from matching inside ``RunnerConfig``.
    """
    name = re.escape(table_name)
    if ext == ".prisma":
        return rf"(?<![\w])model\s+{name}\s*\{{[^}}]*\}}"
    if ext == ".go":
        return rf"type\s+{name}\s+struct\s*\{{[^}}]*\}}"
    if ext == ".rs":
        return rf"struct\s+{name}\s*\{{[^}}]*\}}"
    return None


def _python_delete_span(source: str, name: str) -> tuple[int, int] | None:
    """Line span (0-based, inclusive) of the top-level definition called ``name``.

    Used by ``delete_block`` on Python files.  It replaced a regex,
    ``^[ \\t]*(?:async\\s+)?(?:def|class)\\s+NAME\\b[^\\r\\n]*:(?:\\r?\\n(?:[ \\t]+[^\\r\\n]*|$))*``,
    which had two independent failure modes on real repositories, both silent:

    * the header had to end with ``:`` on the *same line* as ``def``, so any multi-line
      signature (``def f(\\n    x: int\\n) -> None:``) never matched and the step reported
      "could not find the block" for a block that was right there;
    * a decorator above the def was not part of the match, so deleting the def left the
      decorator dangling over whatever came next -- a syntax error reported as
      ``applied=True``.

    ``ast`` gives the exact span, decorators included.  The regex is kept only as a fallback
    for files that do not parse at all, where line numbers are a lost cause anyway.

    The returned span ends on the last line of the definition *plus* any blank lines that
    immediately follow it, because those are the block's separator and removing them keeps
    the surrounding blocks apart.  ``None`` when there is no such top-level definition.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        pattern = (
            rf"(?m)^[ \t]*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b[^\r\n]*:"
            rf"(?:\r?\n(?:[ \t]+[^\r\n]*|$))*"
        )
        m = re.search(pattern, source)
        if not m:
            return None
        start = source[: m.start()].count("\n")
        end = source[: m.end()].count("\n") - 1
        return (start, end)

    target = None
    for node in tree.body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ) and node.name == name:
            target = node
            break
    if target is None or target.end_lineno is None:
        return None
    start = target.lineno - 1
    if target.decorator_list:
        start = min(d.lineno for d in target.decorator_list) - 1
    end = target.end_lineno - 1
    lines = source.splitlines()
    while end + 1 < len(lines) and lines[end + 1].strip() == "":
        end += 1
    return (start, end)


async def _execute_single_step(
    root: Path,
    step: PlanStep,
) -> StepExecutionResult:
    """Execute a single plan step, locating target, generating code, and validating."""
    t_start = time.perf_counter()

    # Special handling for operational service management
    if step.action == "start_service":
        from sylphae.coprocessor.service_runner import launch_and_verify_service
        srv_rep = await launch_and_verify_service(root, target_hint=step.target_hint or "backend")
        step_ms = (time.perf_counter() - t_start) * 1000.0
        applied = srv_rep.status in ("running", "already_running")
        syntax_ok = applied
        diff = (
            f"Service '{srv_rep.service_name}' {srv_rep.status} at {srv_rep.url} "
            f"(PID {srv_rep.pid}, {srv_rep.latency_ms:.1f}ms)"
            if applied
            else f"Service '{srv_rep.service_name}' {srv_rep.status}"
        )
        return StepExecutionResult(
            step_id=step.step_id,
            target_path=srv_rep.working_dir or "server",
            op_kind=step.action,
            applied=applied,
            syntax_ok=syntax_ok,
            diff_summary=diff,
            error=srv_rep.error_summary if not applied else None,
            latency_ms=step_ms,
            service_report=srv_rep,
        )

    # 1. Resolve target path
    target_path = await _resolve_step_target(root, step)
    if not target_path:
        return StepExecutionResult(
            step_id=step.step_id,
            target_path=step.target_hint or "unknown",
            op_kind=step.action,
            applied=False,
            syntax_ok=False,
            error=f"Could not locate target file for '{step.description}'",
            latency_ms=(time.perf_counter() - t_start) * 1000.0,
        )

    resolved_target = _resolve_inside(root, target_path)
    if resolved_target is None:
        return StepExecutionResult(
            step_id=step.step_id,
            target_path=target_path,
            op_kind=step.action,
            applied=False,
            syntax_ok=False,
            error=f"Security error: target path escapes repository root: {target_path}",
            latency_ms=(time.perf_counter() - t_start) * 1000.0,
        )

    full_target = resolved_target
    target_path = resolved_target.relative_to(root).as_posix()
    ext = full_target.suffix.lower()

    # 2. Synthesize code modification based on action
    applied = False
    diff_summary = ""
    error_msg = None

    try:
        if step.action == "create_table":
            table_spec = step.schema_spec or parse_table_description(step.description)
            dialect = "sqlite"
            if ext == ".sql":
                dialect = "sqlite"
            elif ext == ".py":
                dialect = "sqlalchemy" if (full_target.exists() and "Base" in read_source(full_target)) else "sqlite"
            elif ext == ".prisma":
                dialect = "prisma"
            elif ext == ".go":
                dialect = "go"
            elif ext == ".rs":
                dialect = "rust"

            schema_code = render_table_schema(table_spec, target_dialect=dialect)

            if not full_target.exists():
                full_target.parent.mkdir(parents=True, exist_ok=True)
                write_source(full_target, schema_code)
                applied = True
                diff_summary = f"Created {target_path} with table {table_spec.name}"
            else:
                old_text = read_source(full_target)
                ending = line_ending(old_text)
                new_text = old_text.rstrip() + ending * 2 + with_ending(schema_code, ending)
                write_source(full_target, new_text)
                applied = True
                diff_summary = f"Appended table {table_spec.name} to {target_path}"

        elif step.action == "add_field":
            table_name, col_spec = parse_field_spec(step.description)
            dialect = "sqlite"
            if ext == ".sql":
                dialect = "sqlite"
            elif ext == ".prisma":
                dialect = "prisma"
            elif ext == ".go":
                dialect = "go"
            elif ext == ".rs":
                dialect = "rust"
            elif ext == ".lua":
                dialect = "lua"

            field_code = render_field_addition(table_name, col_spec, target_dialect=dialect)
            if not full_target.exists():
                full_target.parent.mkdir(parents=True, exist_ok=True)
                write_source(full_target, field_code)
                applied = True
                diff_summary = f"Created {target_path} with field {col_spec.name}"
            else:
                old_text = read_source(full_target)
                # The generated field uses LF.  Spliced into a CRLF file unchanged it would
                # leave that file carrying two conventions at once, so it is re-ended to
                # match the file it is going into.  MEASURED 2026-09-23.
                ending = line_ending(old_text)
                field_in_file = with_ending(field_code, ending)
                inserted = False
                struct_re = _struct_pattern(ext, table_name)
                if struct_re is not None:
                    # .prisma / .go / .rs: the field belongs *inside* the named struct, and
                    # only there.  Matching is case-insensitive on the whole declared name --
                    # `"RunnerConfig".capitalize()` is `"Runnerconfig"`, which matched
                    # nothing and the step used to fall through to the append below, placing
                    # a struct member at module level.  That is not valid in any of these
                    # languages and verify() has no compiler for them, so it was reported
                    # applied=True syntax_ok=True on a file that would not build.
                    # MEASURED 2026-09-23.
                    m = re.search(struct_re, old_text, re.DOTALL | re.IGNORECASE)
                    if m:
                        idx = m.end() - 1
                        new_text = old_text[:idx] + field_in_file + old_text[idx:]
                        write_source(full_target, new_text)
                        applied = True
                        diff_summary = f"Added field {col_spec.name} to {table_name} in {target_path}"
                        inserted = True
                    else:
                        error_msg = (
                            f"struct/model {table_name!r} not found in {target_path}; a struct "
                            "field cannot be appended at module level, so the file was not "
                            "written -- escalate the step or name the struct"
                        )
                if not inserted and not error_msg:
                    # .sql and .lua: an ALTER TABLE is a legal standalone statement, so
                    # appending it is a real edit.  This is deliberately not the path the
                    # struct dialects take.
                    new_text = old_text.rstrip() + ending + field_in_file
                    write_source(full_target, new_text)
                    applied = True
                    diff_summary = f"Added field {col_spec.name} to {target_path}"

        elif step.action in ("add_function", "insert_statement"):
            lang = "python"
            if ext in (".ts", ".tsx", ".js"):
                lang = "typescript"
            elif ext == ".go":
                lang = "go"
            elif ext == ".rs":
                lang = "rust"

            # A snippet that is blank after stripping names no content.  Honouring it would
            # append whitespace and report applied=True -- the "instruction did nothing and
            # said it did" shape.  A missing snippet is different: the description is meant
            # to be synthesised from, so that path is kept.  MEASURED 2026-09-23.
            if step.code_snippet is not None and not step.code_snippet.strip():
                error_msg = (
                    "code snippet is empty or whitespace-only; nothing to insert "
                    "(no content was named, so the step was refused rather than honoured)"
                )
            else:
                code = step.code_snippet or synthesize_code_snippet(step.description, target_lang=lang)

                if not full_target.exists():
                    full_target.parent.mkdir(parents=True, exist_ok=True)
                    write_source(full_target, code)
                    applied = True
                    diff_summary = f"Created {target_path} with {step.description}"
                else:
                    old_text = read_source(full_target)
                    ending = line_ending(old_text)
                    new_text = old_text.rstrip() + ending * 2 + with_ending(code, ending)
                    write_source(full_target, new_text)
                    applied = True
                    diff_summary = f"Inserted implementation into {target_path}"

        elif step.action == "delete_block":
            if full_target.exists():
                content = read_source(full_target)
                ending = line_ending(content)
                desc = step.description.strip()
                desc_clean = re.sub(r"^[-!*~+]?\s*(?:delete|remove|rm|drop)\s*", "", desc, flags=re.I)
                target_name = desc_clean.split()[0].strip() if desc_clean else ""
                if "(" in target_name:
                    target_name = target_name.split("(")[0].strip()

                deleted = False
                # The snippet arrives from a plan, which is written with LF.  It is re-ended
                # to the file it is being searched in rather than the file being normalised
                # to the snippet: the untouched bytes must stay untouched.  MEASURED
                # 2026-09-23: an LF snippet cannot match raw CRLF content at all.
                snippet = with_ending(step.code_snippet, ending) if step.code_snippet else ""
                if snippet and snippet in content:
                    content = content.replace(snippet, "")
                    deleted = True
                elif target_name:
                    if ext == ".py":
                        # ast-based: the regex it replaced could not match a multi-line
                        # signature and left decorators dangling.  See
                        # ``_python_delete_span``.
                        span = _python_delete_span(content, target_name)
                        if span is not None:
                            keepends = content.splitlines(keepends=True)
                            content = "".join(keepends[: span[0]]) + "".join(
                                keepends[span[1] + 1 :]
                            )
                            deleted = True
                    else:
                        p_go = r"(?s)(?:func|type)\s+" + re.escape(target_name) + r"\b[^{]*\{[^}]*\}"
                        p_sql = (
                            rf"(?is)CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
                            rf"{re.escape(target_name)}\b[^;]*;\s*"
                        )
                        for pat in (p_go, p_sql):
                            if re.search(pat, content):
                                content = re.sub(pat, "", content, count=1)
                                deleted = True
                                break

                if deleted:
                    write_source(full_target, content)
                    applied = True
                    diff_summary = f"Deleted block '{target_name or desc}' from {target_path}"
                else:
                    error_msg = f"Could not find block to delete matching '{step.description}'"
            else:
                error_msg = f"Target file does not exist: {target_path}"

        elif step.action in ("replace_literal", "style", "fix", "update"):
            if step.code_snippet and "->" in step.code_snippet:
                parts = step.code_snippet.split("->", 1)
                old_str, new_str = parts[0].strip(), parts[1].strip()
                if full_target.exists():
                    old_content = read_source(full_target)
                    ending = line_ending(old_content)
                    # Same rule as delete_block: re-end both sides of the substitution to the
                    # file, so a CRLF file is matched and edited without being normalised.
                    search_str = with_ending(old_str, ending)
                    replace_str = with_ending(new_str, ending)
                    if search_str in old_content:
                        new_content = old_content.replace(search_str, replace_str, 1)
                        write_source(full_target, new_content)
                        applied = True
                        diff_summary = f"Replaced '{old_str}' with '{new_str}' in {target_path}"
                    else:
                        error_msg = f"Search string '{old_str}' not found in {target_path}"
                else:
                    error_msg = f"Target file does not exist: {target_path}"
            else:
                from sylphae.agent.effort import LowEffort
                from sylphae.agent.reflex import ReflexAgent
                agent = ReflexAgent(root=root, effort=LowEffort)
                spec, new_source = agent._build_patch(prompt=step.description, target_path=target_path, op_kind="replace_literal", param_val="")
                if spec is not None and new_source is not None:
                    apply_res = apply_to_tree(root, target_path, spec, dry_run=False)
                    applied = apply_res.ok
                    diff_summary = f"Reflex edit applied to {target_path}"
                    if not applied:
                        error_msg = f"Patch apply failed: {apply_res.error}"
                else:
                    res = await agent.run(prompt=step.description, dry_run=False, verify_in_background=False)
                    applied = res.immediate_patch_ok
                    diff_summary = f"Reflex edit applied to {res.target_path}"
                    if not applied and res.escalation_reason:
                        error_msg = res.escalation_reason

    except Exception as exc:
        error_msg = str(exc)
        applied = False

    # 3. Verify-and-heal
    #
    # This used to call ``verify(root, [target_path], [])`` -- an empty test list, so it could
    # only ever check ``compile()``.  A step that broke a test reported ``syntax_ok=True``,
    # which is how plans came back green with broken files.  Now the step runs against the
    # tests its edit can actually touch, and the loop repairs a mechanical failure on its own
    # before escalating.  MEASURED 2026-09-27.
    syntax_ok = False
    heal_note = ""
    warnings: list[str] = []
    if applied:
        all_tests = [f for f in iter_project_files(root) if _is_test_path(f)]
        symbols, modules = index_repo(root)
        rep, outcome = verify_and_heal(
            root, [target_path], all_tests, repo_symbols=symbols, repo_modules=modules
        )
        syntax_ok = rep.compiles
        # A real defect is a compile failure, or tests that ran and failed.  "No test touches
        # this file" is a limit on what could be verified, not a defect in the edit -- it goes
        # to ``warnings`` so a test-free target can still report a clean pass.
        no_tests = any(e.startswith("no-tests:") for e in rep.errors)
        tests_ran = rep.tests_run > 0
        tests_ok = tests_ran and not rep.errors
        if outcome.attempts:
            trail = "; ".join(
                f"attempt {a.n}: {a.failure_class} -> {a.repair} ({'cleared' if a.cleared else 'kept'})"
                for a in outcome.attempts
            )
            heal_note = f" [self-heal: {trail}]"
        if not syntax_ok:
            error_msg = f"Syntax errors: {rep.errors}{heal_note}"
        elif not tests_ok and not no_tests:
            # The escalation carries the classified failure, so the planner inherits the
            # evidence rather than a bare "it broke".
            error_msg = (
                f"Verification failed: {'; '.join(rep.errors)}"
                f"{heal_note}"
                + (f" -- escalated: {outcome.escalation}" if outcome.escalation else "")
            )
        if syntax_ok and no_tests:
            warnings.append(
                "compile-only verification: no test file touches this target, so the edit "
                "was checked by compile() alone"
            )

    step_ms = (time.perf_counter() - t_start) * 1000.0

    return StepExecutionResult(
        step_id=step.step_id,
        target_path=target_path,
        op_kind=step.action,
        applied=applied,
        syntax_ok=syntax_ok,
        diff_summary=diff_summary,
        error=error_msg,
        warnings=warnings,
        latency_ms=step_ms,
    )


async def _resolve_step_target(root: Path, step: PlanStep) -> str | None:
    """Resolve target file path using hint, filename search, or explorer."""
    proj_files = iter_project_files(root)

    # 1. Target hint exact or substring match
    if step.target_hint:
        hint_clean = step.target_hint.strip().replace("\\", "/").lower()
        for f in proj_files:
            f_norm = f.replace("\\", "/").lower()
            if hint_clean in f_norm:
                return f
        # Extension matching - ensure hint does not escape root
        if "." in hint_clean:
            candidate = _resolve_inside(root, step.target_hint.strip())
            if candidate is None:
                return None
            return candidate.relative_to(root).as_posix()

    # 2. Action specific heuristics
    if step.action in ("create_table", "add_field"):
        # Look for schema / db / migrations / models
        for f in proj_files:
            f_l = f.lower()
            if any(k in f_l for k in ("schema.sql", "models.py", "schema.prisma", "db.go", "schema.rs", "database.py")):
                return f
        # Default schema file if none exists
        return "schema.sql"

    # 3. Parallel Explorer search
    explorer = ParallelExplorer(root=root)
    async for cands in explorer.stream_explore(step.description, early_lock_threshold=0.80):
        if cands:
            return cands[0].path

    # Fallback to first project file or default
    return proj_files[0] if proj_files else "main.py"
