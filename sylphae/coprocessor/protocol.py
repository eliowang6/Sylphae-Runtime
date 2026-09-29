"""Frontier LLM <-> Fast Act Coprocessor Protocol.

Defines the bidirectional communication contract:
1. Frontier LLM requests fast codebase analysis -> Sylphae emits ActContextReport
   inside its probe budget (see :data:`sylphae.coprocessor.collector.TARGET_PROBE_MS`
   for what that budget does and does not cover).
2. Frontier LLM formulates macro-plan -> Sylphae receives ActPlan and parallelly executes it.
3. Sylphae returns ActExecutionReport with verified diffs, AST patches, and compilation results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


@dataclass
class DiscoveredSchema:
    name: str
    path: str
    dialect: str  # 'sqlite', 'postgres', 'mysql', 'prisma', 'sqlalchemy', 'gorm', 'diesel'
    fields: list[str] = field(default_factory=list)


@dataclass
class DiscoveredEndpoint:
    route: str
    method: str
    handler_name: str
    source_file: str


@dataclass
class RelevantSnippet:
    file_path: str
    start_line: int
    end_line: int
    code_snippet: str
    kind: str  # 'base_style', 'theme_override', 'visual_dominant', 'nested_control', 'model_field', 'route_handler'
    importance: float  # Calibrated probability in [0.0, 1.0]
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActContextReport:
    """Fast, high-density codebase intelligence emitted to the LLM."""
    repo_root: str
    macro_prompt: str
    discovered_schemas: list[DiscoveredSchema] = field(default_factory=list)
    discovered_endpoints: list[DiscoveredEndpoint] = field(default_factory=list)
    detected_dialects: list[str] = field(default_factory=list)
    active_frameworks: list[str] = field(default_factory=list)
    candidate_target_files: list[str] = field(default_factory=list)
    relevant_snippets: list[RelevantSnippet] = field(default_factory=list)
    probe_latency_ms: float = 0.0
    #: The budget the probe ran under, in milliseconds.
    budget_ms: float = 0.0
    #: ``True`` when a phase ran out of budget and was dropped.  Without this, a
    #: truncated probe and a codebase with nothing to find produce the same empty
    #: report and the caller has no way to tell them apart.
    budget_exceeded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prompt_context(self) -> str:
        """Render a concise context block for immediate LLM prompt injection."""
        lines = [
            f"=== Codebase Context Probe ({self.probe_latency_ms:.1f}ms) ===",
        ]
        if self.budget_exceeded:
            # Say so at the top: a reader who cannot see this will read an empty
            # section as "the codebase has none of this".
            lines.append(
                f"INCOMPLETE: the probe hit its {self.budget_ms:.0f}ms budget and "
                f"dropped at least one section below."
            )
        lines.extend([
            f"Frameworks: {', '.join(self.active_frameworks) or 'standard'}",
            f"Database / ORM: {', '.join(self.detected_dialects) or 'unknown'}",
        ])
        if self.discovered_schemas:
            lines.append("Existing Models & Schemas:")
            for s in self.discovered_schemas[:4]:
                fields_str = f" ({', '.join(s.fields[:3])})" if s.fields else ""
                lines.append(f"  - {s.name} in {s.path}{fields_str}")
        if self.discovered_endpoints:
            lines.append("Existing Routes / Handlers:")
            for ep in self.discovered_endpoints[:4]:
                lines.append(f"  - {ep.method} {ep.route} in {ep.source_file}")
        if self.candidate_target_files:
            lines.append(f"Suggested Modification Targets: {', '.join(self.candidate_target_files[:5])}")
        if self.relevant_snippets:
            lines.append("Relevant Code Snippets (Ranked by Importance):")
            for idx, snip in enumerate(self.relevant_snippets[:6], 1):
                kind_tag = f" ({snip.kind.replace('_', ' ').title()})" if snip.kind else ""
                lines.append(f"  {idx}. [P={snip.importance:.2f}]{kind_tag} {snip.file_path}:{snip.start_line}-{snip.end_line}")
                for s_line in snip.code_snippet.strip().splitlines()[:3]:
                    lines.append(f"     {s_line}")
                if snip.rationale:
                    lines.append(f"     Rationale: {snip.rationale}")
        lines.append("==================================================")
        return "\n".join(lines)


@dataclass
class ColumnSpec:
    name: str
    col_type: str = "TEXT"
    primary_key: bool = False
    not_null: bool = False
    default: str | None = None


@dataclass
class TableSpec:
    name: str
    columns: list[ColumnSpec] = field(default_factory=list)
    dialect: str = "sqlite"


@dataclass
class PlanStep:
    step_id: str
    action: str  # 'create_table', 'insert_statement', 'add_function', 'replace_literal'
    description: str
    target_hint: str = ""
    schema_spec: TableSpec | None = None
    code_snippet: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActPlan:
    """Structured plan emitted by Frontier LLM or autonomous planner."""
    plan_id: str
    macro_intent: str
    steps: list[PlanStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepExecutionResult:
    step_id: str
    target_path: str
    op_kind: str
    applied: bool
    syntax_ok: bool
    diff_summary: str = ""
    error: str | None = None
    #: Non-fatal findings that do not affect ``all_passed``.  Verification that could not run
    #: any test reports here rather than failing the step: an edit is not broken because the
    #: repo has no test to exercise it.
    warnings: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    service_report: ServiceLaunchReport | None = None


@dataclass
class ServiceLaunchReport:
    """Detailed report from launching or inspecting a local development/test service."""
    status: str  # 'running', 'already_running', 'failed', 'unresolvable'
    service_name: str
    pid: int | None = None
    port: int | None = None
    url: str | None = None
    runtime_path: str | None = None
    working_dir: str | None = None
    latency_ms: float = 0.0
    log_file: str | None = None
    error_summary: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActExecutionReport:
    """Aggregated report of parallel execution results across all plan steps."""
    plan_id: str
    step_results: list[StepExecutionResult] = field(default_factory=list)
    service_reports: list[ServiceLaunchReport] = field(default_factory=list)
    all_passed: bool = False
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActionVerb(str, Enum):
    CREATE = "create_table"
    ADD_FIELD = "add_field"
    ADD_FUNC = "add_function"
    ROUTE = "insert_statement"
    UPDATE = "replace_literal"
    STYLE = "replace_literal"
    DELETE = "delete_block"
    FIX = "replace_literal"
    START_SERVICE = "start_service"
    AUTO = "auto"


@dataclass
class SemiAtomicAction:
    target_path: str
    verb: ActionVerb
    description: str
    raw_text: str = ""


def _parse_directive_to_step(step_id: str, target: str, text: str) -> PlanStep:
    """Parse a single semi-atomic line or directive into a PlanStep."""
    import re
    cleaned = re.sub(r"^(?:[-*]\s+|\d+\.\s*)(?:\[[ xX]\]\s*)?", "", text).strip()
    directive = cleaned

    if not target:
        # Check if line has "path: directive" or "path -> directive"
        m_delim = re.split(r"\s*(?::|->)\s*", cleaned, maxsplit=1)
        if len(m_delim) == 2 and ("." in m_delim[0] or "/" in m_delim[0]):
            target = m_delim[0].strip()
            directive = m_delim[1].strip()

    # 1. Bracket verb tags: [CREATE], [ADD], [STYLE], [UPDATE], [DELETE], [ROUTE], [START], [SERVICE]
    m_bracket = re.match(r"^\[([A-Z_]+)\]\s*(.*)", directive)
    if m_bracket:
        v_str = m_bracket.group(1).upper()
        payload = m_bracket.group(2).strip()
        v_map = {
            "CREATE": "create_table",
            "TABLE": "create_table",
            "FIELD": "add_field",
            "ADD": "add_function" if not any(k in payload.lower() for k in ("route", "api", "post", "get")) else "insert_statement",
            "ROUTE": "insert_statement",
            "STYLE": "replace_literal",
            "UPDATE": "replace_literal",
            "EDIT": "replace_literal",
            "DELETE": "delete_block",
            "DEL": "delete_block",
            "FIX": "replace_literal",
            "START": "start_service",
            "RUN": "start_service",
            "SERVICE": "start_service",
        }
        action = v_map.get(v_str, "insert_statement")
        return PlanStep(step_id=step_id, action=action, description=payload, target_hint=target)

    # 2. Prefix symbol tags: +create, +route, +field, +func, *style, ~update, -delete, +start, +service
    action = "add_function"
    payload = directive

    if directive.startswith(("+start", "+run", "+service")):
        action = "start_service"
        payload = re.sub(r"^\+(?:start|run|service)\s*", "", directive)
    elif directive.startswith(("+create", "+table")):
        action = "create_table"
        payload = re.sub(r"^\+(?:create|table)\s*", "", directive)
    elif directive.startswith(("+field", "+col")):
        action = "add_field"
        payload = re.sub(r"^\+(?:field|col)\s*", "", directive)
    elif directive.startswith(("+route", "+api")):
        action = "insert_statement"
        payload = re.sub(r"^\+(?:route|api)\s*", "", directive)
    elif directive.startswith(("+func", "+def", "+fn")):
        action = "add_function"
        payload = re.sub(r"^\+(?:func|def|fn)\s*", "", directive)
    elif directive.startswith(("*style", "*")):
        action = "replace_literal"
        payload = re.sub(r"^\*(?:style)?\s*", "", directive)
    elif directive.startswith(("~update", "~edit", "~")):
        action = "replace_literal"
        payload = re.sub(r"^~(?:update|edit)?\s*", "", directive)
    elif directive.startswith(("-delete", "-del", "-")):
        action = "delete_block"
        payload = re.sub(r"^-(?:delete|del)?\s*", "", directive)
    elif directive.startswith(("delete ", "del ", "remove ", "rm ")):
        action = "delete_block"
        payload = re.sub(r"^(?:delete|del|remove|rm)\s*", "", directive)
    elif directive.startswith(("!fix", "!")):
        action = "replace_literal"
        payload = re.sub(r"^!(?:fix)?\s*", "", directive)
    elif directive.startswith("+"):
        payload = directive[1:].strip()
        ext = target.split(".")[-1].lower() if "." in target else ""
        if ext in ("sql", "prisma"):
            action = "create_table"
        elif any(k in payload.lower() for k in ("route", "api", "post", "get", "handler")):
            action = "insert_statement"
        else:
            action = "add_function"
    else:
        # Default fallback inference from target extension
        ext = target.split(".")[-1].lower() if "." in target else ""
        if ext in ("sql", "prisma"):
            action = "create_table"
        elif ext in ("vue", "tsx", "jsx", "css", "scss"):
            action = "replace_literal"
        elif any(k in payload.lower() for k in ("route", "api", "handler", "server", "app")):
            action = "insert_statement"
        else:
            action = "add_function"

    return PlanStep(step_id=step_id, action=action, description=payload.strip(), target_hint=target)


def parse_act_plan(
    raw_input: Any,
    macro_intent: str = "Execute plan",
    root: Path | None = None,
) -> ActPlan:
    """Polymorphically parse any plan input format into a canonical ActPlan.
    
    Supports:
    1. Flat Dict: {"schema.sql": "+create users(id, pwd)", "server.py": "+route POST /logout"}
    2. Markdown Checklist: "- [ ] schema.sql: +create users(id, pwd)"
    3. String Array or Micro-DSL lines: ["schema.sql: +create ...", ...]
    4. Legacy Dict: {"plan_id": ..., "steps": [...]}
    5. Freeform Natural Language: "把右上角退出按钮调深点，在用户表加失效期"
    """
    if isinstance(raw_input, ActPlan):
        return raw_input

    # File path input
    if isinstance(raw_input, (str, Path)):
        p = Path(str(raw_input))
        if p.is_file():
            content = p.read_text(encoding="utf-8", errors="replace").strip()
            return parse_act_plan(content, macro_intent=macro_intent, root=root or p.parent)

    # Legacy nested dict format: {"plan_id": ..., "steps": [...]}
    if isinstance(raw_input, dict) and "steps" in raw_input and isinstance(raw_input["steps"], list):
        steps = []
        for s in raw_input["steps"]:
            steps.append(PlanStep(
                step_id=s.get("step_id", f"step_{len(steps)+1}"),
                action=s.get("action", "insert_statement"),
                description=s.get("description", ""),
                target_hint=s.get("target_hint", ""),
                code_snippet=s.get("code_snippet"),
            ))
        return ActPlan(
            plan_id=raw_input.get("plan_id", "legacy_plan"),
            macro_intent=raw_input.get("macro_intent", macro_intent),
            steps=steps,
        )

    # Flat dict format: {"target": "directive", ...}
    if isinstance(raw_input, dict):
        steps = []
        for idx, (target, directive) in enumerate(raw_input.items(), 1):
            steps.append(_parse_directive_to_step(f"step_{idx}", target, str(directive)))
        return ActPlan(plan_id="flat_dict_plan", macro_intent=macro_intent, steps=steps)

    # List of directives or sub-dicts
    if isinstance(raw_input, list):
        steps = []
        for idx, item in enumerate(raw_input, 1):
            if isinstance(item, dict):
                for tgt, direct in item.items():
                    steps.append(_parse_directive_to_step(f"step_{idx}", tgt, str(direct)))
            else:
                steps.append(_parse_directive_to_step(f"step_{idx}", "", str(item)))
        return ActPlan(plan_id="list_plan", macro_intent=macro_intent, steps=steps)

    # String input (Markdown, multi-line, JSON, or Natural Language)
    if isinstance(raw_input, str):
        text = raw_input.strip()
        # Try JSON decode first
        if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
            import json
            try:
                data = json.loads(text)
                return parse_act_plan(data, macro_intent=macro_intent, root=root)
            except json.JSONDecodeError:
                pass

        lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]

        def _is_structured_line(line: str) -> bool:
            import re
            cleaned = re.sub(r"^(?:[-*]|\d+\.)\s*(?:\[[ xX]\]\s*)?", "", line).strip()
            # 1. Bracket tag: [CREATE], [ADD], [ROUTE], [STYLE], etc.
            if re.match(r"^\[[A-Z_]+\]", cleaned):
                return True
            # 2. Verb prefixes: +create, ~update, *style, -delete, +route, +start, +service, etc.
            if re.match(r"^(?:\+(?:create|table|field|col|route|api|func|def|fn|start|run|service)|[~*!](?:update|edit|style|fix)?|-delete|-del)\b", cleaned, re.I):
                return True
            # 3. Path prefix with file extension: filename.ext: or path/to/file.ext:
            m = re.split(r"\s*(?::|->)\s*", cleaned, maxsplit=1)
            if len(m) == 2 and any(ext in m[0].lower() for ext in (".py", ".sql", ".js", ".ts", ".tsx", ".jsx", ".vue", ".go", ".rs", ".prisma", ".css", ".html", ".json", ".toml", ".yaml", ".yml")):
                return True
            return False

        is_structured = any(line.startswith(("- [ ]", "- [x]", "* [ ]", "* [x]")) or _is_structured_line(line) for line in lines)

        # Markdown checklist or multi-line directives
        if is_structured:
            steps = []
            for idx, line in enumerate(lines, 1):
                steps.append(_parse_directive_to_step(f"step_{idx}", "", line))
            return ActPlan(plan_id="structured_plan", macro_intent=macro_intent, steps=steps)

        # Freeform natural language query -> AI semantic grounding
        from sylphae.coprocessor.grounding import ground_natural_language_sync
        grounded_steps = ground_natural_language_sync(text, root=root)
        return ActPlan(plan_id="nl_grounded_plan", macro_intent=text, steps=grounded_steps)

    raise ValueError(f"Unsupported plan input format: {type(raw_input)}")

