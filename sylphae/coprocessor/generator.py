"""AI-augmented language-agnostic schema and code generator.

Translates high-level plan step descriptions (e.g. '新增一个sqlite用户表，拥有user_id，password项',
'add postgres users table with id, email, hashed_password') into universal TableSpec ASTs and renders
them across SQL (SQLite, Postgres, MySQL), ORMs (SQLAlchemy, Prisma), and languages (Go, Rust, TypeScript).
"""

from __future__ import annotations

import re

from sylphae.coprocessor.protocol import ColumnSpec, TableSpec

__all__ = [
    "parse_table_description",
    "render_table_schema",
    "synthesize_code_snippet",
]


def parse_table_description(description: str, default_dialect: str = "sqlite") -> TableSpec:
    """Parse a natural language or structured plan description into a TableSpec AST using AI SemanticGrounder."""
    from sylphae.model.semantic_grounder import get_semantic_grounder
    grounder = get_semantic_grounder()
    lower = description.lower()

    # 1. Detect dialect
    dialect = default_dialect
    if any(k in lower for k in ("sqlite", "sqlite3")):
        dialect = "sqlite"
    elif any(k in lower for k in ("postgres", "postgresql", "pg")):
        dialect = "postgres"
    elif "mysql" in lower:
        dialect = "mysql"
    elif "prisma" in lower:
        dialect = "prisma"
    elif "sqlalchemy" in lower:
        dialect = "sqlalchemy"
    elif "gorm" in lower:
        dialect = "go"
    elif "diesel" in lower or "rust" in lower:
        dialect = "rust"

    # 2. Extract table name using AI SemanticGrounder
    paren_cols = None
    m_paren = re.search(r"\b([a-zA-Z0-9_]+)\s*\(([^)]*)\)", description)
    if m_paren and m_paren.group(1).lower() not in ("create", "add", "table", "new", "model", "insert"):
        table_name = m_paren.group(1)
        raw_cols = [c.strip().split()[0] for c in m_paren.group(2).split(",") if c.strip()]
        if raw_cols:
            paren_cols = raw_cols
    else:
        # AI semantic entity extraction (maps concepts like 用户/会员/客户 -> users, 订单/流水 -> orders, etc.)
        ent = grounder.extract_semantic_entity(description, default_entity="users")
        table_name = ent.normalized_name

    # 3. Extract column specifications
    if paren_cols:
        col_names = paren_cols
    else:
        # Extract items following delimiter / listing patterns or field tokens
        m_cols = re.search(r"(?:拥有|包含|字段|项|columns|fields|with|having)\s*[:：]?\s*([a-zA-Z0-9_,\s，]+)", description, re.I)
        if m_cols:
            raw_items = re.split(r"[,，\s]+", m_cols.group(1).strip())
            col_names = [it.strip() for it in raw_items if it.strip() and it.strip().lower() not in ("项", "字段", "等", "a", "an", "the", "columns", "fields")]
        else:
            col_names = re.findall(r"\b[a-zA-Z0-9_]+(?:_id|id|password|email|name|hash|token|time|at|status)\b", description, re.I)

    if not col_names:
        col_names = ["user_id", "password"]

    columns: list[ColumnSpec] = []
    for i, col in enumerate(col_names):
        col_lower = col.lower()
        is_pk = (i == 0) or ("id" in col_lower and ("user" in col_lower or "pk" in col_lower))
        inferred_type = grounder.infer_column_type(col, dialect=dialect)
        not_null = is_pk or inferred_type in ("VARCHAR(64)", "TIMESTAMP") or any(k in col_lower for k in ("password", "email", "token"))

        columns.append(ColumnSpec(
            name=col,
            col_type=inferred_type,
            primary_key=is_pk,
            not_null=not_null,
        ))

    return TableSpec(name=table_name, columns=columns, dialect=dialect)


def render_table_schema(spec: TableSpec, target_dialect: str | None = None) -> str:
    """Render a TableSpec into target dialect syntax."""
    dialect = target_dialect or spec.dialect

    if dialect in ("sqlite", "sql"):
        col_defs = []
        for c in spec.columns:
            parts = [f"    {c.name} {c.col_type}"]
            if c.primary_key:
                parts.append("PRIMARY KEY")
            elif c.not_null:
                parts.append("NOT NULL")
            if c.default:
                parts.append(f"DEFAULT {c.default}")
            col_defs.append(" ".join(parts))
        return f"CREATE TABLE IF NOT EXISTS {spec.name} (\n" + ",\n".join(col_defs) + "\n);\n"

    elif dialect == "postgres":
        col_defs = []
        for c in spec.columns:
            t = "VARCHAR(255)" if c.col_type == "TEXT" else c.col_type
            parts = [f"    {c.name} {t}"]
            if c.primary_key:
                parts.append("PRIMARY KEY")
            elif c.not_null:
                parts.append("NOT NULL")
            col_defs.append(" ".join(parts))
        return f"CREATE TABLE IF NOT EXISTS {spec.name} (\n" + ",\n".join(col_defs) + "\n);\n"

    elif dialect == "sqlalchemy":
        lines = [
            f"class {spec.name.capitalize()}(Base):",
            f'    __tablename__ = "{spec.name}"',
            "",
        ]
        for c in spec.columns:
            t = "String" if c.col_type.startswith(("TEXT", "VARCHAR")) else "Integer"
            opts = []
            if c.primary_key:
                opts.append("primary_key=True")
            if c.not_null and not c.primary_key:
                opts.append("nullable=False")
            opts_str = f", {', '.join(opts)}" if opts else ""
            lines.append(f"    {c.name} = Column({t}{opts_str})")
        return "\n".join(lines) + "\n"

    elif dialect == "prisma":
        lines = [f"model {spec.name.capitalize()} {{"]
        for c in spec.columns:
            t = "String" if c.col_type.startswith(("TEXT", "VARCHAR")) else "Int"
            attr = " @id" if c.primary_key else ""
            lines.append(f"  {c.name.ljust(12)} {t}{attr}")
        lines.append("}\n")
        return "\n".join(lines)

    elif dialect == "go":
        lines = [f"type {spec.name.capitalize()} struct {{"]
        for c in spec.columns:
            field_name = "".join(part.capitalize() for part in c.name.split("_"))
            tag = f'`json:"{c.name}"'
            if c.primary_key:
                tag += ' gorm:"primaryKey"'
            tag += '`'
            lines.append(f"\t{field_name.ljust(12)} string {tag}")
        lines.append("}\n")
        return "\n".join(lines)

    elif dialect == "rust":
        lines = [
            "#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]",
            f"pub struct {spec.name.capitalize()} {{",
        ]
        for c in spec.columns:
            lines.append(f"    pub {c.name}: String,")
        lines.append("}\n")
        return "\n".join(lines)

    # Universal fallback: Standard SQL
    return render_table_schema(spec, target_dialect="sqlite")


def synthesize_code_snippet(description: str, target_lang: str = "python") -> str:
    """Synthesize language-idiomatic helper function code from natural language description."""
    lower = description.lower()

    # 1. Password hashing / verification utility
    if any(k in lower for k in ("hash", "password", "密码", "哈希", "verify")):
        if target_lang in ("python", "py"):
            return """def hash_password(password: str) -> str:
    \"\"\"Secure SHA-256 password hash helper.\"\"\"
    import hashlib
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    \"\"\"Verify password against hash.\"\"\"
    return hash_password(password) == hashed
"""
        elif target_lang in ("go", "golang"):
            return """func HashPassword(password string) string {
\th := sha256.Sum256([]byte(password))
\treturn hex.EncodeToString(h[:])
}

func VerifyPassword(password, hashed string) bool {
\treturn HashPassword(password) == hashed
}
"""
        elif target_lang in ("typescript", "ts", "javascript", "js"):
            return """export function hashPassword(password: string): string {
  const crypto = require('crypto');
  return crypto.createHash('sha256').update(password).digest('hex');
}

export function verifyPassword(password: string, hashed: string): boolean {
  return hashPassword(password) === hashed;
}
"""

    # 2. Route handler synthesis (dynamic method & path)
    if any(k in lower for k in ("route", "endpoint", "api", "post", "get", "put", "delete", "login", "logout", "register", "auth", "注销", "登录", "注册", "接口")):
        method = "post"
        for m in ("get", "post", "put", "delete", "patch"):
            if re.search(rf"\b{m}\b", lower):
                method = m
                break

        m_path = re.search(r"(/[a-zA-Z0-9_\-/{}:]+)", description)
        route_path = m_path.group(1) if m_path else ("/auth/logout" if any(k in lower for k in ("logout", "注销", "signout")) else "/auth/login")
        path_parts = [p for p in route_path.split("/") if p and not p.startswith(":") and not p.startswith("{")]
        func_name = path_parts[-1] if path_parts else "handle_request"
        func_name = re.sub(r"[^a-zA-Z0-9_]", "_", func_name)

        if target_lang in ("python", "py"):
            if "logout" in func_name or "注销" in lower:
                return f"""@app.{method}("{route_path}")
async def {func_name}():
    \"\"\"User logout endpoint.\"\"\"
    return {{"status": "success", "message": "logged out"}}
"""
            else:
                return f"""@app.{method}("{route_path}")
async def {func_name}(username: str = "", password: str = ""):
    \"\"\"Endpoint handler for {route_path}.\"\"\"
    return {{"status": "success", "user": username}}
"""
        elif target_lang in ("go", "golang"):
            handler_cap = "".join(p.capitalize() for p in func_name.split("_")) + "Handler"
            return f"""func {handler_cap}(w http.ResponseWriter, r *http.Request) {{
\tw.Header().Set("Content-Type", "application/json")
\tw.Write([]byte(`{{"status":"success"}}`))
}}
"""
        elif target_lang in ("typescript", "ts"):
            return f"""export async function {method.upper()}(request: Request) {{
  return Response.json({{ status: "success", route: "{route_path}" }});
}}
"""

    return f"# Implementation for: {description}\n"


def parse_field_spec(description: str, default_table: str = "users") -> tuple[str, ColumnSpec]:
    """Extract table name and ColumnSpec for an add_field operation using AI SemanticGrounder."""
    from sylphae.model.semantic_grounder import get_semantic_grounder
    grounder = get_semantic_grounder()

    # Syntax 1: table.field:type (e.g. users.expire_at:timestamp or expire_at:text)
    m_dot = re.search(r"\b([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)(?::([a-zA-Z0-9_]+))?", description)
    if m_dot:
        table_name = m_dot.group(1)
        col_name = m_dot.group(2)
        col_type = m_dot.group(3).upper() if m_dot.group(3) else grounder.infer_column_type(col_name)
        return table_name, ColumnSpec(name=col_name, col_type=col_type)

    # Syntax 2: add <field> to <table>
    m_add_to = re.search(r"add\s+([a-zA-Z0-9_]+)\s+to\s+([a-zA-Z0-9_]+)", description, re.I)
    if m_add_to:
        col_name = m_add_to.group(1)
        table_name = m_add_to.group(2)
        col_type = grounder.infer_column_type(col_name)
        return table_name, ColumnSpec(name=col_name, col_type=col_type)

    # AI Semantic Entity Extraction for table name
    ent = grounder.extract_semantic_entity(description, default_entity=default_table)
    table_name = ent.normalized_name

    # AI Semantic Field Extraction (explicit identifier or conceptual prototype match)
    col_name = grounder.extract_semantic_field(description, default_field="expire_at")
    col_type = grounder.infer_column_type(col_name)
    return table_name, ColumnSpec(name=col_name, col_type=col_type)


def render_field_addition(table_name: str, col: ColumnSpec, target_dialect: str = "sqlite") -> str:
    """Render an incremental field addition across SQL dialects or model languages."""
    dialect = target_dialect.lower()
    if dialect in ("sqlite", "sql", "postgres", "mysql"):
        return f"ALTER TABLE {table_name} ADD COLUMN {col.name} {col.col_type};\n"
    elif dialect == "prisma":
        return f"  {col.name.ljust(12)} String?\n"
    elif dialect == "go":
        field_name = "".join(p.capitalize() for p in col.name.split("_"))
        return f"\t{field_name.ljust(12)} string `json:\"{col.name},omitempty\"`\n"
    elif dialect == "rust":
        return f"    pub {col.name}: Option<String>,\n"
    elif dialect in ("lua", "config"):
        return f"    {col.name} = nil,\n"
    return f"-- Add {col.name} to {table_name}\n"

