"""Polyglot structured AST & style operations for real-world web and systems projects.

Supports:
- CSS/SCSS/PostCSS property modification & injection (nested selectors, variables, @apply)
- JSX / TSX / Vue / HTML class manipulation (className="...", className={cn(...)}, className={clsx(...)}, template literals)
- Polyglot constants, literals, and symbol manipulation (TypeScript, Go, Rust, Python, JSON, YAML)

This module is the **producer** side only: it turns a request into an
:class:`~sylphae.ops.spec.OpSpec`.  Applying a spec is
:func:`sylphae.ops.apply.apply_op` and nothing else -- there used to be a private
``apply_polyglot_patch`` here that re-implemented three of the appliers with
different semantics (a regex substitution for ``rename_symbol``, a silent
return-unchanged on a span mismatch, a substring fallback for
``replace_literal``); it had no callers and is gone, because two appliers for one
vocabulary is a contradiction and the quiet one was the dangerous one.
"""

from __future__ import annotations

import re

from sylphae.ops.spec import OpSpec

__all__ = [
    "replace_css_property",
    "modify_jsx_class",
    "add_css_properties",
    "replace_polyglot_constant",
]


def replace_css_property(
    src: str,
    prop_name: str,
    new_val: str,
    context_hint: str | None = None,
    line_hint: int | None = None,
    path: str = "inline",
) -> tuple[str, OpSpec | None]:
    """Replace or update a CSS/SCSS property, custom property, or Tailwind @apply declaration."""
    lines = src.splitlines(keepends=True)
    if not lines:
        return src, None

    # Pattern 1: Standard CSS property: prop_name \s*:\s* [^;]+ ;?
    prop_pattern = re.compile(
        rf"(\b{re.escape(prop_name)}\s*:\s*)([^;]+)(;?)",
        re.IGNORECASE,
    )

    # Pattern 2: Tailwind @apply line: @apply \s+ (.*)
    apply_pattern = re.compile(r"(@apply\s+)([^;]+)(;?)", re.IGNORECASE)

    target_idx: int | None = None
    is_apply_match: bool = False

    # 1. If line_hint is provided
    if line_hint is not None and 1 <= line_hint <= len(lines):
        candidate_line = lines[line_hint - 1]
        if prop_pattern.search(candidate_line):
            target_idx = line_hint - 1
        elif apply_pattern.search(candidate_line):
            target_idx = line_hint - 1
            is_apply_match = True

    # 2. Search around context hint (selector, class name, or rule block)
    if target_idx is None and context_hint:
        clean_hint = context_hint.strip(".#")
        for i, line in enumerate(lines):
            if clean_hint.lower() in line.lower():
                # Scan next 15 lines within the selector block
                for j in range(i, min(len(lines), i + 18)):
                    if prop_pattern.search(lines[j]):
                        target_idx = j
                        break
                    elif apply_pattern.search(lines[j]) and any(tok in new_val for tok in ("bg-", "text-", "p-", "rounded")):
                        target_idx = j
                        is_apply_match = True
                        break
                if target_idx is not None:
                    break

    # 3. Fallback scan across all lines
    if target_idx is None:
        for i, line in enumerate(lines):
            if prop_pattern.search(line):
                target_idx = i
                break

    if target_idx is None:
        return src, None

    orig_line = lines[target_idx]

    if is_apply_match:
        m = apply_pattern.search(orig_line)
        if not m:
            return src, None
        tokens = m.group(2).split()
        # Find matching utility class to replace with new_val
        prefix = new_val.split("-")[0] if "-" in new_val else new_val
        replaced = False
        new_tokens = []
        for t in tokens:
            if t.startswith(f"{prefix}-") or t == prefix:
                new_tokens.append(new_val)
                replaced = True
            else:
                new_tokens.append(t)
        if not replaced:
            new_tokens.append(new_val)
        semicolon = m.group(3) or ";"
        new_line = orig_line[:m.start(2)] + " ".join(new_tokens) + semicolon + orig_line[m.end(3):]
        old_val = m.group(2).strip()
    else:
        m = prop_pattern.search(orig_line)
        if not m:
            return src, None
        old_val = m.group(2).strip()
        semicolon = m.group(3) or ";"
        new_line = orig_line[:m.start(2)] + new_val + semicolon + orig_line[m.end(3):]

    new_lines = list(lines)
    new_lines[target_idx] = new_line

    spec = OpSpec(
        kind="replace_literal",
        path=path or "inline",
        span=(target_idx + 1, target_idx + 1),
        props={
            "old_text": orig_line,
            "new_text": new_line,
            "property": prop_name,
            "old": old_val,
            "new": new_val,
        },
    )
    return "".join(new_lines), spec


def modify_jsx_class(
    src: str,
    old_class_token: str,
    new_class_token: str,
    line_hint: int | None = None,
    context_hint: str | None = None,
    path: str = "inline",
) -> tuple[str, OpSpec | None]:
    """Modify or swap a CSS / Tailwind class in JSX/TSX/Vue attributes.

    Supports:
    - className="..." and class="..."
    - className={cn("...", ...)} and className={clsx("...", ...)}
    - className={`...`} (template literals)
    - Vue :class="..." and @apply in style blocks
    """
    lines = src.splitlines(keepends=True)
    if not lines:
        return src, None

    # Flexible pattern for class container
    patterns = [
        # standard className="..." or class="..."
        re.compile(r'((?:class(?:Name)?)=["\'])([^"\']+)(["\'])'),
        # template string: className={`...`}
        re.compile(r'((?:class(?:Name)?)=\{\s*`)([^`]+)(`\s*\})'),
        # cn("...", ...) or clsx("...", ...)
        re.compile(r'((?:cn|clsx)\s*\([^)]*["\'])([^"\']*?' + re.escape(old_class_token) + r'[^"\']*?)(["\'])'),
    ]

    target_idx: int | None = None
    matched_pat = None

    # 1. line_hint
    if line_hint is not None and 1 <= line_hint <= len(lines):
        line = lines[line_hint - 1]
        if old_class_token in line:
            for pat in patterns:
                if pat.search(line):
                    target_idx = line_hint - 1
                    matched_pat = pat
                    break

    # 2. context_hint
    if target_idx is None and context_hint:
        for i, line in enumerate(lines):
            if context_hint.lower() in line.lower() and old_class_token in line:
                for pat in patterns:
                    if pat.search(line):
                        target_idx = i
                        matched_pat = pat
                        break
                if target_idx is not None:
                    break

    # 3. Scan all lines for token
    if target_idx is None:
        for i, line in enumerate(lines):
            if old_class_token in line:
                for pat in patterns:
                    if pat.search(line):
                        target_idx = i
                        matched_pat = pat
                        break
                if target_idx is not None:
                    break

    # 4. Fallback: simple token substring replacement on any line containing old_class_token
    if target_idx is None:
        for i, line in enumerate(lines):
            if old_class_token in line and any(k in line for k in ("class", "className", "cn(", "clsx(")):
                target_idx = i
                break

    if target_idx is None:
        return src, None

    orig_line = lines[target_idx]

    # Perform token replacement
    if old_class_token in orig_line:
        # Replace word boundary if possible, else substring
        boundary_pat = re.compile(rf"(?<=\s|[\"\'`]){re.escape(old_class_token)}(?=\s|[\"\'`]|$)")
        if boundary_pat.search(orig_line):
            new_line = boundary_pat.sub(new_class_token, orig_line, count=1)
        else:
            new_line = orig_line.replace(old_class_token, new_class_token, 1)
    else:
        # Append class inside the first quotes
        m_quote = re.search(r'(class(?:Name)?=["\'])([^"\']+)(["\'])', orig_line)
        if m_quote:
            new_line = orig_line[:m_quote.end(2)] + f" {new_class_token}" + orig_line[m_quote.end(2):]
        else:
            return src, None

    new_lines = list(lines)
    new_lines[target_idx] = new_line

    spec = OpSpec(
        kind="replace_literal",
        path=path or "inline",
        span=(target_idx + 1, target_idx + 1),
        props={
            "old_text": orig_line,
            "new_text": new_line,
            "old": old_class_token,
            "new": new_class_token,
        },
    )
    return "".join(new_lines), spec


def add_css_properties(
    src: str,
    selector_or_hint: str,
    properties: dict[str, str],
    path: str = "inline",
) -> tuple[str, OpSpec | None]:
    """Inject CSS properties into an existing CSS rule block or create a new rule block."""
    lines = src.splitlines(keepends=True)
    clean_hint = selector_or_hint.strip(".#")

    # Find the rule block matching selector
    rule_start: int | None = None
    rule_end: int | None = None

    for i, line in enumerate(lines):
        if clean_hint.lower() in line.lower() and "{" in line:
            rule_start = i
            for j in range(i, min(len(lines), i + 40)):
                if "}" in lines[j]:
                    rule_end = j
                    break
            break

    if rule_start is not None and rule_end is not None:
        closing_line = lines[rule_end]
        indent = "  "
        if rule_end > rule_start:
            prev_line = lines[rule_end - 1]
            indent_m = re.match(r"^(\s+)", prev_line)
            if indent_m:
                indent = indent_m.group(1)

        injected = "".join(f"{indent}{k}: {v};\n" for k, v in properties.items())
        new_closing = injected + closing_line
        orig_closing = closing_line

        new_lines = list(lines)
        new_lines[rule_end] = new_closing

        spec = OpSpec(
            kind="insert_statement",
            path=path or "inline",
            span=(rule_end + 1, rule_end + 1),
            props={
                "old_text": orig_closing,
                "new_text": new_closing,
                "text": injected,
                "line": str(rule_end + 1),
            },
        )
        return "".join(new_lines), spec

    # If no rule block found, append a new rule block
    ending = "\n" if src.endswith("\n") else "\n\n"
    new_block = f".{clean_hint} {{\n" + "".join(f"  {k}: {v};\n" for k, v in properties.items()) + "}\n"
    new_src = src + ending + new_block
    spec = OpSpec(
        kind="insert_statement",
        path=path or "inline",
        span=(len(lines) + 1, len(lines) + 1),
        props={
            "old_text": "",
            "new_text": new_block,
            "text": new_block,
            "line": str(len(lines) + 1),
        },
    )
    return new_src, spec


def replace_polyglot_constant(
    src: str,
    const_name: str,
    new_val: str,
    path: str = "inline",
) -> tuple[str, OpSpec | None]:
    """Replace a numeric or string constant across Go, Rust, TS/JS, Python, or JSON."""
    lines = src.splitlines(keepends=True)
    if not lines:
        return src, None

    # Multi-language constant patterns:
    # Go: const Timeout = 30 * time.Second / Timeout = 30
    # Rust: const TIMEOUT: u64 = 30; / pub const TIMEOUT = 30;
    # TS/JS: export const TIMEOUT = 30; / const TIMEOUT = 30;
    # Python: TIMEOUT = 30 / TIMEOUT: int = 30
    # JSON: "timeout": 30
    patterns = [
        re.compile(rf"(\b(?:const|let|var)\s+{re.escape(const_name)}\b[^=]*=\s*)([^;\n]+)(;?)"),
        re.compile(rf"(\b{re.escape(const_name)}\b\s*[:=]\s*)([^;\n,]+)([;,\n]?)"),
        re.compile(rf"(\"{re.escape(const_name)}\"\s*:\s*)([^,\n]+)(,?)"),
    ]

    target_idx: int | None = None
    matched_m = None

    for i, line in enumerate(lines):
        for pat in patterns:
            m = pat.search(line)
            if m:
                target_idx = i
                matched_m = m
                break
        if target_idx is not None:
            break

    if target_idx is None or matched_m is None:
        return src, None

    orig_line = lines[target_idx]
    old_val = matched_m.group(2).strip()
    trailing = matched_m.group(3) or ""
    new_line = orig_line[:matched_m.start(2)] + str(new_val) + trailing + orig_line[matched_m.end(3):]

    new_lines = list(lines)
    new_lines[target_idx] = new_line

    spec = OpSpec(
        kind="replace_literal",
        path=path or "inline",
        span=(target_idx + 1, target_idx + 1),
        props={
            "old_text": orig_line,
            "new_text": new_line,
            "old": old_val,
            "new": str(new_val),
            "identifier": const_name,
        },
    )
    return "".join(new_lines), spec
