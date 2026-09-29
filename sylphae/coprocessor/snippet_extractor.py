"""High-density syntactic snippet extraction for polyglot codebases.

Extracts candidate blocks (CSS rules, Vue/TSX components, AST declarations)
with rich contextual metadata (theme tags, visual containment ratios, container types).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sylphae.agent.explorer import extract_keywords
from sylphae.agent.spatial import resolve_spatial_anchors


@dataclass
class RawCandidateSnippet:
    file_path: str
    start_line: int
    end_line: int
    code_snippet: str
    target_entity: str          # e.g. 'navbar', 'button', 'user_table'
    kind: str                   # 'base_style', 'theme_override', 'visual_dominant', 'nested_control', 'declaration'
    selector_or_tag: str        # e.g. '.navbar', '<nav>', '[data-theme="dark"] .navbar'
    theme_name: str | None = None  # e.g. 'default', 'dark', 'neon'
    is_default_theme: bool = False
    is_visual_dominant: bool = False
    is_container_root: bool = False
    color_or_value: str = ""
    classes_or_props: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def extract_candidate_snippets(
    root: Path,
    candidate_files: list[str],
    query: str,
    max_candidates: int = 12,
) -> list[RawCandidateSnippet]:
    """Extract syntactically coherent candidate snippets for a developer query."""
    root = Path(root).resolve()
    query_lower = query.lower()
    keywords = extract_keywords(query)
    spatial = resolve_spatial_anchors(query)

    # Core concept detection
    is_color_query = any(k in query_lower for k in ("颜色", "底色", "背景", "深", "浅", "color", "bg", "background", "theme"))
    is_nav_query = any(k in query_lower for k in ("导航", "导航栏", "页眉", "header", "navbar", "nav", "topbar"))
    is_btn_query = any(k in query_lower for k in ("按钮", "btn", "button"))

    # Theme intent detection in query
    requested_theme: str | None = None
    if any(k in query_lower for k in ("暗黑", "深色主题", "黑夜", "dark", "night")):
        requested_theme = "dark"
    elif any(k in query_lower for k in ("明亮", "浅色主题", "白天", "light", "day")):
        requested_theme = "light"
    elif any(k in query_lower for k in ("默认", "default")):
        requested_theme = "default"
    else:
        # Check custom named theme pattern: <name>主题 or theme-<name>
        m_theme = re.search(r"([a-zA-Z0-9_\-]+)\s*(?:主题|theme)", query_lower)
        if m_theme:
            t_cand = m_theme.group(1).lower()
            if t_cand not in ("默认", "系统", "当前", "这个"):
                requested_theme = t_cand

    candidates: list[RawCandidateSnippet] = []

    for rel_path in candidate_files:
        full_path = root / rel_path
        if not full_path.is_file():
            continue
        try:
            st = full_path.stat()
            if st.st_size > 512 * 1024 or st.st_size == 0:
                continue
            text = full_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        ext = Path(rel_path).suffix.lower()
        lines = text.splitlines()

        # 1. CSS / SCSS / LESS files
        if ext in (".css", ".scss", ".less"):
            _extract_css_snippets(
                rel_path, text, lines, query_lower, requested_theme, is_nav_query, candidates
            )

        # 2. Vue Single File Components (.vue)
        elif ext == ".vue":
            # Check <style> blocks in Vue
            _extract_css_snippets(
                rel_path, text, lines, query_lower, requested_theme, is_nav_query, candidates
            )
            # Check <template> HTML / Tailwind in Vue
            _extract_template_snippets(
                rel_path, text, lines, query_lower, requested_theme, is_nav_query, candidates
            )

        # 3. TSX / JSX / HTML files
        elif ext in (".tsx", ".jsx", ".html"):
            _extract_template_snippets(
                rel_path, text, lines, query_lower, requested_theme, is_nav_query, candidates
            )

        # 4. Python / Go / Rust / SQL files
        else:
            _extract_code_decl_snippets(
                rel_path, text, lines, query_lower, keywords, candidates
            )

        if len(candidates) >= max_candidates * 2:
            break

    return candidates[:max_candidates]


def _extract_css_snippets(
    rel_path: str,
    text: str,
    lines: list[str],
    query_lower: str,
    requested_theme: str | None,
    is_nav_query: bool,
    out: list[RawCandidateSnippet],
) -> None:
    """Extract CSS rulesets with theme metadata."""
    # Match CSS ruleset: selector { ... }
    for m in re.finditer(r"([^{}\n][^{}]+?)\s*\{([^}]+)\}", text):
        raw_selector = m.group(1).strip()
        body = m.group(2).strip()
        sel_lower = raw_selector.lower()
        body_lower = body.lower()

        # Check if rule touches relevant concepts
        has_color = any(k in body_lower for k in ("color", "background", "bg", "fill"))
        has_nav = any(k in sel_lower for k in ("navbar", "nav", "header", "topbar"))

        if is_nav_query and not has_nav and not any(k in rel_path.lower() for k in ("nav", "header", "theme")):
            continue

        if not has_color and not any(k in query_lower for k in ("height", "width", "size", "font")):
            continue

        start_line = text[:m.start()].count("\n") + 1
        end_line = text[:m.end()].count("\n") + 1
        rule_snippet = text[m.start():m.end()].strip()

        # Classify theme characteristics
        theme_name = None
        is_default = False
        is_theme_override = False

        if "theme" in sel_lower or "theme" in rel_path.lower() or "dark" in sel_lower:
            is_theme_override = True
            if "default" in sel_lower or "theme.css" in rel_path.lower():
                theme_name = "default"
                is_default = True
            elif "dark" in sel_lower or "night" in sel_lower:
                theme_name = "dark"
            elif "light" in sel_lower:
                theme_name = "light"
            else:
                m_t = re.search(r"(?:data-theme=[\"']?|theme-)([a-zA-Z0-9_\-]+)", sel_lower)
                theme_name = m_t.group(1) if m_t else "custom"

        # Check if visual dominant / container
        is_dominant = "100%" in body_lower or "cover" in body_lower or "absolute" in body_lower
        is_root = not is_theme_override and any(k in sel_lower for k in (".navbar", ".nav", ".header", "nav", "header"))

        kind = "theme_override" if is_theme_override else ("visual_dominant" if is_dominant and not is_root else "base_style")

        out.append(RawCandidateSnippet(
            file_path=rel_path,
            start_line=start_line,
            end_line=end_line,
            code_snippet=rule_snippet,
            target_entity="navbar" if has_nav else "component",
            kind=kind,
            selector_or_tag=raw_selector,
            theme_name=theme_name,
            is_default_theme=is_default,
            is_visual_dominant=is_dominant,
            is_container_root=is_root,
            color_or_value=body.strip(),
            classes_or_props=[raw_selector],
        ))


def _extract_template_snippets(
    rel_path: str,
    text: str,
    lines: list[str],
    query_lower: str,
    requested_theme: str | None,
    is_nav_query: bool,
    out: list[RawCandidateSnippet],
) -> None:
    """Extract HTML/JSX/Vue elements with visual dominance & containment analysis."""
    # Pattern matching opening tags with class attributes
    pattern = r"<([a-zA-Z0-9_\-]+)([^>]*?(?:class|className)=[\"'{]([^>\"'}]+)[\"'\}][^>]*?)(\/?>)"
    for m in re.finditer(pattern, text, re.DOTALL):
        tag_name = m.group(1).lower()
        attrs = m.group(2)
        classes_str = m.group(3)
        classes_lower = classes_str.lower()

        # Tag relevance
        is_nav_tag = tag_name in ("nav", "header") or any(k in classes_lower for k in ("navbar", "nav-bar", "header", "topbar"))
        is_card_or_container = tag_name in ("div", "section", "article", "main") or any(k in classes_lower for k in ("card", "container", "panel", "box", "wrapper"))
        is_btn = tag_name in ("button", "a") or any(k in classes_lower for k in ("btn", "button"))

        if is_nav_query and not (is_nav_tag or is_card_or_container or is_btn):
            continue

        # Check visual dominance: elements that occupy 100% width and height of container
        has_w_full = any(k in classes_lower for k in ("w-full", "w-screen", "width: 100%", "w-[100%]"))
        has_h_full = any(k in classes_lower for k in ("h-full", "min-h-full", "height: 100%", "h-[100%]"))
        has_inset = any(k in classes_lower for k in ("inset-0", "absolute inset-0", "fixed inset-0"))
        has_flex_fill = "flex-1" in classes_lower

        # Visual dominant: card/container inside nav that almost completely covers the navbar
        is_visual_dominant = (has_w_full and has_h_full) or has_inset or (has_w_full and has_flex_fill and is_card_or_container)

        # Check color / style presence
        color_match = re.findall(r"\b(?:bg-[a-zA-Z0-9_\-\[\]#]+|text-[a-zA-Z0-9_\-]+)\b", classes_str)

        start_line = text[:m.start()].count("\n") + 1
        end_line = text[:m.end()].count("\n") + 1
        tag_snippet = text[m.start():m.end()].strip()

        # Theme context in template
        theme_name = None
        is_default = False
        if any(k in classes_lower for k in ("dark:", "dark", "theme-dark")):
            theme_name = "dark"
        elif any(k in classes_lower for k in ("default", "theme-default")):
            theme_name = "default"
            is_default = True

        is_root = is_nav_tag and not is_visual_dominant

        if is_visual_dominant:
            kind = "visual_dominant"
        elif theme_name:
            kind = "theme_override"
        elif is_root:
            kind = "base_style"
        else:
            kind = "nested_control"

        out.append(RawCandidateSnippet(
            file_path=rel_path,
            start_line=start_line,
            end_line=end_line,
            code_snippet=tag_snippet,
            target_entity="navbar" if is_nav_query else tag_name,
            kind=kind,
            selector_or_tag=f"<{tag_name}>",
            theme_name=theme_name,
            is_default_theme=is_default,
            is_visual_dominant=is_visual_dominant,
            is_container_root=is_root,
            color_or_value=" ".join(color_match),
            classes_or_props=classes_str.split(),
            metadata={"has_w_full": has_w_full, "has_h_full": has_h_full, "is_nav_tag": is_nav_tag},
        ))


def _extract_code_decl_snippets(
    rel_path: str,
    text: str,
    lines: list[str],
    query_lower: str,
    keywords: list[str],
    out: list[RawCandidateSnippet],
) -> None:
    """Extract code declarations (functions, models, config constants)."""
    for i, line in enumerate(lines, 1):
        line_l = line.lower()
        if any(kw in line_l for kw in keywords if len(kw) >= 3):
            # Capture a 3-line contextual span
            start = max(1, i - 1)
            end = min(len(lines), i + 2)
            snippet = "\n".join(lines[start - 1:end])
            out.append(RawCandidateSnippet(
                file_path=rel_path,
                start_line=start,
                end_line=end,
                code_snippet=snippet,
                target_entity="code_declaration",
                kind="declaration",
                selector_or_tag=line.strip().split()[0] if line.strip() else "",
            ))
            if len(out) >= 8:
                return
