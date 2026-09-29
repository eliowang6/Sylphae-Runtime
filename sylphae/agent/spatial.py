"""Spatial and UI layout anchor reasoning from developer prompts.

Translates intuitive spatial concepts (e.g. '右上角', '按钮', '顶部导航', 'search bar in header') into
concrete code search terms, CSS selectors, and JSX component anchors with bilingual support.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "SpatialTarget",
    "resolve_spatial_anchors",
    "SPATIAL_KEYWORDS",
]

# Spatial mappings to code keywords, CSS classes and layout identifiers
SPATIAL_KEYWORDS: dict[str, list[str]] = {
    # Positions
    "右上角": [
        "top-right", "header-actions", "navbar-end", "navbar", "float-right",
        "justify-end", "align-right", "header", "topbar", "nav",
        "action-button", "right", "top",
    ],
    "左上角": [
        "top-left", "navbar-start", "brand", "logo", "header",
        "justify-start", "left", "top",
    ],
    "右下角": [
        "bottom-right", "float-right", "justify-end", "footer", "bottom",
    ],
    "左下角": [
        "bottom-left", "justify-start", "footer", "bottom",
    ],
    "顶部": [
        "header", "topbar", "navbar", "nav", "top", "app-bar",
    ],
    "页眉": [
        "header", "topbar", "nav", "masthead",
    ],
    "底部": [
        "footer", "bottom", "bottom-bar", "dock",
    ],
    "页脚": [
        "footer", "colophon", "bottom",
    ],
    "左侧": [
        "sidebar", "left", "drawer", "aside", "nav-sidebar",
    ],
    "右侧": [
        "sidebar-right", "right", "aside", "actions", "panel-right",
    ],
    "居中": [
        "center", "mx-auto", "justify-center", "items-center", "text-center",
    ],
    "首屏": [
        "hero", "banner", "jumbotron", "landing", "featured",
    ],

    # Components
    "按钮": [
        "button", "btn", "<Button", "Button", "btn-primary", "onClick", "action",
    ],
    "导航栏": [
        "navbar", "nav", "navigation", "topbar", "menu",
    ],
    "卡片": [
        "card", "panel", "box", "container", "tile", "widget",
    ],
    "输入框": [
        "input", "textbox", "search", "field", "textarea", "form-control",
    ],
    "搜索框": [
        "search", "search-input", "searchInput", "search-bar", "filter",
    ],
    "标题": [
        "title", "heading", "h1", "h2", "h3", "header-text",
    ],
    "弹窗": [
        "modal", "dialog", "popup", "drawer", "sheet", "overlay",
    ],
    "标签": [
        "label", "badge", "tag", "chip", "pill",
    ],
    "下拉框": [
        "dropdown", "select", "menu", "combobox",
    ],
    "头像": [
        "avatar", "profile-pic", "user-icon", "user-image",
    ],
    "表格": [
        "table", "grid", "data-table", "row", "cell", "tbody",
    ],
    "图标": [
        "icon", "svg", "lucide", "fa-", "heroicon",
    ],
    "选项卡": [
        "tab", "tabs", "tab-list", "nav-tabs",
    ],
}


@dataclass
class SpatialTarget:
    original_prompt: str
    spatial_position: str | None = None
    component_type: str | None = None
    search_keywords: list[str] = field(default_factory=list)
    css_selector_hints: list[str] = field(default_factory=list)


def resolve_spatial_anchors(prompt: str) -> SpatialTarget:
    """Analyze a natural language prompt and extract spatial and UI component targets."""
    target = SpatialTarget(original_prompt=prompt)
    lower = prompt.lower()

    # 1. Detect spatial position (Chinese & English)
    pos_mappings = [
        ("右上角", ["右上角", "右上", "top-right", "top right", "upper right"]),
        ("左上角", ["左上角", "左上", "top-left", "top left", "upper left"]),
        ("右下角", ["右下角", "右下", "bottom-right", "bottom right", "lower right"]),
        ("左下角", ["左下角", "左下", "bottom-left", "bottom left", "lower left"]),
        ("顶部", ["顶部", "页眉", "顶栏", "top", "header", "topbar"]),
        ("底部", ["底部", "页脚", "底栏", "bottom", "footer"]),
        ("左侧", ["左侧", "侧边栏", "左栏", "left", "sidebar"]),
        ("右侧", ["右侧", "右栏", "right"]),
        ("居中", ["居中", "中间", "center", "middle", "centered"]),
        ("首屏", ["首屏", "横幅", "hero", "banner"]),
    ]

    for canon_pos, aliases in pos_mappings:
        if any(alias in lower or alias in prompt for alias in aliases):
            target.spatial_position = canon_pos
            for kw in SPATIAL_KEYWORDS.get(canon_pos, []):
                target.search_keywords.append(kw)
            break

    # 2. Detect component type (Chinese & English)
    comp_mappings = [
        ("按钮", ["按钮", "按键", "button", "btn"]),
        ("搜索框", ["搜索框", "搜索栏", "search bar", "search input", "search"]),
        ("输入框", ["输入框", "文本框", "input", "textbox", "field", "textarea"]),
        ("导航栏", ["导航栏", "导航", "navbar", "nav", "navigation"]),
        ("卡片", ["卡片", "面板", "card", "panel", "box"]),
        ("标题", ["标题", "title", "heading", "h1", "h2", "h3"]),
        ("弹窗", ["弹窗", "对话框", "modal", "dialog", "popup"]),
        ("标签", ["标签", "徽章", "tag", "badge", "label", "chip"]),
        ("下拉框", ["下拉框", "选择器", "dropdown", "select"]),
        ("头像", ["头像", "avatar", "profile"]),
        ("表格", ["表格", "列表", "table", "grid", "list"]),
        ("图标", ["图标", "icon", "svg"]),
        ("选项卡", ["选项卡", "标签页", "tab", "tabs"]),
    ]

    for canon_comp, aliases in comp_mappings:
        if any(alias in lower or alias in prompt for alias in aliases):
            target.component_type = canon_comp
            for kw in SPATIAL_KEYWORDS.get(canon_comp, []):
                target.search_keywords.append(kw)
            break

    # 3. Dynamic regex-based noun phrase extraction (for unexpected components)
    if target.component_type is None:
        # e.g., "把进度条的高度缩短", "让提示框更圆"
        m_comp = re.search(r'(?:把|将|让|修复|修改)\s*([\u4e00-\u9fa5a-zA-Z0-9_\-]+?)\s*(?:的|在|居中|加|调|变)', prompt)
        if m_comp:
            extracted = m_comp.group(1).strip()
            if len(extracted) >= 2 and extracted not in ("右上角", "左上角", "顶部", "底部", "左侧", "右侧", "居中"):
                target.component_type = extracted
                target.search_keywords.append(extracted)

    # 4. Synthesize intelligent CSS selector hints
    comp = target.component_type or "button"
    pos = target.spatial_position or "顶部"

    if comp in ("按钮", "button"):
        if pos in ("右上角", "top-right"):
            target.css_selector_hints.extend([
                ".header .btn", "header button", ".top-right .btn",
                ".navbar-end .btn", ".btn-header", ".header-action",
                "button[type='submit']",
            ])
        else:
            target.css_selector_hints.extend([".btn", "button", ".button", "button.primary"])
    elif comp in ("搜索框", "search"):
        target.css_selector_hints.extend([
            ".search-input", "input[type='search']", ".search-box", ".search-bar",
        ])
    elif comp in ("卡片", "card"):
        target.css_selector_hints.extend([".card", ".panel", ".box", ".card-body"])
    elif comp in ("导航栏", "navbar"):
        target.css_selector_hints.extend([".navbar", "nav", "header nav", ".nav-bar"])
    elif comp in ("弹窗", "modal"):
        target.css_selector_hints.extend([".modal", ".dialog", ".popup", "[role='dialog']"])
    elif comp in ("输入框", "input"):
        target.css_selector_hints.extend([".input", "input", ".form-control", "textarea"])

    # 5. Deduplicate keywords preserving order
    seen: set[str] = set()
    dedup: list[str] = []
    for kw in target.search_keywords:
        kw_clean = kw.lower().strip()
        if kw_clean and kw_clean not in seen:
            seen.add(kw_clean)
            dedup.append(kw)
    target.search_keywords = dedup

    return target
