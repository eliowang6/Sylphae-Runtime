"""Qualitative and fuzzy intent delta reasoning engine.

Translates subjective / relative developer requests (e.g. '颜色加深', '字体调大一些',
'间距紧凑一点', 'darken button', 'make cards more rounded', 'increase padding')
into concrete color, font-size, layout spacing, and dimension delta modifications.
"""

from __future__ import annotations

import colorsys
import re
from dataclasses import dataclass, field

__all__ = [
    "QualitativeIntent",
    "decompose_qualitative_intents",
    "darken_color",
    "lighten_color",
    "scale_font_size",
    "scale_dimension",
    "adjust_tailwind_class",
]

# Tailwind scales
_TAILWIND_SHADES = ["50", "100", "200", "300", "400", "500", "600", "700", "800", "900", "950"]
_TAILWIND_SIZES = ["xs", "sm", "base", "lg", "xl", "2xl", "3xl", "4xl", "5xl"]
_TAILWIND_SPACING = ["0", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4", "5", "6", "8", "10", "12", "16"]
_TAILWIND_RADIUS = ["none", "sm", "DEFAULT", "md", "lg", "xl", "2xl", "3xl", "full"]

# Basic CSS named colors mapped to hex
_NAMED_COLORS = {
    "white": "#ffffff",
    "black": "#000000",
    "gray": "#6b7280",
    "grey": "#6b7280",
    "red": "#ef4444",
    "blue": "#3b82f6",
    "green": "#10b981",
    "yellow": "#f59e0b",
    "purple": "#8b5cf6",
    "slate": "#64748b",
}


@dataclass
class QualitativeIntent:
    category: str              # 'color', 'font_size', 'dimension', 'spacing', 'border_radius', 'text_overflow', 'opacity'
    action: str                # 'darken', 'lighten', 'enlarge', 'shrink', 'compact', 'spacious', 'round', 'fix_overflow'
    target_properties: list[str] = field(default_factory=list)
    raw_subprompt: str = ""


def decompose_qualitative_intents(prompt: str) -> list[QualitativeIntent]:
    """Decompose compound developer prompts into discrete qualitative sub-intents."""
    intents: list[QualitativeIntent] = []
    lower = prompt.lower()

    # 1. Color darken / lighten
    darken_pats = ("加深", "深色", "更深", "暗一点", "深一点", "darken", "darker", "deepen", "dim")
    lighten_pats = ("变浅", "浅色", "更浅", "淡一点", "亮一点", "lighten", "lighter", "brighten", "pale")

    if any(p in lower or p in prompt for p in darken_pats):
        intents.append(QualitativeIntent(
            category="color",
            action="darken",
            target_properties=["background-color", "background", "color", "border-color"],
            raw_subprompt="颜色加深",
        ))
    elif any(p in lower or p in prompt for p in lighten_pats):
        intents.append(QualitativeIntent(
            category="color",
            action="lighten",
            target_properties=["background-color", "background", "color", "border-color"],
            raw_subprompt="颜色变浅",
        ))

    # 2. Font size enlarge / shrink
    font_large_pats = ("字体调大", "字号加大", "字体变大", "字调大", "字体大一点", "加大字号", "更大字",
                       "larger font", "increase font", "bigger font", "font size larger")
    font_small_pats = ("字体调小", "字号减小", "字体缩小", "字调小", "字体小一点", "减小字号",
                       "smaller font", "decrease font", "font size smaller")

    if any(p in lower or p in prompt for p in font_large_pats):
        intents.append(QualitativeIntent(
            category="font_size",
            action="enlarge",
            target_properties=["font-size", "fontSize"],
            raw_subprompt="字体调大",
        ))
    elif any(p in lower or p in prompt for p in font_small_pats):
        intents.append(QualitativeIntent(
            category="font_size",
            action="shrink",
            target_properties=["font-size", "fontSize"],
            raw_subprompt="字体调小",
        ))

    # 3. Dimension (height / width) shrink / expand
    dim_shrink_pats = ("缩短", "减小高度", "变矮", "矮一点", "收紧", "变窄", "收缩", "shorten", "narrower", "decrease height")
    dim_expand_pats = ("拉长", "增加高度", "变高", "加宽", "展宽", "变宽", "widen", "wider", "increase height", "taller")

    if any(p in lower or p in prompt for p in dim_shrink_pats):
        intents.append(QualitativeIntent(
            category="dimension",
            action="shrink",
            target_properties=["height", "width", "min-height", "max-height"],
            raw_subprompt="尺寸缩短",
        ))
    elif any(p in lower or p in prompt for p in dim_expand_pats):
        intents.append(QualitativeIntent(
            category="dimension",
            action="enlarge",
            target_properties=["height", "width", "min-height", "max-height"],
            raw_subprompt="尺寸增加",
        ))

    # 4. Spacing (padding / margin / gap) compact / spacious
    spacing_compact_pats = ("紧凑", "间距减小", "边距缩小", "更紧", "compact", "tighter", "reduce padding", "reduce gap")
    spacing_spacious_pats = ("宽松", "增加间距", "边距加大", "更松散", "spacious", "more padding", "increase gap", "looser")

    if any(p in lower or p in prompt for p in spacing_compact_pats):
        intents.append(QualitativeIntent(
            category="spacing",
            action="compact",
            target_properties=["padding", "margin", "gap"],
            raw_subprompt="间距紧凑",
        ))
    elif any(p in lower or p in prompt for p in spacing_spacious_pats):
        intents.append(QualitativeIntent(
            category="spacing",
            action="spacious",
            target_properties=["padding", "margin", "gap"],
            raw_subprompt="间距增加",
        ))

    # 5. Border radius (rounded corners)
    if any(p in lower or p in prompt for p in ("圆角加大", "更圆", "更圆润", "增加圆角", "more rounded", "round corners", "rounded")):
        intents.append(QualitativeIntent(
            category="border_radius",
            action="round",
            target_properties=["border-radius", "borderRadius"],
            raw_subprompt="圆角加大",
        ))

    # 6. Text overflow / truncation defect
    if any(p in lower or p in prompt for p in ("溢出", "文字截断", "放不下", "换行", "超出", "overflow", "clipped text", "ellipsis")):
        intents.append(QualitativeIntent(
            category="text_overflow",
            action="fix_overflow",
            target_properties=["overflow", "text-overflow", "white-space", "flex-wrap"],
            raw_subprompt="文字溢出修复",
        ))

    return intents


def darken_color(color_str: str, factor: float = 0.25) -> str:
    """Darken a hex, rgb, or named color by factor (0.0 to 1.0)."""
    color = color_str.strip().strip("'\"")

    # Named color
    if color.lower() in _NAMED_COLORS:
        color = _NAMED_COLORS[color.lower()]

    # Hex color #RRGGBB or #RGB or #RRGGBBAA
    hex_m = re.match(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$", color)
    if hex_m:
        val = hex_m.group(1)
        if len(val) == 3:
            r, g, b = int(val[0] * 2, 16), int(val[1] * 2, 16), int(val[2] * 2, 16)
        else:
            r, g, b = int(val[0:2], 16), int(val[2:4], 16), int(val[4:6], 16)

        h, lightness, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
        new_l = max(0.0, lightness * (1.0 - factor))
        nr, ng, nb = colorsys.hls_to_rgb(h, new_l, s)
        return f"#{int(nr * 255):02x}{int(ng * 255):02x}{int(nb * 255):02x}"

    # rgb(r, g, b)
    rgb_m = re.match(r"^rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*([\d\.]+))?\s*\)$", color, re.I)
    if rgb_m:
        r, g, b = int(rgb_m.group(1)), int(rgb_m.group(2)), int(rgb_m.group(3))
        alpha = rgb_m.group(4)
        h, lightness, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
        new_l = max(0.0, lightness * (1.0 - factor))
        nr, ng, nb = colorsys.hls_to_rgb(h, new_l, s)
        if alpha:
            return f"rgba({int(nr * 255)}, {int(ng * 255)}, {int(nb * 255)}, {alpha})"
        return f"rgb({int(nr * 255)}, {int(ng * 255)}, {int(nb * 255)})"

    return color


def lighten_color(color_str: str, factor: float = 0.25) -> str:
    """Lighten a hex, rgb, or named color by factor (0.0 to 1.0)."""
    color = color_str.strip().strip("'\"")

    if color.lower() in _NAMED_COLORS:
        color = _NAMED_COLORS[color.lower()]

    hex_m = re.match(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$", color)
    if hex_m:
        val = hex_m.group(1)
        r = int(val[0] * 2 if len(val) == 3 else val[0:2], 16)
        g = int(val[1] * 2 if len(val) == 3 else val[2:4], 16)
        b = int(val[2] * 2 if len(val) == 3 else val[4:6], 16)

        h, lightness, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
        new_l = min(1.0, lightness + (1.0 - lightness) * factor)
        nr, ng, nb = colorsys.hls_to_rgb(h, new_l, s)
        return f"#{int(nr * 255):02x}{int(ng * 255):02x}{int(nb * 255):02x}"

    rgb_m = re.match(r"^rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*([\d\.]+))?\s*\)$", color, re.I)
    if rgb_m:
        r, g, b = int(rgb_m.group(1)), int(rgb_m.group(2)), int(rgb_m.group(3))
        alpha = rgb_m.group(4)
        h, lightness, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
        new_l = min(1.0, lightness + (1.0 - lightness) * factor)
        nr, ng, nb = colorsys.hls_to_rgb(h, new_l, s)
        if alpha:
            return f"rgba({int(nr * 255)}, {int(ng * 255)}, {int(nb * 255)}, {alpha})"
        return f"rgb({int(nr * 255)}, {int(ng * 255)}, {int(nb * 255)})"

    return color


def scale_font_size(size_str: str, scale: float = 1.2) -> str:
    """Scale an explicit font size string (e.g. 14px, 1rem, 0.875rem)."""
    return scale_dimension(size_str, scale)


def scale_dimension(dim_str: str, scale: float = 1.2) -> str:
    """Scale an explicit CSS dimension string (e.g. 14px, 1rem, 100%, 24vh)."""
    clean = dim_str.strip().strip(";'\"")
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)(px|rem|em|pt|%|vh|vw)$", clean, re.I)
    if m:
        num = float(m.group(1))
        unit = m.group(2)
        new_num = round(num * scale, 1)
        if unit == "px" or unit == "%":
            new_num = int(round(num * scale))
        return f"{new_num}{unit}"
    return dim_str


def adjust_tailwind_class(class_token: str, action: str) -> str:
    """Adjust a Tailwind CSS token along color, font, spacing, or radius scales."""
    token = class_token.strip()

    # 1. Color shade (e.g. bg-blue-500, text-gray-700, border-slate-300) with optional opacity (/80)
    color_m = re.match(r"^(bg|text|border|ring|fill|stroke)-([a-z]+)-([0-9]+)(/[0-9]+)?$", token)
    if color_m:
        prefix, color_name, shade, opacity = color_m.group(1), color_m.group(2), color_m.group(3), color_m.group(4) or ""
        if shade in _TAILWIND_SHADES:
            idx = _TAILWIND_SHADES.index(shade)
            step = 2
            if action in ("darken", "enlarge"):
                new_idx = min(len(_TAILWIND_SHADES) - 1, idx + step)
            else:
                new_idx = max(0, idx - step)
            return f"{prefix}-{color_name}-{_TAILWIND_SHADES[new_idx]}{opacity}"

    # Arbitrary hex in Tailwind (e.g. bg-[#3b82f6])
    arb_col_m = re.match(r"^(bg|text|border)-\[#([0-9a-fA-F]{3,8})\]$", token)
    if arb_col_m:
        prefix, hex_val = arb_col_m.group(1), arb_col_m.group(2)
        adj_hex = darken_color(f"#{hex_val}") if action in ("darken", "enlarge") else lighten_color(f"#{hex_val}")
        return f"{prefix}-[{adj_hex}]"

    # 2. Font size (e.g. text-sm -> text-base, text-base -> text-lg)
    font_m = re.match(r"^text-(xs|sm|base|lg|xl|2xl|3xl|4xl|5xl)$", token)
    if font_m:
        size = font_m.group(1)
        if size in _TAILWIND_SIZES:
            idx = _TAILWIND_SIZES.index(size)
            if action in ("enlarge", "spacious"):
                new_idx = min(len(_TAILWIND_SIZES) - 1, idx + 1)
            else:
                new_idx = max(0, idx - 1)
            return f"text-{_TAILWIND_SIZES[new_idx]}"

    # 3. Spacing: padding, margin, gap (e.g. p-2 -> p-4 or p-1, gap-3 -> gap-4)
    space_m = re.match(r"^(p|px|py|pt|pb|pl|pr|m|mx|my|mt|mb|ml|mr|gap|gap-x|gap-y)-([0-9\.]+)$", token)
    if space_m:
        prefix, val = space_m.group(1), space_m.group(2)
        if val in _TAILWIND_SPACING:
            idx = _TAILWIND_SPACING.index(val)
            if action in ("spacious", "enlarge"):
                new_idx = min(len(_TAILWIND_SPACING) - 1, idx + 1)
            else:
                new_idx = max(0, idx - 1)
            return f"{prefix}-{_TAILWIND_SPACING[new_idx]}"

    # 4. Border radius (e.g. rounded-sm -> rounded-md -> rounded-lg)
    rad_m = re.match(r"^rounded(?:-([a-z0-9]+))?$", token)
    if rad_m:
        cur_rad = rad_m.group(1) or "DEFAULT"
        if cur_rad in _TAILWIND_RADIUS:
            idx = _TAILWIND_RADIUS.index(cur_rad)
            if action in ("round", "enlarge"):
                new_idx = min(len(_TAILWIND_RADIUS) - 1, idx + 1)
            else:
                new_idx = max(0, idx - 1)
            target = _TAILWIND_RADIUS[new_idx]
            return "rounded" if target == "DEFAULT" else f"rounded-{target}"

    return token
