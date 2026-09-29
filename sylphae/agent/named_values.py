"""Named values in a request, and the literals they stand for.

A request can name a value ("改成深蓝色") or describe a change to one ("再深一点").
``qualitative`` handles the second.  This module handles the first.

The two look identical from the outside -- neither produced a patch -- but they are
different problems.  "Darker" needs a transform *and* a starting value, so it may
genuinely be unanswerable.  "Dark blue" needs only a value, and colour names and CSS
property names are finite, well-known sets, so the answer is a lookup rather than an
inference.  Treating the lookup as inference is what made "把 navbar 背景改成深蓝色"
locate the right file, name no number, and build no patch at all.

Both tables are deliberately small and closed.  A name that is not in them returns
``None`` and the caller escalates, which is the right outcome -- an approximate colour
invented for an unknown name would be indistinguishable from a correct one downstream.
"""

from __future__ import annotations

import re

__all__ = [
    "NAMED_COLORS",
    "CSS_PROPERTY_NAMES",
    "COLOR_PROPERTY_CUES",
    "color_property_names",
    "find_assigned_color",
    "find_valued_property",
]

#: Colour names a request may use, to the literal they mean.  Keys are matched
#: longest-first, because Chinese colour names compose: "深蓝色" contains both "深蓝"
#: and "蓝色", and the more specific reading is the one meant.
NAMED_COLORS: dict[str, str] = {
    # Chinese
    "深蓝色": "#1e3a8a",
    "深蓝": "#1e3a8a",
    "浅蓝色": "#bfdbfe",
    "浅蓝": "#bfdbfe",
    "天蓝色": "#0ea5e9",
    "天蓝": "#0ea5e9",
    "藏青": "#1e3a8a",
    "蓝色": "#3b82f6",
    "蓝": "#3b82f6",
    "深红色": "#991b1b",
    "深红": "#991b1b",
    "浅红色": "#fca5a5",
    "浅红": "#fca5a5",
    "红色": "#ef4444",
    "红": "#ef4444",
    "深绿色": "#166534",
    "深绿": "#166534",
    "浅绿色": "#86efac",
    "浅绿": "#86efac",
    "绿色": "#22c55e",
    "绿": "#22c55e",
    "深灰色": "#374151",
    "深灰": "#374151",
    "浅灰色": "#d1d5db",
    "浅灰": "#d1d5db",
    "灰色": "#6b7280",
    "灰": "#6b7280",
    "黑色": "#000000",
    "黑": "#000000",
    "白色": "#ffffff",
    "白": "#ffffff",
    "黄色": "#eab308",
    "黄": "#eab308",
    "橙色": "#f97316",
    "橙": "#f97316",
    "紫色": "#8b5cf6",
    "紫": "#8b5cf6",
    "粉色": "#ec4899",
    "粉": "#ec4899",
    "棕色": "#78350f",
    "棕": "#78350f",
    "青色": "#06b6d4",
    "青": "#06b6d4",
    "金色": "#d4af37",
    "金": "#d4af37",
    "银色": "#c0c0c0",
    "银": "#c0c0c0",
    # English
    "dark blue": "#1e3a8a",
    "light blue": "#bfdbfe",
    "navy": "#1e3a8a",
    "blue": "#3b82f6",
    "dark red": "#991b1b",
    "light red": "#fca5a5",
    "red": "#ef4444",
    "dark green": "#166534",
    "light green": "#86efac",
    "green": "#22c55e",
    "dark gray": "#374151",
    "dark grey": "#374151",
    "light gray": "#d1d5db",
    "light grey": "#d1d5db",
    "gray": "#6b7280",
    "grey": "#6b7280",
    "black": "#000000",
    "white": "#ffffff",
    "yellow": "#eab308",
    "orange": "#f97316",
    "purple": "#8b5cf6",
    "pink": "#ec4899",
    "brown": "#78350f",
    "cyan": "#06b6d4",
    "gold": "#d4af37",
    "silver": "#c0c0c0",
}

#: Property names a request may use, to the CSS property they name.
CSS_PROPERTY_NAMES: dict[str, str] = {
    "圆角半径": "border-radius",
    "圆角": "border-radius",
    "内边距": "padding",
    "外边距": "margin",
    "间距": "gap",
    "边框": "border",
    "阴影": "box-shadow",
    "透明度": "opacity",
    "字体大小": "font-size",
    "字号": "font-size",
    "行高": "line-height",
    "字重": "font-weight",
    "背景色": "background-color",
    "底色": "background-color",
    "背景": "background-color",
    "文字颜色": "color",
    "字体颜色": "color",
    "颜色": "color",
    "最大宽度": "max-width",
    "最小宽度": "min-width",
    "宽度": "width",
    "高度": "height",
    "border-radius": "border-radius",
    "padding": "padding",
    "margin": "margin",
    "gap": "gap",
    "opacity": "opacity",
}

#: Verbs that introduce the value being assigned.  A colour mentioned before the cue is
#: the current one ("把红色的按钮改成蓝色"), so only what follows it is the answer.
_ASSIGN_CUE = re.compile(
    r"(?:改成|改为|变成|换成|替换为|设为|设置为|调成|调为|设成|"
    r"\bto\b|\binto\b)",
    re.IGNORECASE,
)

#: A CSS length: a number with an optional unit.  Unitless is allowed because zero is
#: the common case and ``line-height`` is legitimately unitless.
_CSS_LENGTH = re.compile(r"(\d+(?:\.\d+)?\s*(?:px|rem|em|%|vh|vw|pt|ch)?)")

_COLOR_KEYS = sorted(NAMED_COLORS, key=len, reverse=True)
_PROPERTY_KEYS = sorted(CSS_PROPERTY_NAMES, key=len, reverse=True)


#: Which CSS colour properties the request's wording points at.  A cue names a *region*
#: ("背景"), not a property, so it maps to a set of properties rather than one: the file
#: decides whether the region is declared as ``background`` or ``background-color``.
#:
#: The sets are written out rather than matched by substring, because substring
#: matching gets the common case backwards -- ``"color" in "background-color"`` is true,
#: so a request about a badge's text colour rewrote the navbar's background.  A property
#: either is in the set or is not.
COLOR_PROPERTY_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("背景", ("background", "background-color")),
    ("底色", ("background", "background-color")),
    ("边框", ("border", "border-color")),
    ("描边", ("border", "border-color")),
    ("文字", ("color",)),
    ("字体", ("color",)),
    ("标题", ("color",)),
    ("background", ("background", "background-color")),
    ("border", ("border", "border-color")),
    ("text", ("color",)),
)


def color_property_names(text: str) -> tuple[str, ...] | None:
    """The colour properties ``text`` points at, or ``None`` when it names no region."""
    lowered = text.lower()
    for cue, names in COLOR_PROPERTY_CUES:
        if cue.isascii():
            if re.search(rf"(?<![a-z]){re.escape(cue)}(?![a-z])", lowered):
                return names
        elif cue in text:
            return names
    return None


def find_assigned_color(text: str) -> str | None:
    """The colour literal ``text`` assigns, or ``None``.

    Reads only what follows the assignment cue, so a request that names both the current
    and the intended colour resolves to the intended one.
    """
    cue = _ASSIGN_CUE.search(text)
    if cue is None:
        return None
    tail = text[cue.end() :]
    lowered = tail.lower()
    for key in _COLOR_KEYS:
        if key.isascii():
            # English names need boundaries: "gold" must not fire inside "goldenrod".
            if re.search(rf"(?<![a-z]){re.escape(key)}(?![a-z])", lowered):
                return NAMED_COLORS[key]
        elif key in tail:
            return NAMED_COLORS[key]
    return None


def find_valued_property(text: str) -> tuple[str, str] | None:
    """The CSS property and value ``text`` gives, or ``None``.

    ``"给 badge 加一个圆角 8px"`` -> ``("border-radius", "8px")``, and so does
    ``"把 badge 的圆角调成 8px"``.  Whether that becomes an insertion or a replacement
    is a question about the file, not about the wording, so it is not decided here --
    what both phrasings agree on is the (property, value) pair, which is all this
    returns.

    Longest property name first, for the same reason as colours: a shorter key would
    otherwise shadow a longer one that shares its tail.
    """
    for key in _PROPERTY_KEYS:
        m = re.search(
            rf"{re.escape(key)}\s*(?:[:：]\s*)?"
            rf"(?:(?:调成|调为|改成|改为|变成|设为|设置为|加到|加上|加|到|为|to|of)\s*)?"
            rf"{_CSS_LENGTH.pattern}",
            text,
            re.IGNORECASE if key.isascii() else 0,
        )
        if m:
            return CSS_PROPERTY_NAMES[key], m.group(1).strip()
    return None
