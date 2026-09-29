from sylphae.agent.qualitative import (
    adjust_tailwind_class,
    darken_color,
    decompose_qualitative_intents,
    lighten_color,
    scale_font_size,
)
from sylphae.agent.spatial import resolve_spatial_anchors


def test_resolve_spatial_anchors():
    target = resolve_spatial_anchors("把右上角的按钮颜色加深，字体调大一些")
    assert target.spatial_position == "右上角"
    assert target.component_type == "按钮"
    assert "button" in target.search_keywords or "btn" in target.search_keywords
    assert any("btn" in s for s in target.css_selector_hints)

    target_left = resolve_spatial_anchors("左上角品牌logo调整")
    assert target_left.spatial_position == "左上角"


def test_decompose_qualitative_intents():
    intents = decompose_qualitative_intents("把右上角的按钮颜色加深，字体调大一些")
    assert len(intents) == 2
    cats = {i.category: i.action for i in intents}
    assert cats["color"] == "darken"
    assert cats["font_size"] == "enlarge"

    overflow_intent = decompose_qualitative_intents("图中文字溢出控件了，请修复")
    assert any(i.category == "text_overflow" for i in overflow_intent)


def test_color_and_font_deltas():
    # Hex darken
    dark = darken_color("#3b82f6", 0.25)
    assert dark != "#3b82f6"
    assert dark.startswith("#")

    # Hex lighten
    light = lighten_color("#1d4ed8", 0.25)
    assert light != "#1d4ed8"

    # Scale font size
    scaled_px = scale_font_size("14px", 1.2)
    assert scaled_px == "17px"

    scaled_rem = scale_font_size("1rem", 1.25)
    assert scaled_rem == "1.2rem"

    # Tailwind adjustments
    new_bg = adjust_tailwind_class("bg-blue-400", "darken")
    assert new_bg == "bg-blue-600"

    new_text = adjust_tailwind_class("text-sm", "enlarge")
    assert new_text == "text-base"
