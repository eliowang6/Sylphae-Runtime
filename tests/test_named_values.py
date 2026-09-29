"""Invariants for the named-value lookup.

A request can name the value it wants ("改成深蓝色") or pair a property with one
("圆角 8px").  Neither is a transform of an existing value, so neither was reachable
before: the request located the right file, named no number for the numeric branch to
substitute, and produced no patch.  These cases pin the lookup that closed that gap.
"""

from __future__ import annotations

import pytest

from sylphae.agent.named_values import (
    CSS_PROPERTY_NAMES,
    NAMED_COLORS,
    color_property_names,
    find_assigned_color,
    find_valued_property,
)


# --------------------------------------------------------------------------- #
# colours
# --------------------------------------------------------------------------- #


def test_colour_after_the_assignment_cue() -> None:
    assert find_assigned_color("把 navbar 背景改成深蓝色") == NAMED_COLORS["深蓝色"]
    assert find_assigned_color("把标题改成金色") == NAMED_COLORS["金色"]


def test_only_what_follows_the_cue_is_the_target() -> None:
    # The colour named first is the current one, not the intended one.
    assert find_assigned_color("把红色的按钮改成蓝色") == NAMED_COLORS["蓝色"]


def test_longest_colour_name_wins() -> None:
    # "深蓝色" contains both "深蓝" and "蓝色"; the specific reading is the one meant, and
    # resolving by dict order instead would make the answer depend on insertion order.
    assert find_assigned_color("把背景改成深蓝色") == NAMED_COLORS["深蓝色"]
    assert NAMED_COLORS["深蓝色"] != NAMED_COLORS["蓝色"]
    assert find_assigned_color("把背景改成蓝色") == NAMED_COLORS["蓝色"]


def test_english_colour_needs_a_boundary() -> None:
    assert find_assigned_color("change the header to dark blue") == NAMED_COLORS["dark blue"]
    # "gold" must not fire inside a longer word.
    assert find_assigned_color("set it to goldenrod please") is None


def test_no_cue_and_no_colour_yield_nothing() -> None:
    assert find_assigned_color("把导航栏背景改深一点") is None
    assert find_assigned_color("把背景改成 #1f2937") is None


# --------------------------------------------------------------------------- #
# colour property regions
# --------------------------------------------------------------------------- #


def test_region_maps_to_a_property_set() -> None:
    assert color_property_names("把 navbar 背景改成深蓝色") == (
        "background",
        "background-color",
    )
    assert color_property_names("把文字颜色改成红色") == ("color",)


def test_text_colour_does_not_select_background_color() -> None:
    # Substring matching gets this backwards: "color" is inside "background-color", so a
    # request about text silently qualified the background as a target too.
    names = color_property_names("把 badge 的文字颜色改成红色")
    assert names is not None
    assert "background-color" not in names
    assert "background" not in names


# --------------------------------------------------------------------------- #
# property / value pairs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ("给 badge 加一个圆角 8px", ("border-radius", "8px")),
        ("把 badge 的圆角调成 8px", ("border-radius", "8px")),
        ("多加点内边距 16px", ("padding", "16px")),
        ("把外边距改为 0", ("margin", "0")),
        ("按钮的高度 40px", ("height", "40px")),
    ],
)
def test_property_and_value_are_extracted(prompt: str, expected: tuple[str, str]) -> None:
    assert find_valued_property(prompt) == expected


def test_longest_property_name_wins() -> None:
    # "字体大小" and "字号" share no tail, but "宽度" is a tail of "最大宽度".
    assert find_valued_property("把最大宽度设为 800px") == ("max-width", "800px")
    assert find_valued_property("把宽度设为 800px") == ("width", "800px")


def test_a_prompt_without_a_property_yields_nothing() -> None:
    assert find_valued_property("把导航栏高度缩短一些") is None
    assert find_valued_property("改一下样式") is None


def test_every_property_name_maps_to_a_css_property() -> None:
    # A value here that is not a real CSS property would produce a declaration the
    # browser drops, which is worse than refusing: the patch would look applied.
    for key, prop in CSS_PROPERTY_NAMES.items():
        assert prop == prop.lower()
        assert prop.replace("-", "").isalnum()
        assert prop == prop.strip()
