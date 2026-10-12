"""ROUND 34: a value the request quotes with a unit is the same site as the bare number the regex finds.

``_OLD_VALUE_RE`` reads ``from 200px to 300px`` and hands the builder ``old_val_literal="200px"``.
``_LITERAL_RE``, which locates the value in the file, captures only the number: on
``height: 200px;`` it matches ``200`` and stops, leaving the unit in the line.

The two were compared raw, so a request that had located its line correctly escalated with
"no patch could be constructed" -- and had it not, the emission would have spliced ``300px``
in front of the surviving ``px`` and written ``height: 300pxpx``.

The site is the number plus the unit behind it. These tests pin the locate half, the emit
half, and the negative case: a line whose unit differs from the one the request quoted is a
different literal and must not be claimed.
"""

from pathlib import Path

from sylphae.agent.reflex import ReflexAgent

CSS = """\
.navbar {
  height: 200px;
  padding: 8px 16px;
  color: #111;
}
"""


def _agent(tmp_path: Path, src: str = CSS, name: str = "styles/main.css") -> ReflexAgent:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(src, encoding="utf-8", newline="\n")
    return ReflexAgent(root=tmp_path)


def test_a_unit_quoted_by_the_request_still_locates_the_value(tmp_path):
    """``change .navbar height from 200px to 300px`` found the right file and the right
    line and then declined, because the located ``200`` never equalled the quoted ``200px``."""
    agent = _agent(tmp_path)
    spec, new_source = agent._build_patch(
        "change .navbar height from 200px to 300px",
        "styles/main.css", "replace_literal", "300px")
    assert spec is not None, "a correctly located value escalated instead of patching"
    assert spec.kind == "replace_literal"
    assert "height: 300px;" in (new_source or ""), new_source


def test_the_units_replace_rather_than_stack(tmp_path):
    """``200`` -> ``300px`` must consume the trailing ``px`` the regex did not capture.
    ``300pxpx`` is the failure this guards: the value applied, and the file is worse."""
    agent = _agent(tmp_path)
    spec, new_source = agent._build_patch(
        "change .navbar height from 200px to 300px",
        "styles/main.css", "replace_literal", "300px")
    assert new_source is not None
    assert "300pxpx" not in new_source, "the old unit survived beneath the new one"
    assert spec is not None and spec.props["new"] == "300px"


def test_a_bare_number_against_a_quoted_unit_still_matches(tmp_path):
    """The request names the unit, the parameter does not. ``从 200px 改为 300`` leaves
    the line's ``px`` standing -- the unit is not stripped, only the number is replaced."""
    agent = _agent(tmp_path)
    spec, new_source = agent._build_patch(
        "把 .navbar 的高度从 200px 改为 300",
        "styles/main.css", "replace_literal", "300")
    assert spec is not None, "a bare replacement against a quoted unit escalated"
    assert "height: 300px;" in (new_source or ""), new_source


def test_a_line_whose_unit_differs_is_not_the_quoted_value(tmp_path):
    """``from 200px to 300px`` must not claim a line holding ``200rem``. The number
    agreeing is not enough -- the unit behind the match is part of the literal."""
    src = CSS.replace("height: 200px;", "height: 200rem;")
    agent = _agent(tmp_path, src)
    spec, _ = agent._build_patch(
        "change .navbar height from 200px to 300px",
        "styles/main.css", "replace_literal", "300px")
    assert spec is None, f"a mismatched unit was accepted: {spec}"


def test_the_unnamed_path_is_unchanged(tmp_path):
    """The control. ``缩短一半`` names no old value, so the anchor is the keyword and the
    multiplier replaces the number while the unit stands: ``0.5x`` is the vocabulary's
    intent there, and the unit logic must not reach it."""
    agent = _agent(tmp_path)
    spec, new_source = agent._build_patch(
        "把导航栏的高度缩短一些", "styles/main.css", "replace_literal", "100")
    assert spec is not None, "the keyword path regressed"
    assert "height: 100px;" in (new_source or ""), new_source
