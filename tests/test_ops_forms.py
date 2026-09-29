"""Tests for :mod:`sylphae.ops.forms` and the detectors built on it.

The module exists to close a producer/detector gap and to stop a Python lexer
from being used as a stand-in on files it cannot lex, so the tests come in pairs:
every accepted form has a near-miss that must be refused, and every classification
is checked against the replay invariant.
"""

from __future__ import annotations

import pytest

from sylphae.ops.apply import apply_op, apply_sequence
from sylphae.ops.detect import detect_op, detect_sequence
from sylphae.ops.forms import (
    SYNTAXES,
    content_of,
    deleted_members,
    extract_members,
    has_python_lexer,
    indent_of,
    inserted_members,
    is_comment,
    syntax_for,
)
from sylphae.ops.vocab import FREE_FORM_KIND

CSS = "t.css"
TS = "t.ts"
JSON = "t.json"
YAML = "t.yaml"
TOML = "t.toml"
ENV = "t.env"


def _lines(src: str) -> list[str]:
    return src.splitlines(keepends=True)


def _member(src: str, path: str = CSS, line: int = 0):
    syntax = syntax_for(path)
    assert syntax is not None
    return syntax.member(_lines(src)[line])


# --------------------------------------------------------------------------- #
# syntax dispatch
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path,expected_family",
    [
        ("a.css", "brace"),
        ("a.scss", "brace"),
        ("a.less", "brace"),
        ("a.json", "brace"),
        ("a.ts", "brace"),
        ("a.tsx", "brace"),
        ("a.jsx", "brace"),
        ("a.vue", "brace"),
        ("a.svelte", "brace"),
        ("a.toml", "indent"),
        ("a.yaml", "indent"),
        ("a.yml", "indent"),
        ("a.env", "indent"),
        ("a.ini", "indent"),
        ("a.properties", "indent"),
    ],
)
def test_syntax_for_maps_extensions(path: str, expected_family: str) -> None:
    syntax = syntax_for(path)
    assert syntax is not None and syntax.family == expected_family


@pytest.mark.parametrize(
    "path",
    ["a.py", "a.pyi", "a.go", "a.rs", "a.java", "a.c", "a.h", "a.html", "a.sql", "", "inline"],
)
def test_syntax_for_declines_languages_without_a_member_form(path: str) -> None:
    assert syntax_for(path) is None


def test_a_bare_extension_is_a_hidden_file_not_a_filename_with_a_suffix() -> None:
    """``.css`` on its own is a file *named* ``.css``; it carries no extension.

    ``os.path.splitext`` follows the POSIX reading of a leading dot, so this is
    correctly declined -- there is no language to read the line forms from.
    """
    assert syntax_for(".css") is None
    assert syntax_for("dir/.css") is None
    assert syntax_for("dir/t.css") is not None


def test_extension_matching_is_case_insensitive() -> None:
    assert syntax_for("T.CSS") is not None
    assert syntax_for("Component.TSX") is not None


def test_every_registered_extension_is_unique() -> None:
    seen: set[str] = set()
    for syntax in SYNTAXES:
        for ext in syntax.extensions:
            assert ext not in seen, f"{ext} is claimed by two syntaxes"
            seen.add(ext)


@pytest.mark.parametrize(
    "path,expected",
    [
        ("a.py", True),
        ("a.pyi", True),
        ("", True),
        ("inline", True),
        ("a.css", False),
        ("a.ts", False),
        ("a.go", False),
        ("a.rs", False),
        ("a.yaml", False),
    ],
)
def test_has_python_lexer(path: str, expected: bool) -> None:
    """``tokenize`` lexes Python; everywhere else it is a stand-in."""
    assert has_python_lexer(path) is expected


# --------------------------------------------------------------------------- #
# member form
# --------------------------------------------------------------------------- #


def test_css_declaration_reads_as_a_member() -> None:
    member = _member("  color: red;\n")
    assert member is not None
    assert (member.indent, member.key, member.value, member.terminator) == (
        "  ",
        "color",
        "red",
        ";",
    )


def test_css_declaration_without_its_semicolon_still_reads_as_a_member() -> None:
    member = _member("  color: red\n")
    assert member is not None and member.value == "red" and member.terminator == ""


def test_css_custom_property_reads_as_a_member() -> None:
    member = _member("  --main-gap: 4px;\n")
    assert member is not None and member.key == "--main-gap" and member.value == "4px"


def test_css_value_keeps_an_inner_colon() -> None:
    member = _member("  background: url(http://example.com/a.png);\n")
    assert member is not None
    assert member.key == "background"
    assert member.value == "url(http://example.com/a.png)"


@pytest.mark.parametrize(
    "line",
    [
        "  a:hover {\n",  # pseudo-class header, not a declaration
        "  * {\n",  # universal selector header
        "  @apply flex gap-2;\n",  # at-rule, no separator after the key
        "  }\n",  # closer
        "  // note\n",  # comment
        "  /* note */\n",  # comment
        "\n",  # blank
        "  color:\n",  # empty value
        "  color: ;\n",  # empty value
        "  onClick: () => { go(); },\n",  # brace in the value
    ],
)
def test_non_member_lines_are_refused(line: str) -> None:
    assert _member(line) is None


def test_the_brace_guard_is_what_saves_pseudo_selectors() -> None:
    """``a:hover {`` would otherwise read as key ``a`` with value ``hover {``."""
    assert _member("  a:hover {\n") is None
    # The same text without the brace is a legitimate (if odd) member.
    member = _member("  a:hover\n")
    assert member is not None and member.key == "a" and member.value == "hover"


def test_yaml_container_header_is_not_a_member() -> None:
    assert _member("root:\n", YAML) is None
    assert _member("  child: 1\n", YAML) is not None


def test_json_quoted_key_reads_as_a_member() -> None:
    member = _member('  "a-b": 1,\n', JSON)
    assert member is not None
    assert (member.key, member.value, member.terminator) == ('"a-b"', "1", ",")


def test_env_key_reads_as_a_member() -> None:
    member = _member("API_KEY=abc\n", ENV)
    assert member is not None and member.key == "API_KEY" and member.value == "abc"


def test_toml_key_reads_as_a_member() -> None:
    member = _member("retries = 3\n", TOML)
    assert member is not None and member.key == "retries" and member.value == "3"


def test_star_guard_keeps_the_universal_selector_out_of_the_comment_branch() -> None:
    syntax = syntax_for(CSS)
    assert syntax is not None
    assert is_comment("* comment", syntax) is True
    assert is_comment("* { margin: 0 }", syntax) is False


def test_line_helpers() -> None:
    assert content_of("  x = 1\r\n") == "  x = 1"
    assert indent_of("  x = 1") == "  "
    assert indent_of("\tx = 1") == "\t"
    assert indent_of("x = 1") == ""


# --------------------------------------------------------------------------- #
# extract_members
# --------------------------------------------------------------------------- #


def test_extract_members_reads_several_lines_at_one_indent() -> None:
    syntax = syntax_for(CSS)
    assert syntax is not None
    members = extract_members("  color: red;\n  height: 4px;\n", syntax)
    assert members is not None and [m.key for m in members] == ["color", "height"]


def test_extract_members_skips_blank_lines() -> None:
    syntax = syntax_for(CSS)
    assert syntax is not None
    members = extract_members("\n  color: red;\n\n", syntax)
    assert members is not None and len(members) == 1


@pytest.mark.parametrize(
    "text",
    [
        "",
        "\n",
        "  color: red;\n  }\n",  # one good line, one non-member
        "  color: red;\n    height: 4px;\n",  # mixed indentation
        "  a:hover {\n",  # a block header is not a member
    ],
)
def test_extract_members_refuses_rather_than_partially_reading(text: str) -> None:
    syntax = syntax_for(CSS)
    assert syntax is not None
    assert extract_members(text, syntax) is None


# --------------------------------------------------------------------------- #
# insertion
# --------------------------------------------------------------------------- #

CSS_TWO_RULES = ".a {\n  color: red;\n}\n\n.b {\n  color: blue;\n}\n"


def test_member_is_inserted_after_its_sibling() -> None:
    lines = _lines(".a {\n  color: red;\n}\n")
    members = inserted_members(CSS, lines, 3, "  height: 4px;\n")
    assert members is not None and members[0].key == "height"


def test_member_is_inserted_as_the_first_line_of_an_existing_block() -> None:
    lines = _lines(".a {\n}\n")
    members = inserted_members(CSS, lines, 2, "  color: red;\n")
    assert members is not None and members[0].key == "color"


def test_member_is_appended_after_the_last_line_of_a_file() -> None:
    lines = _lines("A=1\n")
    members = inserted_members(ENV, lines, 2, "B=2\n")
    assert members is not None and members[0].key == "B"


def test_nested_yaml_member_is_proved_by_its_sibling() -> None:
    lines = _lines("root:\n  a: 1\n")
    assert inserted_members(YAML, lines, 3, "  b: 2\n") is not None


def test_deeper_line_after_a_shallow_non_opener_is_refused() -> None:
    # `x = 1` at depth 0 does not open a block, so nothing at depth 2 can belong
    # to it -- the indentation is simply wrong.
    lines = _lines("x = 1\n")
    assert inserted_members(ENV, lines, 2, "  B=2\n") is None
    assert inserted_members(CSS, lines, 2, "  color: red;\n") is None


def test_member_at_the_top_of_a_brace_file_is_refused() -> None:
    # A brace block cannot begin at the top of the file: there is no opener.
    assert inserted_members(CSS, _lines(""), 1, "  color: red;\n") is None


def test_inserting_before_a_misindented_closer_is_still_the_first_member() -> None:
    """The closer is *after* the insert point, so the opener is the predecessor."""
    lines = _lines(".a {\n  }\n")
    members = inserted_members(CSS, lines, 2, "  color: red;\n")
    assert members is not None and members[0].key == "color"


def test_a_line_deeper_than_a_closed_block_is_refused() -> None:
    """After ``}`` the block is over; a depth-2 line has nothing to belong to."""
    lines = _lines(".a {\n  color: red;\n}\n")
    assert inserted_members(CSS, lines, 4, "  height: 4px;\n") is None


def test_sibling_at_the_same_indent_that_is_not_a_member_breaks_the_chain() -> None:
    # A nested ``a:hover`` header sits at the same indent as the insert point and
    # is not a member, so it cannot vouch for the new line the way a declaration
    # would.  (It is a block opener, but not a *shallower* one.)
    lines = _lines(".a {\n  a:hover {\n    x: 1;\n  }\n}\n")
    assert inserted_members(CSS, lines, 3, "  color: red;\n") is None


def test_unknown_extension_is_refused() -> None:
    lines = _lines("x = 1\n")
    assert inserted_members("m.py", lines, 2, "  color: red;\n") is None


def test_mixed_indentation_insert_is_refused() -> None:
    lines = _lines(".a {\n  color: red;\n}\n")
    assert inserted_members(CSS, lines, 3, "  x: 1;\n    y: 2;\n") is None


# --------------------------------------------------------------------------- #
# deletion
# --------------------------------------------------------------------------- #


def test_member_deletion_is_proved_by_the_preceding_sibling() -> None:
    lines = _lines(".a {\n  color: red;\n  height: 4px;\n}\n")
    assert deleted_members(CSS, lines, 3, 3) is not None


def test_only_member_deletion_is_proved_by_the_closing_brace() -> None:
    lines = _lines(".a {\n  color: red;\n}\n")
    assert deleted_members(CSS, lines, 2, 2) is not None


def test_first_member_deletion_is_proved_by_the_following_sibling() -> None:
    lines = _lines(".a {\n  color: red;\n  height: 4px;\n}\n")
    assert deleted_members(CSS, lines, 2, 2) is not None


def test_deleting_a_brace_or_a_non_member_is_refused() -> None:
    lines = _lines(".a {\n  color: red;\n}\n")
    assert deleted_members(CSS, lines, 3, 3) is None  # the closing brace
    assert deleted_members(CSS, lines, 1, 1) is None  # the selector header


def test_out_of_range_deletion_is_refused() -> None:
    lines = _lines(".a {\n  color: red;\n}\n")
    assert deleted_members(CSS, lines, 2, 99) is None


# --------------------------------------------------------------------------- #
# detect_op integration — the classification the gap was about
# --------------------------------------------------------------------------- #


def _classify(old: str, new: str, path: str) -> str:
    spec = detect_op(old, new, path)
    assert spec is not None
    if spec.kind != FREE_FORM_KIND:
        assert apply_op(old, spec) == new, "detect_op must only emit replayable specs"
    return spec.kind


def test_css_property_insertion_is_an_insert_statement() -> None:
    old = ".a {\n  color: red;\n}\n"
    new = ".a {\n  color: red;\n  height: 4px;\n}\n"
    spec = detect_op(old, new, "t.css")
    assert spec is not None and spec.kind == "insert_statement"
    assert spec.props["key"] == "height" and spec.props["value"] == "4px"
    assert apply_op(old, spec) == new


def test_css_property_deletion_is_a_delete_statement() -> None:
    old = ".a {\n  color: red;\n  height: 4px;\n}\n"
    new = ".a {\n  color: red;\n}\n"
    spec = detect_op(old, new, "t.css")
    assert spec is not None and spec.kind == "delete_statement"
    assert spec.props["key"] == "height"
    assert apply_op(old, spec) == new


def test_two_inserted_members_carry_no_single_key_prop() -> None:
    """One key/value pair cannot describe a two-line edit, so it is not claimed."""
    old = ".a {\n  color: red;\n}\n"
    new = ".a {\n  color: red;\n  height: 4px;\n  width: 2px;\n}\n"
    spec = detect_op(old, new, "t.css")
    assert spec is not None and spec.kind == "insert_statement"
    assert spec.props["n_stmts"] == "2"
    assert "key" not in spec.props and "value" not in spec.props
    assert apply_op(old, spec) == new


def test_css_value_change_is_a_line_replacement_not_a_rename() -> None:
    """``red`` -> ``blue`` is a value change; there is no symbol in a CSS file."""
    old = ".a {\n  color: red;\n}\n"
    new = ".a {\n  color: blue;\n}\n"
    assert _classify(old, new, "t.css") == "replace_line"


def test_ts_object_member_insertion_is_an_insert_statement() -> None:
    old = "const o = {\n  a: 1,\n};\n"
    new = "const o = {\n  a: 1,\n  b: 2,\n};\n"
    spec = detect_op(old, new, "t.ts")
    assert spec is not None and spec.kind == "insert_statement"
    assert spec.props["key"] == "b" and spec.props["value"] == "2"
    assert apply_op(old, spec) == new


def test_yaml_toml_and_env_insertions_are_insert_statements_with_key_props() -> None:
    for path, old, new, key, value in [
        (YAML, "root:\n  a: 1\n", "root:\n  a: 1\n  b: 2\n", "b", "2"),
        (TOML, "[s]\na = 1\n", "[s]\na = 1\nb = 2\n", "b", "2"),
        (ENV, "A=1\n", "A=1\nB=2\n", "B", "2"),
    ]:
        spec = detect_op(old, new, path)
        assert spec is not None and spec.kind == "insert_statement", path
        assert spec.props["key"] == key and spec.props["value"] == value, path
        assert apply_op(old, spec) == new, path


def test_adding_a_member_that_needs_a_terminator_repair_is_a_two_op_sequence() -> None:
    """JSON members need the previous line's comma: one op cannot say both."""
    old = '{\n  "a": 1\n}\n'
    new = '{\n  "a": 1,\n  "b": 2\n}\n'
    seq = detect_sequence(old, new, "t.json")
    assert [s.kind for s in seq] == ["replace_line", "insert_statement"]
    # The key prop carries the source text of the key, quotes included -- the same
    # convention ``add_dict_entry`` follows on the Python path.
    assert seq[1].props["key"] == '"b"' and seq[1].props["value"] == "2"
    assert apply_sequence(old, seq) == new


def test_a_ts_member_addition_with_a_terminator_repair_is_also_a_sequence() -> None:
    old = "const o = {\n  a: 1\n};\n"
    new = "const o = {\n  a: 1,\n  b: 2\n};\n"
    seq = detect_sequence(old, new, "t.ts")
    assert [s.kind for s in seq] == ["replace_line", "insert_statement"]
    assert apply_sequence(old, seq) == new


def test_inserting_a_whole_block_stays_freeform() -> None:
    """A rule block is not a member line; the member primitive must not claim it.

    Pinning this down on purpose: ``add_css_properties`` falls back to appending a
    whole new block when its selector is missing, and that shape -- blank line plus
    a balanced brace block -- is a *different* structural form from a member.
    """
    old = ".a {\n  color: red;\n}\n"
    new = ".a {\n  color: red;\n}\n\n.b {\n  color: blue;\n}\n"
    assert _classify(old, new, "t.css") == FREE_FORM_KIND
    seq = detect_sequence(old, new, "t.css")
    assert [s.kind for s in seq] == [FREE_FORM_KIND]


def test_a_pseudo_selector_insertion_is_not_a_member_insertion() -> None:
    old = "a {\n  color: red;\n}\n"
    new = "a {\n  color: red;\n  a:hover { color: blue; }\n}\n"
    assert _classify(old, new, "t.css") == FREE_FORM_KIND


# --------------------------------------------------------------------------- #
# the rename_symbol gate
# --------------------------------------------------------------------------- #


def test_rename_symbol_still_fires_on_python() -> None:
    old = "def f(node):\n    return node\n"
    new = "def f(item):\n    return item\n"
    assert _classify(old, new, "m.py") == "rename_symbol"


def test_rename_symbol_is_suppressed_where_python_is_not_the_language() -> None:
    """A consistent identifier swap in CSS is a value change, not a rename."""
    old = "a {\n  color: red;\n}\n"
    new = "a {\n  color: blue;\n}\n"
    spec = detect_op(old, new, "t.css")
    assert spec is not None and spec.kind != "rename_symbol"


def test_rename_symbol_is_suppressed_on_typescript_too() -> None:
    old = "const a = 1;\nrender(a);\n"
    new = "const b = 1;\nrender(b);\n"
    spec = detect_op(old, new, "t.ts")
    assert spec is not None and spec.kind != "rename_symbol"


# --------------------------------------------------------------------------- #
# ordering: the member path must not steal Python's
# --------------------------------------------------------------------------- #


def test_a_python_insertion_still_goes_through_the_python_detector() -> None:
    """``x = 1`` looks like a member, but a ``.py`` file has no member form."""
    old = "a = 0\n"
    new = "a = 0\nb = 1\n"
    spec = detect_op(old, new, "m.py")
    assert spec is not None and spec.kind == "insert_statement"
    assert spec.props["n_stmts"] == "1"
    assert "key" not in spec.props
    assert apply_op(old, spec) == new


def test_python_files_do_not_get_the_member_detectors_at_all() -> None:
    from sylphae.ops.detect import _DETECTORS, _detectors_for

    assert _detectors_for("m.py") == _DETECTORS
    assert _detectors_for("t.css")[:2] != _DETECTORS[:2]
    assert len(_detectors_for("t.css")) == len(_DETECTORS) + 2


def test_detect_op_still_returns_none_when_nothing_changed() -> None:
    assert detect_op("a = 1\n", "a = 1\n", "t.css") is None
