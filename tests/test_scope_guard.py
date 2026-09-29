"""Regression tests for patch placement evidence.

The patch builder picks a *file* from candidate ranking and then picks a *line* to edit.
For a long time the second step ignored the first: the anchor line was merely tried
first in a whole-file scan, so when it could not satisfy the request the search widened
to whichever other line matched next.

That produced a silent wrong edit.  Asked to round a badge's corners, the agent changed
the navbar's ``height: 64px`` to ``height: 8px`` instead -- the located element was
correct and the patch landed elsewhere, so the file still parsed, verification still
passed, and the run reported success.

The fix is not CSS-specific.  Two language-independent kinds of evidence live in
``sylphae/ops/scope.py``: syntactic agreement (same block) and symbolic agreement (same
identifier).  A patch outside the anchor's block must supply at least one of them.

Invariants frozen here:

1. **Cross-entity without symbolic agreement is refused.**  The toolbar-height bug.  The
   request used for this is one that names no property, because a request that names one
   is now served by the named-value lookup -- see ``test_named_values.py`` -- and that
   path constrains itself to the anchor's block rather than scanning the file.
2. **Cross-entity with symbolic agreement still lands.**  A constant declared at file
   scope is legitimately edited from the struct field that references it; requiring
   syntactic agreement alone would have blocked Go, Rust and Python constant edits
   across five benchmark scenarios.
3. **No anchor means no constraint.**  Absence of positional evidence is not evidence
   of a wrong position, so the whole-file scan stays available to text-only callers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sylphae.agent.effort import LowEffort
from sylphae.agent.explorer import CandidateResult
from sylphae.agent.reflex import ReflexAgent
from sylphae.ops.scope import entity_start_line, same_entity, symbols_in

CSS_REPO = """.navbar {
  background-color: #1f2937;
  height: 64px;
}
.badge {
  color: #ffffff;
}
"""

GO_REPO = """package config

import "time"

type ServerConfig struct {
\tPort         int
\tReadTimeout  time.Duration
}

const (
\tDefaultPort        = 8080
\tDefaultMaxRequests = 1000
)

func LoadConfig() ServerConfig {
\treturn ServerConfig{
\t\tPort:         DefaultPort,
\t}
}
"""


def _agent(tmp_path: Path) -> ReflexAgent:
    return ReflexAgent(root=tmp_path, effort=LowEffort)


def _write(tmp_path: Path, rel: str, content: str) -> Path:
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


# --------------------------------------------------------------------------- #
# The primitive: entity boundaries are read from text, not from file extension.
# --------------------------------------------------------------------------- #


def test_entity_boundaries_are_language_independent() -> None:
    css = CSS_REPO.splitlines()
    assert entity_start_line(css, 3) == 1  # inside .navbar
    assert entity_start_line(css, 6) == 5  # inside .badge
    assert same_entity(css, 5, 6) is True  # selector and its declaration
    assert same_entity(css, 5, 3) is False  # two different rules

    go = GO_REPO.splitlines()
    assert same_entity(go, 10, 11) is True  # const block and its entry
    assert entity_start_line(go, 11) == 10  # DefaultPort belongs to `const (`
    assert same_entity(go, 11, 17) is False  # const entry vs referencing field

    python = ["def foo():", "    x = 1", "", "def bar():", "    x = 2"]
    assert same_entity(python, 1, 2) is True
    assert same_entity(python, 1, 5) is False


def test_flat_files_degrade_to_identity() -> None:
    flat = ["SELECT a FROM t", "WHERE b = 1"]
    assert entity_start_line(flat, 1) == 1
    assert entity_start_line(flat, 2) == 2


def test_short_identifiers_are_ignored() -> None:
    # Two-character names collide by chance and must not create false agreement.
    assert symbols_in("i = 1") == set()
    assert symbols_in("x = 1\ny = 2") == set()
    # A qualified name stays whole: TIMEOUT_MS is one symbol, not TIMEOUT plus MS.
    assert symbols_in("const TIMEOUT_MS = 3") == {"const", "timeout_ms"}


# --------------------------------------------------------------------------- #
# Invariant 1: the toolbar-height bug stays refused.
# --------------------------------------------------------------------------- #


def test_cross_entity_patch_without_symbol_agreement_is_refused(tmp_path: Path) -> None:
    _write(tmp_path, "theme.css", CSS_REPO)
    agent = _agent(tmp_path)

    # Deliberately a request with no property name in it.  Once a request names a
    # property ("圆角 8px") the named-value lookup handles it and the block it names is
    # the constraint; the numeric branch below is the path that still scans the file by
    # keyword, and it is the one this invariant guards.
    anchor = CandidateResult(
        path="theme.css", score=0.5, snippet=".badge {", line=5
    )
    spec, new_source = agent._build_patch(
        "把 64 改成 30", "theme.css", "replace_literal", "30", cand=anchor
    )

    # "64" is on the navbar's line 3.  The request says nothing that ties it to the
    # badge, so the numeric branch must not reach across into another rule.
    assert spec is None
    assert new_source is None


def test_named_property_is_added_inside_the_anchor_block(tmp_path: Path) -> None:
    # The same request that used to be refused is now carried out, because the property
    # and its value are both named and the anchor says which rule they belong to.
    _write(tmp_path, "theme.css", CSS_REPO)
    agent = _agent(tmp_path)

    anchor = CandidateResult(
        path="theme.css", score=0.5, snippet=".badge {", line=5
    )
    spec, new_source = agent._build_patch(
        "给 badge 加一个圆角 8px", "theme.css", "replace_literal", "8", cand=anchor
    )

    assert spec is not None
    assert new_source is not None
    lines = new_source.splitlines()
    # Layout is: .navbar { / background / height / } / .badge { / color / } -- the new
    # declaration lands at index 6, i.e. inside .badge and immediately before its
    # closing brace, which gets pushed down to 7.
    assert lines[4] == ".badge {"
    assert lines[5] == "  color: #ffffff;"
    assert lines[6] == "  border-radius: 8px;"
    assert lines[7] == "}"
    # The navbar above is untouched.
    assert lines[2] == "  height: 64px;"


def test_refusal_is_visible_in_the_run_result(tmp_path: Path) -> None:
    _write(tmp_path, "theme.css", CSS_REPO)
    agent = _agent(tmp_path)

    # A candidate whose anchor points at .badge while the only match for the request
    # lives in the navbar rule.
    anchor = CandidateResult(
        path="theme.css", score=0.5, snippet=".badge {", line=5
    )
    spec, _ = agent._build_patch(
        "把 64 改成 30", "theme.css", "replace_literal", "30", cand=anchor
    )

    # No patch built, so the caller's None branch must escalate rather than report a
    # confidence number and exit clean.
    assert spec is None


# --------------------------------------------------------------------------- #
# Invariant 2: legitimate cross-entity edits still land.
# --------------------------------------------------------------------------- #


def test_cross_entity_patch_with_symbol_agreement_lands(tmp_path: Path) -> None:
    _write(tmp_path, "config.go", GO_REPO)
    agent = _agent(tmp_path)

    # The explorer anchors on the struct literal that *references* DefaultPort,
    # while the value to change lives in the const block above it.
    anchor = CandidateResult(
        path="config.go",
        score=0.33,
        snippet="Port:         DefaultPort,",
        line=17,
    )
    spec, new_source = agent._build_patch(
        "把服务端口修改为9000", "config.go", "replace_literal", "9000", cand=anchor
    )

    assert spec is not None, "shared symbol DefaultPort must license this edit"
    assert spec.props["old"] == "8080"
    assert spec.props["new"] == "9000"
    assert new_source is not None and "DefaultPort        = 9000" in new_source


def test_same_entity_patch_is_unaffected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "settings.py",
        "def load():\n    TIMEOUT = 3000\n    return TIMEOUT\n",
    )
    agent = _agent(tmp_path)

    anchor = CandidateResult(
        path="settings.py", score=0.5, snippet="def load():", line=1
    )
    spec, _ = agent._build_patch(
        "把 TIMEOUT 改成 5000", "settings.py", "replace_literal", "5000", cand=anchor
    )

    assert spec is not None
    assert spec.props["old"] == "3000"
    assert spec.props["new"] == "5000"


# --------------------------------------------------------------------------- #
# Invariant 3: no anchor, no constraint.
# --------------------------------------------------------------------------- #


def test_absent_anchor_leaves_the_whole_file_scan_available(tmp_path: Path) -> None:
    _write(tmp_path, "app.ts", "const TIMEOUT_MS = 3000;\n")
    agent = _agent(tmp_path)

    anchorless = CandidateResult(path="app.ts", score=0.4, snippet="", line=None)
    spec, _ = agent._build_patch(
        "把 TIMEOUT_MS 改成 5000",
        "app.ts",
        "replace_literal",
        "5000",
        cand=anchorless,
    )

    assert spec is not None
    assert spec.props["old"] == "3000"
    assert spec.props["new"] == "5000"


@pytest.mark.parametrize("anchor_line", [0, -1, 9999])
def test_out_of_range_anchor_is_treated_as_absent(tmp_path: Path, anchor_line: int) -> None:
    _write(tmp_path, "app.ts", "const TIMEOUT_MS = 3000;\n")
    agent = _agent(tmp_path)

    anchor = CandidateResult(
        path="app.ts", score=0.4, snippet="", line=anchor_line
    )
    spec, _ = agent._build_patch(
        "把 TIMEOUT_MS 改成 5000", "app.ts", "replace_literal", "5000", cand=anchor
    )

    assert spec is not None
    assert spec.props["new"] == "5000"
