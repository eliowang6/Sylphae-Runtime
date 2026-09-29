"""Placement evidence: is a patch landing where the user was pointing?

The patch builder does two things that can disagree.  Candidate ranking decides *where
the user is pointing* (a path, a snippet, sometimes a line).  Patch synthesis separately
decides *which line to edit*.  When those two disagree, the patch edits something the
user never asked about -- and nothing downstream notices, because the file still parses
and the tests still pass.

Two independent kinds of evidence are offered here, and the patch builder treats them as
alternatives rather than as a single gate.  Both are deliberately blind to file
extension and import no parser, so a language added tomorrow needs no new code:

``same_entity``
    Syntactic evidence.  Do two lines sit in the same block?  Keyed off textual block
    structure: brace nesting (``{}``) for CSS/SCSS, TS/JS/TSX, Vue ``<script>``, Go,
    Rust, Java, C/C++; indentation for Python and YAML.

``symbols_in``
    Semantic evidence.  Do two lines name the same identifier?  This is what keeps a
    legitimate cross-entity edit alive: a constant declared at the top of a file is
    routinely edited from the struct field that references it, so the two lines are in
    different blocks yet unmistakably the same subject.

Requiring syntactic agreement alone would reject those legitimate edits; requiring
neither lets a patch wander to any keyword-matching line in the file.  The patch builder
asks for one or the other.

The module stays import-light on purpose: it sits on the probe path that AGENTS.md 2.2
holds to a sub-150ms budget.
"""

from __future__ import annotations

import re
from typing import Sequence

__all__ = ["entity_start_line", "entity_end_line", "same_entity", "symbols_in"]

_BLANK = re.compile(r"^\s*$")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _indent_width(line: str) -> int:
    """Visual indent of a line, counting a tab as one column."""
    return len(line) - len(line.lstrip())


def _brace_depths(lines: Sequence[str]) -> list[int]:
    """Brace nesting depth at the *start* of each line.

    Quotes and comments are not tracked.  A brace inside a string literal therefore
    skews the depth, which is acceptable here: the result only needs to tell entities
    apart, and a miscount of this kind is symmetric across the lines being compared.
    """
    depths: list[int] = []
    depth = 0
    for line in lines:
        depths.append(depth)
        depth += line.count("{") - line.count("}")
        if depth < 0:
            depth = 0
    return depths


def _brace_entity(lines: Sequence[str], depths: Sequence[int], line_no: int) -> int | None:
    """Entity start implied by brace structure, or None when braces decide nothing."""
    if not any("{" in line for line in lines):
        return None

    index = line_no - 1
    depth = depths[index]

    if depth > 0:
        for k in range(index - 1, -1, -1):
            if depths[k] < depth:
                return k + 1
        return None

    # Depth zero: the line may still open a block of its own (a CSS selector, a Go func).
    if "{" in lines[index]:
        return line_no
    return None


def _indent_entity(lines: Sequence[str], line_no: int) -> int | None:
    """Entity start implied by indentation, or None when indent decides nothing."""
    index = line_no - 1
    if _BLANK.match(lines[index]):
        return None

    indent = _indent_width(lines[index])
    if indent == 0:
        return None

    for k in range(index - 1, -1, -1):
        if _BLANK.match(lines[k]):
            continue
        if _indent_width(lines[k]) < indent:
            return k + 1
    return None


def entity_start_line(lines: Sequence[str], line_no: int) -> int:
    """Return the 1-based line that opens the entity containing ``line_no``.

    Whichever of the two structural readings is available gets used; when both are,
    the outer one wins.  Preferring the outer boundary keeps the guard conservative:
    it narrows the cases that get rejected rather than inventing boundaries that would
    block edits the user did ask for.

    A line at top level is its own entity, so the function degrades to the identity
    mapping on flat files and never raises.
    """
    if line_no < 1 or line_no > len(lines):
        return line_no

    depths = _brace_depths(lines)
    candidates = [
        candidate
        for candidate in (
            _brace_entity(lines, depths, line_no),
            _indent_entity(lines, line_no),
        )
        if candidate is not None
    ]
    if not candidates:
        return line_no
    return min(candidates)


def same_entity(lines: Sequence[str], line_a: int, line_b: int) -> bool:
    """True when two 1-based lines live in the same syntactic entity."""
    if line_a == line_b:
        return True
    return entity_start_line(lines, line_a) == entity_start_line(lines, line_b)


def symbols_in(text: str) -> set[str]:
    """Lowercased identifiers of three or more characters appearing in ``text``.

    Single- and double-character names are dropped because they collide by chance
    (``i``, ``id``, ``x``) and would let unrelated lines look related.  The length floor
    is the only tuning knob, and it is language-independent.
    """
    return {token.lower() for token in _IDENTIFIER.findall(text)}


def entity_end_line(lines: Sequence[str], line_no: int) -> int:
    """Last line of the entity ``line_no`` opens, or ``line_no`` when it opens none.

    The counterpart to :func:`entity_start_line`, for edits that need somewhere *inside*
    an entity to put new content -- adding a property to a rule block, for instance.
    Brace nesting answers it when the entity is brace-delimited, and indentation answers
    it otherwise, which is the same pair of signals :func:`entity_start_line` uses in
    reverse.
    """
    if not (1 <= line_no <= len(lines)):
        return line_no

    # Measure from the entity's own opening line, not from ``line_no``: a line in the
    # middle of a block contributes no brace of its own, so counting from there lets the
    # depth drift negative and report the next block's closing brace instead.
    start = entity_start_line(lines, line_no)

    depth = 0
    opened = False
    for i in range(start, len(lines) + 1):
        text = lines[i - 1]
        depth += text.count("{") - text.count("}")
        if "{" in text:
            opened = True
        if opened and depth <= 0:
            return i

    if opened:
        # Unbalanced braces: report the last line rather than inventing one.
        return len(lines)

    # No braces: the entity is the indented run that follows.
    base = _indent_width(lines[start - 1])
    last = start
    for i in range(start + 1, len(lines) + 1):
        text = lines[i - 1]
        if _BLANK.match(text):
            continue
        if _indent_width(text) <= base:
            break
        last = i
    return last
