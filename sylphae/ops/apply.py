"""``apply_op`` — deterministic replay of a structured edit.

Every applier here is *total* with respect to its span: it validates the span
against the source it is given and raises :class:`OpApplyError` when the span is
out of range.  It never clamps, never truncates, never guesses.  That property is
what makes the round-trip invariant meaningful.

The appliers that carry an ``old_text`` / ``text`` prop compare the original
bytes at the span against that prop and raise if they differ, so a spec built
from a *different* revision of the file is rejected instead of silently applied
to the wrong place.
"""

from __future__ import annotations

import io
import tokenize

from .spec import OpApplyError, OpSpec
from .vocab import (
    DELETION_KINDS,
    FREE_FORM_KIND,
    INSERTION_KINDS,
    REPLACEMENT_KINDS,
    WRAP_KINDS,
)

__all__ = ["apply_op", "apply_sequence", "OpApplyError"]


# --------------------------------------------------------------------------- #
# line helpers
# --------------------------------------------------------------------------- #


def _lines(src: str) -> list[str]:
    """Split keeping terminators.  ``"".join(_lines(s)) == s`` for every ``s``."""
    return src.splitlines(keepends=True)


def _check_region(a: int, b: int, n: int) -> None:
    """Validate a 1-based inclusive span over ``n`` lines.  Raise, never clamp."""
    if not isinstance(a, int) or not isinstance(b, int) or isinstance(a, bool) \
            or isinstance(b, bool):
        raise OpApplyError(f"span must be two ints; got ({a!r}, {b!r})")
    if a < 1 or b < a or b > n:
        raise OpApplyError(
            f"span ({a}, {b}) is out of range for a source with {n} line(s)"
        )


def _check_insert_point(p: int, n: int) -> None:
    """Validate an insertion point over ``n`` lines: ``1 <= p <= n + 1``."""
    if not isinstance(p, int) or isinstance(p, bool):
        raise OpApplyError(f"insertion line must be an int; got {p!r}")
    if p < 1 or p > n + 1:
        raise OpApplyError(
            f"insertion line {p} is out of range for a source with {n} line(s) "
            f"(legal range 1..{n + 1})"
        )


def _prop(spec: OpSpec, key: str) -> str:
    try:
        value = spec.props[key]
    except KeyError:
        raise OpApplyError(
            f"op {spec.kind!r} requires props[{key!r}]; props are {sorted(spec.props)}"
        ) from None
    if not isinstance(value, str):
        raise OpApplyError(f"props[{key!r}] must be a str; got {type(value).__name__}")
    return value


def _int_prop(spec: OpSpec, key: str) -> int:
    raw = _prop(spec, key)
    try:
        return int(raw)
    except ValueError:
        raise OpApplyError(f"props[{key!r}] must be an integer string; got {raw!r}") \
            from None


def _expect_span(spec: OpSpec, a: int, b: int) -> None:
    if tuple(spec.span) != (a, b):
        raise OpApplyError(
            f"op {spec.kind!r} expects span ({a}, {b}) to match its props, "
            f"but the spec carries {tuple(spec.span)}"
        )


# --------------------------------------------------------------------------- #
# appliers
# --------------------------------------------------------------------------- #


def _apply_insert(old_src: str, spec: OpSpec) -> str:
    lines = _lines(old_src)
    n = len(lines)
    p = _int_prop(spec, "line")
    _expect_span(spec, p, p)
    _check_insert_point(p, n)
    text = _prop(spec, "text")
    if text == "":
        raise OpApplyError(f"op {spec.kind!r} has an empty 'text' prop; nothing to insert")
    inserted = _lines(text)
    if "".join(inserted) != text:
        raise OpApplyError("'text' prop does not survive a line split; refusing to insert")
    return "".join(lines[: p - 1] + inserted + lines[p - 1 :])


def _apply_delete(old_src: str, spec: OpSpec) -> str:
    lines = _lines(old_src)
    n = len(lines)
    a, b = spec.span
    _check_region(a, b, n)
    text = _prop(spec, "text")
    found = "".join(lines[a - 1 : b])
    if found != text:
        raise OpApplyError(
            f"op {spec.kind!r}: original lines {a}..{b} are {found!r}, "
            f"not the recorded {text!r}"
        )
    return "".join(lines[: a - 1] + lines[b:])


def _apply_block_replace(old_src: str, spec: OpSpec) -> str:
    lines = _lines(old_src)
    n = len(lines)
    a, b = spec.span
    _check_region(a, b, n)
    old_text = _prop(spec, "old_text")
    new_text = _prop(spec, "new_text")
    found = "".join(lines[a - 1 : b])
    if found != old_text:
        raise OpApplyError(
            f"op {spec.kind!r}: original lines {a}..{b} are {found!r}, "
            f"not the recorded {old_text!r}"
        )
    replacement = _lines(new_text)
    if "".join(replacement) != new_text:
        raise OpApplyError("'new_text' prop does not survive a line split")
    return "".join(lines[: a - 1] + replacement + lines[b:])


def _name_positions(src: str, name: str) -> list[tuple[int, int, int]]:
    """``(row, col_start, col_end)`` of every NAME token equal to ``name``."""
    out: list[tuple[int, int, int]] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.NAME and tok.string == name:
                out.append((tok.start[0], tok.start[1], tok.end[1]))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError) as exc:
        raise OpApplyError(f"cannot tokenize source for rename_symbol: {exc}") from None
    return out


def _apply_rename_symbol(old_src: str, spec: OpSpec) -> str:
    old_name = _prop(spec, "old")
    new_name = _prop(spec, "new")
    count = _int_prop(spec, "count")
    if not old_name.isidentifier() or not new_name.isidentifier():
        raise OpApplyError(
            f"rename_symbol names must be identifiers; got {old_name!r} -> {new_name!r}"
        )
    if old_name == new_name:
        raise OpApplyError("rename_symbol with identical names is a no-op, not an edit")

    positions = _name_positions(old_src, old_name)
    if len(positions) != count:
        raise OpApplyError(
            f"rename_symbol recorded {count} occurrence(s) of {old_name!r} "
            f"but the source has {len(positions)}"
        )

    lines = _lines(old_src)
    by_row: dict[int, list[tuple[int, int]]] = {}
    for row, cs, ce in positions:
        by_row.setdefault(row, []).append((cs, ce))
    for row, spans in by_row.items():
        if not 1 <= row <= len(lines):
            raise OpApplyError(f"token row {row} is outside the source")
        line = lines[row - 1]
        for cs, ce in sorted(spans, reverse=True):
            if cs > ce or ce > len(line):
                raise OpApplyError(f"token columns ({cs}, {ce}) exceed line {row}")
            line = line[:cs] + new_name + line[ce:]
        lines[row - 1] = line
    return "".join(lines)


def _apply_wrap_try_except(old_src: str, spec: OpSpec) -> str:
    lines = _lines(old_src)
    n = len(lines)
    a, b = spec.span
    _check_region(a, b, n)
    n_body = _int_prop(spec, "n_body")
    if b - a + 1 != n_body:
        raise OpApplyError(
            f"wrap_try_except span ({a}, {b}) covers {b - a + 1} line(s) "
            f"but n_body is {n_body}"
        )
    try_line = _prop(spec, "try_line")
    pad = _prop(spec, "pad")
    handler = _prop(spec, "handler")
    if len(_lines(try_line)) != 1:
        raise OpApplyError("wrap_try_except 'try_line' must be exactly one line")
    if handler == "":
        raise OpApplyError("wrap_try_except needs a non-empty 'handler'")

    body = lines[a - 1 : b]
    padded = [pad + ln if ln.strip() != "" else ln for ln in body]
    return "".join(
        lines[: a - 1] + [try_line] + padded + _lines(handler) + lines[b:]
    )


def _apply_wrap_block(old_src: str, spec: OpSpec) -> str:
    """Indent the span's lines under a freshly inserted single header line.

    ``wrap_try_except``'s generic sibling: ``header`` is the ``with`` / ``if`` /
    ``for`` / ``while`` line that is inserted before the region, ``pad`` is the
    indentation it adds, and ``n_body`` is the number of original lines consumed.
    There is deliberately no trailing payload (no handler / ``else``): a wrap
    whose header has a trailing clause is not describable by this op.
    """
    lines = _lines(old_src)
    n = len(lines)
    a, b = spec.span
    _check_region(a, b, n)
    n_body = _int_prop(spec, "n_body")
    if b - a + 1 != n_body:
        raise OpApplyError(
            f"wrap_block span ({a}, {b}) covers {b - a + 1} line(s) "
            f"but n_body is {n_body}"
        )
    header = _prop(spec, "header")
    pad = _prop(spec, "pad")
    if len(_lines(header)) != 1:
        raise OpApplyError("wrap_block 'header' must be exactly one line")
    if not header.rstrip("\r\n").rstrip().endswith(":"):
        raise OpApplyError(
            f"wrap_block 'header' must be a compound-statement header ending in ':'; "
            f"got {header!r}"
        )
    if pad == "":
        raise OpApplyError(
            "wrap_block 'pad' must be non-empty; a wrap with no added indentation "
            "would not be a wrap"
        )

    body = lines[a - 1 : b]
    padded = [pad + ln if ln.strip() != "" else ln for ln in body]
    return "".join(lines[: a - 1] + [header] + padded + lines[b:])


_APPLIERS = {
    **{kind: _apply_insert for kind in INSERTION_KINDS},
    **{kind: _apply_delete for kind in DELETION_KINDS},
    **{kind: _apply_block_replace for kind in REPLACEMENT_KINDS},
    "rename_symbol": _apply_rename_symbol,
    "wrap_try_except": _apply_wrap_try_except,
    "wrap_block": _apply_wrap_block,
}

# A wrap op is registered by kind, not by the set, so a new wrap kind cannot be
# silently added to the vocabulary without an applier.
if WRAP_KINDS != frozenset(k for k in _APPLIERS if k.startswith("wrap_")):  # pragma: no cover
    raise AssertionError(
        f"WRAP_KINDS {sorted(WRAP_KINDS)} does not match the registered wrap appliers"
    )


def apply_op(old_src: str, spec: OpSpec) -> str:
    """Reproduce the edited source.  Raises :class:`OpApplyError` if ``spec`` does not fit.

    ``freeform_patch`` is deliberately *not* executable.  The whole point of the
    vocabulary is that a structured op is replayable and a freeform patch is not;
    giving freeform an applier would make the coverage measurement meaningless.
    Callers that meet a freeform spec are expected to escalate.

    Raises:
        OpApplyError: span out of range, recorded original text does not match the
            source, malformed props, unknown kind, or a ``freeform_patch`` spec.
    """
    if not isinstance(spec, OpSpec):
        raise OpApplyError(f"apply_op expects an OpSpec; got {type(spec).__name__}")
    if not isinstance(old_src, str):
        raise OpApplyError(f"apply_op expects a str source; got {type(old_src).__name__}")

    if spec.kind == FREE_FORM_KIND:
        raise OpApplyError(
            "freeform_patch is not replayable by the structured executor; "
            "escalate instead of applying"
        )
    if spec.kind == "no_op":
        # A no_op is well-defined: the source is unchanged.  Still validate the
        # span so an out-of-range no_op is rejected rather than clamped.
        _check_region(spec.span[0], spec.span[1], len(_lines(old_src)) or 1)
        return old_src

    applier = _APPLIERS.get(spec.kind)
    if applier is None:
        raise OpApplyError(f"no applier registered for op kind {spec.kind!r}")
    return applier(old_src, spec)


def apply_sequence(old_src: str, seq) -> str:
    """Apply an ordered op sequence produced by ``detect_sequence``.

    ``seq`` is emitted **top-to-bottom** and every op's span is in the
    **original** source, so the sequence is applied **back-to-front**: the lowest
    op goes first, and because it can only change lines at or below its own span
    it cannot shift the line numbers of the ops above it.  Equivalently,
    ``apply_sequence(old, seq) == functools.reduce(apply_op, reversed(seq), old)``.

    Raises:
        OpApplyError: if any op does not fit the source at the point it is
            applied, or if a spec is a ``freeform_patch`` (not replayable).
    """
    if not isinstance(old_src, str):
        raise OpApplyError(f"apply_sequence expects a str source; got {type(old_src).__name__}")
    result = old_src
    for spec in reversed(list(seq)):
        result = apply_op(result, spec)
    return result
