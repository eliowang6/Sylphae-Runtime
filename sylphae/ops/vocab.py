"""The closed vocabulary of structured edit operations.

This list is the ceiling of the whole approach: only operations that are
*detectable from a diff* and *deterministically executable* are in it.

Ids ``0..15`` are the original frozen vocabulary (``INTERFACES.md`` section 1 as
first written) and are **unchanged**.  The eight kinds ``16..23`` were added when
the coverage measurement showed the vocabulary was too coarse: ``detect_op``
classified a whole file change as one op, and real commits mix several small
edits.  ``detect_sequence`` (see :mod:`sylphae.ops.detect`) decomposes a change into
a sequence of ops drawn from this same list.  Appending rather than reordering
keeps every previously written op id valid, which matters because the corpus and
the model's op head index into this list.

Parameterised vs generative
---------------------------

Not all structured ops are equal, and the difference is the whole point of the
reflex-agent design.  A **parameterised** op is one the model can emit by
*choosing a short value or name* — a literal, an identifier, a keyword-argument
name, a dict key.  A **generative** op requires the model to *write new code
text*, which a small model does badly.  :data:`PARAMETERIZED_OPS` and
:data:`GENERATIVE_OPS` partition the executable kinds; the two coverages are
reported separately so the parameterised number — the real domain of the reflex
agent — is never hidden behind the larger, easier structured number.
"""

from __future__ import annotations

#: Frozen, ordered vocabulary.  Index in this list *is* the op id.
OP_KINDS: list[str] = [
    "no_op",              # 0
    "replace_literal",    # 1  a numeric/string literal changes value, structure identical
    "rename_symbol",      # 2  an identifier is renamed consistently
    "change_condition",   # 3  a line inside if/while/assert changes
    "modify_call_args",   # 4  arguments of a call change
    "change_default",     # 5  a parameter default value changes
    "add_import",         # 6
    "remove_import",      # 7
    "add_guard",          # 8  an early-return / raise guard is inserted
    "insert_statement",   # 9  statements inserted, nothing removed
    "delete_statement",   # 10 statements removed, nothing inserted
    "add_decorator",      # 11
    "change_return",      # 12
    "modify_string",      # 13 only a message string changes
    "wrap_try_except",    # 14
    "freeform_patch",     # 15 anything else — the escape hatch, and the coverage loss
    # -- added with the sequence detector; ids 0..15 above are unchanged --
    "replace_line",       # 16 one whole line is replaced by another whole line
    "add_call_arg",       # 17 an argument line is added to a multi-line call
    "remove_call_arg",    # 18 an argument line is removed from a multi-line call
    "add_dict_entry",     # 19 a ``key: value`` entry is added to a dict literal
    "remove_dict_entry",  # 20 a ``key: value`` entry is removed from a dict literal
    "wrap_block",         # 21 a region is indented under a newly inserted header line
    "modify_string_multi",  # 22 several string literals change in one hunk
    "reindent",           # 23 only indentation changes, content identical
    "add_list_element",   # 24 an element line is added to a list/set/tuple literal
    "remove_list_element",  # 25 an element line is removed from a list/set/tuple literal
]

OP_TO_ID: dict[str, int] = {kind: i for i, kind in enumerate(OP_KINDS)}
ID_TO_OP: dict[int, str] = {i: kind for i, kind in enumerate(OP_KINDS)}
N_OPS: int = len(OP_KINDS)

#: The escape hatch.  Examples labelled with this kind are the coverage loss and
#: are *not* replayable by the structured executor.
FREE_FORM_KIND: str = "freeform_patch"

#: Kinds that ``detect_op`` never emits (kept in the vocabulary because the model
#: head needs a class for "the edit changes nothing").
NEVER_DETECTED: frozenset[str] = frozenset({"no_op"})

#: Kinds whose ``apply_op`` is implemented; i.e. the structured, replayable set.
EXECUTABLE_KINDS: frozenset[str] = frozenset(OP_KINDS) - {FREE_FORM_KIND, "no_op"}

#: Kinds emitted for a *pure insertion* of lines.  ``OpSpec.span`` for these is
#: ``(p, p)`` meaning "insert before 1-based original line p"; ``p`` may legally
#: be ``len(old_lines) + 1`` (append at end).
INSERTION_KINDS: frozenset[str] = frozenset(
    {
        "add_import",
        "add_guard",
        "add_decorator",
        "insert_statement",
        "add_call_arg",
        "add_dict_entry",
        "add_list_element",
    }
)

#: Kinds emitted for a *pure deletion* of lines.  ``OpSpec.span`` covers the
#: deleted original lines.
DELETION_KINDS: frozenset[str] = frozenset(
    {
        "remove_import",
        "delete_statement",
        "remove_call_arg",
        "remove_dict_entry",
        "remove_list_element",
    }
)

#: Kinds that replace a contiguous run of original lines with new text.
REPLACEMENT_KINDS: frozenset[str] = frozenset(
    {
        "replace_literal",
        "change_condition",
        "modify_call_args",
        "change_default",
        "change_return",
        "modify_string",
        "replace_line",
        "modify_string_multi",
        "reindent",
    }
)

#: Kinds that *wrap* an existing region under a newly inserted header line.  The
#: span covers the original region; the header and indentation come from props.
WRAP_KINDS: frozenset[str] = frozenset({"wrap_try_except", "wrap_block"})

#: The **parameterised** ops: the model emits one by choosing a short value or
#: name (a literal, an identifier, a keyword-argument name, a dict key).  This is
#: the "reflex" domain the design is about.  Everything in this set is also in
#: :data:`EXECUTABLE_KINDS`.
PARAMETERIZED_OPS: frozenset[str] = frozenset(
    {
        "replace_literal",     # choose a literal value
        "change_default",      # choose a literal value
        "modify_string",       # choose a string literal
        "rename_symbol",       # choose an identifier
        "add_call_arg",        # choose a keyword-argument name (+ short value)
        "remove_call_arg",     # choose a keyword-argument name
        "add_dict_entry",      # choose a dict key (+ short value)
        "remove_dict_entry",   # choose a dict key
        "add_list_element",    # choose a list/set element value
        "remove_list_element",  # choose a list/set element value
    }
)

#: The **generative** ops: the model must write new code text to emit one.  A
#: small model does this badly, which is why it is measured separately.
GENERATIVE_OPS: frozenset[str] = EXECUTABLE_KINDS - PARAMETERIZED_OPS


def kind_id(kind: str) -> int:
    """Return the frozen integer id of ``kind``, raising on anything unknown."""
    try:
        return OP_TO_ID[kind]
    except KeyError:
        raise ValueError(
            f"unknown op kind {kind!r}; must be one of {OP_KINDS}"
        ) from None


# -- vocabulary self-consistency (a broken partition would silently change every
# -- reported number, so it is checked at import time rather than trusted) ------

if PARAMETERIZED_OPS | GENERATIVE_OPS != EXECUTABLE_KINDS:  # pragma: no cover
    raise AssertionError("PARAMETERIZED_OPS and GENERATIVE_OPS must partition EXECUTABLE_KINDS")
if PARAMETERIZED_OPS & GENERATIVE_OPS:  # pragma: no cover
    raise AssertionError("PARAMETERIZED_OPS and GENERATIVE_OPS must be disjoint")
if FREE_FORM_KIND in PARAMETERIZED_OPS or FREE_FORM_KIND in GENERATIVE_OPS:  # pragma: no cover
    raise AssertionError("freeform_patch is neither parameterised nor generative")
if "no_op" in PARAMETERIZED_OPS or "no_op" in GENERATIVE_OPS:  # pragma: no cover
    raise AssertionError("no_op is neither parameterised nor generative")
