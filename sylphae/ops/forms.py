"""Language-general line *forms* for the non-Python half of the polyglot surface.

Why this module exists
----------------------

:mod:`sylphae.ops.detect` was written for Python: its evidence comes from ``ast`` and
from ``tokenize``.  On a ``.css`` / ``.ts`` / ``.vue`` file neither of those is a
lexer for the language in front of it, and two things went wrong at once.

* **A change the producer side already knows how to describe was not recognisable
  on the way back.**  :func:`sylphae.ops.polyglot.add_css_properties` emits an
  ``insert_statement`` for a CSS property injection, but feeding the resulting
  diff to :func:`~sylphae.ops.detect.detect_op` returned ``freeform_patch`` with
  ``reason="insert_not_a_known_statement"``.  Producer and detector disagreed
  about a change they had both seen.
* **The detectors that did fire on those files fired for the wrong reason.**  A
  CSS ``color: red`` -> ``color: blue`` change was labelled ``rename_symbol``,
  because ``tokenize`` reports ``red`` and ``blue`` as ``NAME`` tokens and the
  file's token count was unchanged.  But a CSS colour keyword is a *value*; there
  is no symbol anywhere in that file to rename.

The fix is not "teach :mod:`~sylphae.ops.detect` about CSS".  It is to name the
structure these languages actually share and test for *that*::

    a **member line** -- ``key <separator> value`` with an optional terminator --
    inside a **block** -- the enclosing brace pair, or the enclosing indentation
    run.

Both halves are read off the source rather than assumed:

* the separator set, terminator set and comment prefixes come from the file's
  extension (:data:`SYNTAXES`), and
* "this line belongs to that block" is *proven* by its indentation matching a
  sibling the block already contains, or by following a line that opens the block
  (:func:`_proved_by_predecessor`).

A language whose lines do not have this shape yields no member and keeps falling
through to ``freeform_patch``, which is the honest outcome.  Nothing here is
CSS-specific: the CSS rule and the YAML rule are the same rule with different
table rows.

What is deliberately *not* modelled
-----------------------------------

* **Brace-bearing values.**  A member's value may not contain ``{`` or ``}``.
  That gives up TS object members whose value is an inline function
  (``onClick: () => { ... }``), and it buys the far more common case in return:
  a CSS pseudo-selector header such as ``a:hover {`` is *not* mistaken for a
  member named ``a`` with value ``hover {``.  Selector headers are everywhere;
  inline arrow-function members are not.
* **Shell prefixes.**  ``export FOO=bar`` in a ``.env`` file is left to
  ``freeform_patch``.  Accepting an optional leading word before the key would
  have to guess where the key starts, and guessing is what this module exists to
  avoid.
* **Blocks opened and closed on one line.**  A member must be indented *deeper*
  than the line that opens its block; a single-line ``.a { color: red; }`` has no
  interior, so nothing can be shown to live inside it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

__all__ = [
    "Member",
    "MemberSyntax",
    "SYNTAXES",
    "syntax_for",
    "has_python_lexer",
    "is_comment",
    "indent_of",
    "content_of",
    "extract_members",
    "inserted_members",
    "deleted_members",
]


# --------------------------------------------------------------------------- #
# line helpers
# --------------------------------------------------------------------------- #

#: Line terminators, in the order ``str.rstrip`` would consume them.
_NEWLINE_CHARS = "\r\n"


def content_of(line: str) -> str:
    """``line`` without its terminator.  Never touches interior characters."""
    return line.rstrip(_NEWLINE_CHARS)


def indent_of(content: str) -> str:
    """The leading run of spaces and tabs of a terminator-free line."""
    return content[: len(content) - len(content.lstrip(" \t"))]


# --------------------------------------------------------------------------- #
# syntax table
# --------------------------------------------------------------------------- #

#: A block delimited by a brace pair: the members are the lines between ``{`` and
#: its matching ``}``, one indentation level deeper than the opening line.
_BRACE = "brace"

#: A block delimited by indentation alone: a line is a member of the run of
#: equally-indented siblings it sits in, whatever surrounds them.
_INDENT = "indent"

#: A key: a quoted string, or a bare token.  The bare form allows a leading
#: ``-`` / ``@`` / ``$`` because CSS custom properties (``--gap``), at-rules
#: (``@apply``) and SCSS variables (``$gap``) all start that way.
_KEY = r"""(?:"[^"\n]*"|'[^'\n]*'|[A-Za-z_@$-][A-Za-z0-9_@$.\-]*)"""

#: Comment prefixes shared by the C-family brace languages.
_C_COMMENTS: tuple[str, ...] = ("//", "/*", "*/", "/*", "*")


@dataclass(frozen=True)
class MemberSyntax:
    """How one language family spells ``key <separator> value`` on a single line."""

    family: str
    extensions: tuple[str, ...]
    separators: tuple[str, ...]
    terminators: tuple[str, ...]
    comment_prefixes: tuple[str, ...]

    def member(self, line: str) -> "Member | None":
        """Read ``line`` as a member, or return ``None`` if it is not one.

        Accepts a line with or without its terminator.  Returns ``None`` for a
        blank line, a comment, a line whose value is empty (a block header such as
        YAML's ``key:``), and for any value containing a brace.
        """
        content = content_of(line)
        stripped = content.strip()
        if not stripped or is_comment(stripped, self):
            return None

        match = _pattern_for(self).match(content)
        if match is None:
            return None
        indent, key, value = match.group(1), match.group(2), match.group(3).strip()
        if not value or "{" in value or "}" in value:
            return None

        terminator = ""
        for candidate in self.terminators:
            if candidate and value.endswith(candidate):
                value = value[: -len(candidate)].rstrip()
                terminator = candidate
                break
        if not value:
            return None
        return Member(indent=indent, key=key, value=value, terminator=terminator)


@dataclass(frozen=True)
class Member:
    """One ``key <separator> value`` line, with the indentation it carries."""

    indent: str
    key: str
    value: str
    terminator: str


#: The registered member syntaxes, most specific extension first.  Adding a
#: language here is the whole of "teaching ``detect`` about it" -- there is no
#: per-language code below this table.
SYNTAXES: tuple[MemberSyntax, ...] = (
    # `prop: value;` inside `selector { ... }` (and SCSS/LESS nesting).
    MemberSyntax(
        _BRACE,
        (".css", ".scss", ".less"),
        (":",),
        (";",),
        _C_COMMENTS,
    ),
    # `"key": value,` inside `{ ... }`.
    MemberSyntax(
        _BRACE,
        (".json",),
        (":",),
        (",",),
        ("//",),
    ),
    # `key: value,` object members inside a TS/JS/SFC brace block.
    MemberSyntax(
        _BRACE,
        (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue", ".svelte"),
        (":",),
        (",",),
        _C_COMMENTS,
    ),
    # `key = value` -- TOML, at any depth; `[table]` headers are not members.
    MemberSyntax(
        _INDENT,
        (".toml",),
        ("=",),
        ("",),
        ("#",),
    ),
    # `key: value` -- YAML.
    MemberSyntax(
        _INDENT,
        (".yaml", ".yml"),
        (":",),
        ("",),
        ("#",),
    ),
    # `KEY=value` -- dotenv, INI, and the .properties family.
    MemberSyntax(
        _INDENT,
        (".env", ".ini", ".cfg", ".conf", ".properties", ".editorconfig"),
        ("=",),
        ("",),
        ("#", ";"),
    ),
)

_BY_EXTENSION: dict[str, MemberSyntax] = {
    ext: syntax for syntax in SYNTAXES for ext in syntax.extensions
}

#: Extensions whose token stream ``tokenize`` actually lexes.  ``rename_symbol``
#: is restricted to these: see :func:`has_python_lexer`.
_PYTHON_EXTENSIONS: frozenset[str] = frozenset({".py", ".pyi", ".pyw"})


def syntax_for(path: str) -> MemberSyntax | None:
    """The member syntax of ``path``, or ``None`` if its lines are not members.

    ``None`` is returned for Python (which has its own detectors), and for any
    extension not in :data:`SYNTAXES` -- HTML, SQL, Go, Rust and the rest.  A file
    whose lines are not ``key <separator> value`` simply has no member form to
    recognise, which is a statement about the language and not a limitation of
    the caller.
    """
    ext = os.path.splitext(path)[1].lower()
    return _BY_EXTENSION.get(ext)


def has_python_lexer(path: str) -> bool:
    """Does the toolchain have a real lexer for ``path``?

    ``tokenize`` is a Python lexer.  Everywhere else in the polyglot surface it is
    being used as a stand-in, which is safe for *literal* evidence -- a changed
    token position that holds a number in both revisions is a changed number in
    any language -- but not for *identifier* evidence.  ``rename_symbol`` claims
    an identifier was renamed consistently, and its applier rewrites every
    occurrence in the file with no scope awareness; manufacturing that claim from
    a lexer we do not have is how a CSS colour change becomes a symbol rename.

    An unknown or extension-less path is treated as Python, which is what
    :func:`~sylphae.ops.detect.detect_op` has always assumed for paths such as
    ``"inline"``.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == "":
        return True
    return ext in _PYTHON_EXTENSIONS


# --------------------------------------------------------------------------- #
# patterns
# --------------------------------------------------------------------------- #

_PATTERN_CACHE: dict[int, re.Pattern[str]] = {}


def _pattern_for(syntax: MemberSyntax) -> re.Pattern[str]:
    """``^(indent)(key)(separator)(value)$`` for one syntax, built once."""
    cached = _PATTERN_CACHE.get(id(syntax))
    if cached is not None:
        return cached
    separators = "|".join(
        re.escape(s) for s in sorted(syntax.separators, key=len, reverse=True)
    )
    pattern = re.compile(
        rf"^([ \t]*)({_KEY})[ \t]*(?:{separators})[ \t]*(.*)$",
        re.DOTALL,
    )
    _PATTERN_CACHE[id(syntax)] = pattern
    return pattern


def is_comment(stripped: str, syntax: MemberSyntax) -> bool:
    """Is a terminator-free, stripped line a comment for ``syntax``?

    The ``*`` case carries a structural guard rather than a keyword list: a line
    starting with ``*`` is treated as a block-comment continuation only when it
    contains no ``{``.  Without that guard CSS's universal selector (``* {``)
    would be read as a comment, and blocks it opens would lose their opener.
    """
    for prefix in syntax.comment_prefixes:
        if not stripped.startswith(prefix):
            continue
        if prefix == "*" and "{" in stripped:
            return False
        return True
    return False


# --------------------------------------------------------------------------- #
# block membership
# --------------------------------------------------------------------------- #


def _is_block_opener(content: str, syntax: MemberSyntax) -> bool:
    """Could ``content`` be the line that owns the block a deeper line sits in?"""
    if syntax.family == _BRACE:
        return "{" in content
    stripped = content.strip()
    # A YAML container header (`key:` with nothing after it) or a sequence item.
    return stripped.endswith(":") or stripped.startswith("- ")


def _proved_by_predecessor(
    lines: list[str], prev_index: int, indent: str, syntax: MemberSyntax
) -> bool:
    """Is the line at ``prev_index`` sibling or opener evidence for ``indent``?

    Walks backwards over blanks and comments to the first line that carries
    content, and then accepts exactly two things:

    * a **member at the same indentation** -- the block already has a sibling
      here, so a line at this depth belongs to it; or
    * a **block opener at a shallower indentation** -- the insert point is the
      first member of a block that already exists.

    Anything else is refused.  In particular a non-opener at a *shallower*
    indentation means the line is deeper than its context allows, which is a
    formatting error rather than a member.
    """
    index = prev_index
    while index >= 0:
        content = content_of(lines[index])
        stripped = content.strip()
        if stripped == "" or is_comment(stripped, syntax):
            index -= 1
            continue
        candidate_indent = indent_of(content)
        if candidate_indent == indent:
            return syntax.member(content) is not None
        if len(candidate_indent) < len(indent) and _is_block_opener(content, syntax):
            return True
        return False
    # Nothing above: a brace block cannot start at the top of the file, but for
    # the indentation family the file itself is a container -- at depth zero.
    return syntax.family == _INDENT and indent == ""


def _proved_by_successor(
    lines: list[str], next_index: int, indent: str, syntax: MemberSyntax
) -> bool:
    """The mirror of :func:`_proved_by_predecessor`, for a deletion."""
    index = next_index
    while index < len(lines):
        content = content_of(lines[index])
        stripped = content.strip()
        if stripped == "" or is_comment(stripped, syntax):
            index += 1
            continue
        candidate_indent = indent_of(content)
        if candidate_indent == indent:
            return syntax.member(content) is not None
        if len(candidate_indent) < len(indent):
            if syntax.family == _BRACE:
                # The block's closing brace, or the opener of the next block.
                return stripped.startswith("}") or "{" in stripped
            return _is_block_opener(content, syntax)
        return False
    return syntax.family == _INDENT and indent == ""


def extract_members(text: str, syntax: MemberSyntax) -> list[Member] | None:
    """Read ``text`` as one or more member lines at a single indentation.

    Returns ``None`` -- never a partial reading -- when there is no member to
    read, when any non-blank line fails to parse as one, or when the lines do not
    all sit at the same indentation.  A mixed-indentation insert is not one
    structural claim, and saying so is better than picking the first line and
    hoping.
    """
    members: list[Member] = []
    for line in text.splitlines():
        if content_of(line).strip() == "":
            continue
        member = syntax.member(line)
        if member is None:
            return None
        if members and member.indent != members[0].indent:
            return None
        members.append(member)
    return members or None


def inserted_members(
    path: str, old_lines: list[str], at: int, text: str
) -> list[Member] | None:
    """Members inserted at 1-based position ``at`` of ``old_lines``, or ``None``.

    ``at`` may be ``len(old_lines) + 1`` (append at end), matching the
    ``OpSpec.span == (p, p)`` convention of :data:`~sylphae.ops.vocab.INSERTION_KINDS`.
    """
    syntax = syntax_for(path)
    if syntax is None:
        return None
    members = extract_members(text, syntax)
    if members is None:
        return None
    if not _proved_by_predecessor(old_lines, at - 2, members[0].indent, syntax):
        return None
    return members


def deleted_members(
    path: str, old_lines: list[str], a: int, b: int
) -> list[Member] | None:
    """Members deleted from 1-based inclusive span ``a..b``, or ``None``.

    Proven from *either* side: the line before the region, or the line after it.
    A deletion that removes the only member of a block has no sibling left, which
    is exactly why the successor walk exists.
    """
    syntax = syntax_for(path)
    if syntax is None:
        return None
    if not (1 <= a <= b <= len(old_lines)):
        return None
    members = extract_members("".join(old_lines[a - 1 : b]), syntax)
    if members is None:
        return None
    indent = members[0].indent
    if _proved_by_predecessor(old_lines, a - 2, indent, syntax):
        return members
    if _proved_by_successor(old_lines, b, indent, syntax):
        return members
    return None
