"""``detect_op`` / ``detect_sequence`` — classify a real ``(old_src, new_src)``
change into the vocabulary.

Design
------

Detectors are *generators*, not judges.  Each one either produces a candidate
:class:`~sylphae.ops.spec.OpSpec` or returns ``None``; :func:`detect_op` then
**actually replays** the candidate through :func:`~sylphae.ops.apply.apply_op` and
accepts it only if the replay reproduces ``new_src`` byte for byte.  Anything that
does not survive that check falls through to the next detector and, ultimately,
to ``freeform_patch``.

That ordering makes the round-trip invariant structural rather than aspirational:
there is no code path in which ``detect_op`` returns a structured spec whose
replay is not exactly the target.  A buggy detector degrades *coverage* instead
of corrupting *labels*, which is the failure mode we want — a wrong structured
label teaches the model a lie.

``detect_op`` classifies the whole file change as **one** op.  That is too coarse:
real commits routinely mix two to five small edits, and requiring one op to
explain all of them was the core modelling error behind the 25.4% coverage first
measured on Tier A.  :func:`detect_sequence` is the general form — it decomposes
the change per hunk and returns an ordered list of ops; ``detect_op`` is its
length-one special case (a sequence whose single op is not ``freeform_patch``).
The ordering / application contract for sequences (top-to-bottom emission,
original-source spans, back-to-front application) is documented on
:func:`detect_sequence`.

Detector order (first match wins, more specific before more general)::

    add_import / remove_import
    add_decorator
    add_dict_entry / remove_dict_entry
    add_list_element / remove_list_element
    add_call_arg / remove_call_arg
    change_default
    add_guard
    wrap_block
    wrap_try_except
    rename_symbol
    change_return
    change_condition
    modify_string_multi
    modify_string
    modify_call_args
    replace_literal
    reindent
    replace_line
    insert_statement / delete_statement
    freeform_patch

Language dispatch
-----------------

Everything above rests on ``ast`` and ``tokenize``, i.e. on a *Python* lexer.
On a file whose lines are ``key <separator> value`` members -- CSS, JSON, the
TS/JS object family, YAML, TOML, dotenv -- two detectors are therefore run
**first**, ahead of the Python suite:

* ``insert_statement`` / ``delete_statement`` gated on
  :func:`~sylphae.ops.forms.inserted_members` /
  :func:`~sylphae.ops.forms.deleted_members` instead of on Python ``ast``.  This is
  what closes the producer/detector gap: :mod:`sylphae.ops.polyglot` emits
  ``insert_statement`` for a CSS property injection, and until this path existed
  the same change came back as ``freeform_patch``.
* ``rename_symbol`` is **suppressed** for those files, because
  :func:`~sylphae.ops.forms.has_python_lexer` says the tokenizer is not a lexer for
  them.  A CSS ``color: red`` -> ``color: blue`` used to be labelled
  ``rename_symbol`` on the strength of two ``NAME`` tokens; it is now a
  ``replace_line``, which is what it is.

The remaining Python detectors still run for those files, deliberately.  Their
literal-level evidence degrades safely: a change at a token position that holds a
number in both revisions is a changed number in any language, and when the
stand-in tokenizer chokes it returns ``None`` and costs coverage rather than
correctness.  See :mod:`sylphae.ops.forms` for the member primitive itself.

Interpretations this module had to make (``INTERFACES.md`` does not settle them):

* ``replace_literal`` vs ``modify_string``.  Both mean "a literal changed".  A
  change confined to ``STRING`` / f-string tokens is labelled ``modify_string``;
  everything else (numbers, ``True`` / ``False`` / ``None``) is
  ``replace_literal``.  The rule is deterministic rather than contextual.
* ``replace_literal`` and ``modify_string`` fire only when **exactly one** literal
  token differs.  With two or more, the ``old`` / ``new`` props could not describe
  the whole edit, and props that under-describe the edit are a lie the model would
  learn.  Two-or-more *string* literals now have their own kind,
  ``modify_string_multi``; anything else falls through.
* ``change_condition``, ``modify_call_args`` and ``change_return`` are labelled by
  the **syntactic role of the changed characters**, not merely by the line they
  sit on: the changed character range must lie inside the relevant ``test``
  expression, argument list, or ``return`` statement.
* ``add_guard`` fires when the inserted ``if`` heads a function body (first
  statement, or second after a docstring) or heads a loop body — a ``continue``
  guard is only meaningful inside a loop, which is why loop bodies are included.
* ``insert_statement`` / ``delete_statement`` require the inserted or deleted
  lines to tile complete ``ast`` statement spans, *modulo blank lines* — the
  whitespace around a block (the common ``def`` + trailing blank shape) is not
  content.  A comment-only insertion is therefore still ``freeform_patch`` —
  conservative on purpose.
* ``add_call_arg`` / ``remove_call_arg`` / ``add_dict_entry`` /
  ``remove_dict_entry`` / ``add_list_element`` / ``remove_list_element`` are for
  **whole inserted or deleted lines** that parse as a single argument / entry /
  element inside an enclosing call, dict, or list/set/tuple.  Within-line
  argument edits remain ``modify_call_args`` (which they have always been), so
  these kinds add coverage without relabelling anything.  The enclosing-node
  check is what keeps them apart: a line inside a list is a list element even
  though it would also parse as a call argument.
* ``wrap_block`` is ``wrap_try_except`` without the trailing handler: the region
  is indented under a newly inserted ``with`` / ``if`` / loop header.  The pad is
  the indentation the header *adds* (``indent(first body line) - indent(header)``),
  so a module-level wrap works too.
* ``reindent`` fires when the region's lines are identical after stripping
  leading whitespace.  ``replace_line`` fires on any *single* line replaced by a
  *single* other line whose non-whitespace content differs; it is the last
  structured detector before the statement insert/delete pair, and it is what
  catches the ``+1/-1`` single-line edits that used to be freeform.
* A pure insertion carries ``span == (p, p)`` meaning "insert before original
  line ``p``"; ``p`` may be ``len(old_lines) + 1``.
* ``insert_statement`` / ``delete_statement`` on a member-form file carry
  ``key`` / ``value`` props **only when exactly one member line moved**.  With
  two or more the same rule applies as for literals: one ``key``/``value`` pair
  could not describe the whole edit, so it is omitted and ``text`` alone carries
  the complete description.
* ``rename_symbol`` is emitted only where the toolchain has a real lexer, i.e.
  Python-family paths (:func:`~sylphae.ops.forms.has_python_lexer`).  An unknown or
  extension-less path counts as Python.  On a ``.ts`` file a consistent
  identifier rename therefore decomposes into ``replace_line`` ops instead -- the
  coverage is kept, the claim shrinks to what a token stream can actually prove,
  and the applier stops doing a scope-blind file-wide substitution.
* ``freeform_patch`` carries hashes and a rejection reason, not the new source,
  and is **not** replayable (see :func:`sylphae.ops.apply.apply_op`).
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import io
import tokenize

from .apply import apply_op, apply_sequence
from .forms import deleted_members, has_python_lexer, inserted_members, syntax_for
from .spec import OpApplyError, OpSpec
from .vocab import FREE_FORM_KIND

__all__ = ["detect_op", "detect_sequence"]


# --------------------------------------------------------------------------- #
# token plumbing
# --------------------------------------------------------------------------- #

_NOISE_TYPES: frozenset[int] = frozenset(
    t
    for t in (
        getattr(tokenize, "ENCODING", None),
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
        tokenize.COMMENT,
    )
    if t is not None
)

_STRING_TYPES: frozenset[int] = frozenset(
    t
    for t in (
        tokenize.STRING,
        getattr(tokenize, "FSTRING_START", None),
        getattr(tokenize, "FSTRING_MIDDLE", None),
        getattr(tokenize, "FSTRING_END", None),
    )
    if t is not None
)

_LITERAL_TYPES: frozenset[int] = _STRING_TYPES | {tokenize.NUMBER}

_KEYWORD_LITERALS: frozenset[str] = frozenset({"True", "False", "None"})


def _is_string_tok(type_: int, string: str) -> bool:
    return type_ in _STRING_TYPES


def _is_literal_tok(type_: int, string: str) -> bool:
    if type_ in _LITERAL_TYPES:
        return True
    return type_ == tokenize.NAME and string in _KEYWORD_LITERALS


def _tokenize(src: str) -> list[tokenize.TokenInfo] | None:
    try:
        return list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return None


def _significant(toks: list[tokenize.TokenInfo] | None):
    if toks is None:
        return None
    return [t for t in toks if t.type not in _NOISE_TYPES]


def _first_at_or_after(toks, row: int, col: int) -> int | None:
    for i, t in enumerate(toks):
        if t.start >= (row, col):
            return i
    return None


def _last_before(toks, row: int, col: int) -> int | None:
    idx = None
    for i, t in enumerate(toks):
        if t.start < (row, col):
            idx = i
        else:
            break
    return idx


def _diff_indices(a: list[tuple[int, str]], b: list[tuple[int, str]]) -> list[int] | None:
    """Indices at which two equal-length token sequences differ, or ``None``."""
    if len(a) != len(b):
        return None
    return [i for i in range(len(a)) if a[i] != b[i]]


# --------------------------------------------------------------------------- #
# source plumbing
# --------------------------------------------------------------------------- #


def _split(src: str) -> list[str]:
    return src.splitlines(keepends=True)


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def _dedent_block(text: str) -> str:
    lines = _split(text)
    indents = [len(_indent_of(ln)) for ln in lines if ln.strip() != ""]
    if not indents:
        return text
    k = min(indents)
    if k == 0:
        return text
    return "".join(ln[k:] if ln.strip() != "" else ln for ln in lines)


def _parse(src: str) -> ast.Module | None:
    try:
        return ast.parse(src)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


def _parse_stmts(text: str) -> list[ast.stmt] | None:
    tree = _parse(_dedent_block(text))
    return None if tree is None else list(tree.body)


def _seg(src: str, node: ast.AST | None) -> str:
    if node is None:
        return ""
    return ast.get_source_segment(src, node) or ""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _iter_funcs(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _collect_funcs(tree: ast.Module) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = prefix + child.name
                out.setdefault(qual, child)
                visit(child, qual + ".")
            elif isinstance(child, ast.ClassDef):
                visit(child, prefix + child.name + ".")
            else:
                visit(child, prefix)

    visit(tree, "")
    return out


def _defaults_of(fn: ast.AST) -> list[tuple[str, ast.AST]]:
    args = fn.args
    out: list[tuple[str, ast.AST]] = []
    positional = list(args.posonlyargs) + list(args.args)
    n_def = len(args.defaults)
    if n_def:
        for arg, default in zip(positional[len(positional) - n_def :], args.defaults):
            out.append((arg.arg, default))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if default is not None:
            out.append((arg.arg, default))
    return out


def _sig_key(fn: ast.AST, src: str) -> str:
    """Everything about a signature *except* the default values."""
    args = fn.args
    parts: list[str] = ["async" if isinstance(fn, ast.AsyncFunctionDef) else "def"]
    for arg in args.posonlyargs:
        parts.append("P:" + arg.arg + ":" + _seg(src, arg.annotation))
    parts.append("/")
    for arg in args.args:
        parts.append("A:" + arg.arg + ":" + _seg(src, arg.annotation))
    if args.vararg is not None:
        parts.append("*" + args.vararg.arg + ":" + _seg(src, args.vararg.annotation))
    else:
        parts.append("*")
    for arg in args.kwonlyargs:
        parts.append("K:" + arg.arg + ":" + _seg(src, arg.annotation))
    if args.kwarg is not None:
        parts.append("**" + args.kwarg.arg + ":" + _seg(src, args.kwarg.annotation))
    else:
        parts.append("**")
    parts.append("R:" + _seg(src, fn.returns))
    parts.append("D:" + str(len(fn.decorator_list)))
    return "|".join(parts)


def _defaults_key(fn: ast.AST, src: str) -> tuple[str, ...]:
    return tuple(_seg(src, d) for _, d in _defaults_of(fn))


# --------------------------------------------------------------------------- #
# change geometry
# --------------------------------------------------------------------------- #


def _changed_span(old_line: str, new_line: str) -> tuple[int, int]:
    """Half-open character range of ``old_line`` that the edit touched.

    Common prefix / common suffix are peeled off, so the result is the minimal
    range such that ``old_line[:p] + <anything> + old_line[q:]`` could be the new
    line.  ``p == q`` means the edit is a pure insertion at column ``p``.
    """
    n = min(len(old_line), len(new_line))
    p = 0
    while p < n and old_line[p] == new_line[p]:
        p += 1
    q, r = len(old_line), len(new_line)
    while q > p and r > p and old_line[q - 1] == new_line[r - 1]:
        q -= 1
        r -= 1
    return p, q


def _inside_node(line: int, p: int, q: int, node: ast.AST) -> bool:
    """Is the changed character range on ``line`` inside ``node``'s span?"""
    lo_l, lo_c = node.lineno, node.col_offset or 0
    hi_l = node.end_lineno or node.lineno
    hi_c = node.end_col_offset if node.end_col_offset is not None else lo_c
    if not (lo_l <= line <= hi_l):
        return False
    if line == lo_l == hi_l:
        return lo_c <= p and q <= hi_c
    if line == lo_l:
        return p >= lo_c
    if line == hi_l:
        return q <= hi_c
    return True


def _inside_arg_region(line: int, p: int, q: int, open_tok, close_tok) -> bool:
    """Is the change inside the parentheses of a call?"""
    open_line = open_tok.start[0]
    close_line = close_tok.start[0]
    if line == open_line == close_line:
        return open_tok.end[1] <= p and q <= close_tok.start[1]
    if line == open_line:
        return p >= open_tok.end[1]
    if line == close_line:
        return q <= close_tok.start[1]
    return open_line < line < close_line


def _regions(old_lines: list[str], new_lines: list[str]) -> list[tuple[int, int, int, int]]:
    """Adjacent non-equal opcodes merged into single change regions.

    Each region is ``(i1, i2, j1, j2)``: 0-based half-open line ranges in the old
    and the new source.
    """
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    regions: list[tuple[int, int, int, int]] = []
    pending: list[int] | None = None
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            if pending is not None:
                regions.append(tuple(pending))  # type: ignore[arg-type]
                pending = None
            continue
        if pending is None:
            pending = [i1, i2, j1, j2]
        else:
            pending[1] = i2
            pending[3] = j2
    if pending is not None:
        regions.append(tuple(pending))  # type: ignore[arg-type]
    return regions


class _Ctx:
    """Everything the detectors need, computed at most once per call."""

    __slots__ = (
        "old_src",
        "new_src",
        "path",
        "old_lines",
        "new_lines",
        "regions",
        "_old_tree",
        "_new_tree",
        "_old_toks",
        "_new_toks",
    )

    def __init__(self, old_src: str, new_src: str, path: str) -> None:
        self.old_src = old_src
        self.new_src = new_src
        self.path = path
        self.old_lines = _split(old_src)
        self.new_lines = _split(new_src)
        self.regions = _regions(self.old_lines, self.new_lines)
        self._old_tree: object = _UNSET
        self._new_tree: object = _UNSET
        self._old_toks: object = _UNSET
        self._new_toks: object = _UNSET

    @property
    def old_tree(self) -> ast.Module | None:
        if self._old_tree is _UNSET:
            self._old_tree = _parse(self.old_src)
        return self._old_tree  # type: ignore[return-value]

    @property
    def new_tree(self) -> ast.Module | None:
        if self._new_tree is _UNSET:
            self._new_tree = _parse(self.new_src)
        return self._new_tree  # type: ignore[return-value]

    @property
    def old_toks(self) -> list[tokenize.TokenInfo] | None:
        if self._old_toks is _UNSET:
            self._old_toks = _tokenize(self.old_src)
        return self._old_toks  # type: ignore[return-value]

    @property
    def new_toks(self) -> list[tokenize.TokenInfo] | None:
        if self._new_toks is _UNSET:
            self._new_toks = _tokenize(self.new_src)
        return self._new_toks  # type: ignore[return-value]

    # -- region accessors ---------------------------------------------------

    def single_region(self) -> tuple[int, int, int, int] | None:
        if len(self.regions) != 1:
            return None
        return self.regions[0]

    def pure_insert(self) -> tuple[int, str] | None:
        """``(p, text)`` for a pure line insertion; ``p`` is 1-based."""
        region = self.single_region()
        if region is None:
            return None
        i1, i2, j1, j2 = region
        if i1 != i2 or j1 == j2:
            return None
        return i1 + 1, "".join(self.new_lines[j1:j2])

    def pure_delete(self) -> tuple[int, int, str] | None:
        """``(a, b, text)`` for a pure line deletion; 1-based inclusive."""
        region = self.single_region()
        if region is None:
            return None
        i1, i2, j1, j2 = region
        if j1 != j2 or i1 == i2:
            return None
        return i1 + 1, i2, "".join(self.old_lines[i1:i2])

    def single_replace(self) -> tuple[int, int, str, str] | None:
        """``(a, b, old_text, new_text)`` for one replace region; 1-based inclusive."""
        region = self.single_region()
        if region is None:
            return None
        i1, i2, j1, j2 = region
        if i1 == i2 or j1 == j2:
            return None
        return i1 + 1, i2, "".join(self.old_lines[i1:i2]), "".join(self.new_lines[j1:j2])

    def single_replace_line(self) -> tuple[int, str, str, int, int] | None:
        """``(line, old_text, new_text, p, q)`` when exactly one line was replaced."""
        rep = self.single_replace()
        if rep is None:
            return None
        a, b, old_text, new_text = rep
        if a != b:
            return None
        p, q = _changed_span(old_text, new_text)
        return a, old_text, new_text, p, q

    def region_tokens(self, a: int, b: int, c: int, d: int):
        """Significant tokens on old rows ``a..b`` and new rows ``c..d``."""
        old_t = _significant(self.old_toks)
        new_t = _significant(self.new_toks)
        if old_t is None or new_t is None:
            return None, None
        return (
            [t for t in old_t if a <= t.start[0] <= b],
            [t for t in new_t if c <= t.start[0] <= d],
        )


class _Unset:
    __slots__ = ()


_UNSET = _Unset()


# --------------------------------------------------------------------------- #
# detectors
# --------------------------------------------------------------------------- #


def _import_names(stmts: list[ast.stmt]) -> str:
    names: list[str] = []
    for stmt in stmts:
        if isinstance(stmt, ast.Import):
            names.extend(alias.name for alias in stmt.names)
        elif isinstance(stmt, ast.ImportFrom):
            names.append("." * stmt.level + (stmt.module or ""))
    return ",".join(names)


def _det_add_import(ctx: _Ctx) -> OpSpec | None:
    ins = ctx.pure_insert()
    if ins is None:
        return None
    p, text = ins
    stmts = _parse_stmts(text)
    if not stmts or not all(isinstance(s, (ast.Import, ast.ImportFrom)) for s in stmts):
        return None
    return OpSpec(
        "add_import",
        ctx.path,
        (p, p),
        {"line": str(p), "text": text, "modules": _import_names(stmts)},
    )


def _det_remove_import(ctx: _Ctx) -> OpSpec | None:
    dele = ctx.pure_delete()
    if dele is None:
        return None
    a, b, text = dele
    stmts = _parse_stmts(text)
    if not stmts or not all(isinstance(s, (ast.Import, ast.ImportFrom)) for s in stmts):
        return None
    return OpSpec(
        "remove_import",
        ctx.path,
        (a, b),
        {"text": text, "modules": _import_names(stmts)},
    )


def _det_add_decorator(ctx: _Ctx) -> OpSpec | None:
    ins = ctx.pure_insert()
    if ins is None:
        return None
    tree = ctx.new_tree
    if tree is None:
        return None
    p, text = ins
    j1, j2 = ctx.regions[0][2], ctx.regions[0][3]
    # NOTE: `ast.Decorator` was removed in Python 3.12; a decorator is the bare
    # expression node stored in `FunctionDef.decorator_list`, whose `lineno`
    # points at the expression (i.e. one column past the `@`).
    for fn in _iter_funcs(tree):
        for node in fn.decorator_list:
            if node.end_lineno is None:
                continue
            if node.lineno - 1 == j1 and node.end_lineno == j2:
                decorator = text.strip()
                if decorator.startswith("@"):
                    decorator = decorator[1:]
                return OpSpec(
                    "add_decorator",
                    ctx.path,
                    (p, p),
                    {"line": str(p), "text": text, "decorator": decorator},
                )
    return None


def _det_change_default(ctx: _Ctx) -> OpSpec | None:
    line = ctx.single_replace_line()
    if line is None:
        return None
    a, old_text, new_text, p, q = line
    old_tree, new_tree = ctx.old_tree, ctx.new_tree
    if old_tree is None or new_tree is None:
        return None

    old_funcs = _collect_funcs(old_tree)
    new_funcs = _collect_funcs(new_tree)
    changed: list[tuple[ast.AST, ast.AST]] = []
    for qual, old_fn in old_funcs.items():
        new_fn = new_funcs.get(qual)
        if new_fn is None:
            continue
        if _sig_key(old_fn, ctx.old_src) != _sig_key(new_fn, ctx.new_src):
            continue
        if _defaults_key(old_fn, ctx.old_src) == _defaults_key(new_fn, ctx.new_src):
            continue
        changed.append((old_fn, new_fn))
    if len(changed) != 1:
        return None

    old_fn, new_fn = changed[0]
    if not old_fn.body:
        return None
    sig_hi = max(old_fn.lineno, old_fn.body[0].lineno - 1)
    if not (old_fn.lineno <= a <= sig_hi):
        return None

    param = None
    old_default = None
    for name, node in _defaults_of(old_fn):
        if _inside_node(a, p, q, node):
            param, old_default = name, _seg(ctx.old_src, node)
            break
    if param is None:
        return None
    new_default = None
    for name, node in _defaults_of(new_fn):
        if name == param:
            new_default = _seg(ctx.new_src, node)
            break
    if new_default is None or new_default == old_default:
        return None

    return OpSpec(
        "change_default",
        ctx.path,
        (a, a),
        {
            "line": str(a),
            "old_text": old_text,
            "new_text": new_text,
            "param": param,
            "old_default": old_default,
            "new_default": new_default,
        },
    )


def _guard_bodies(tree: ast.AST):
    """Bodies whose head an early-exit guard can be prepended to.

    Yields ``(body, allow_docstring)``.  Functions allow the guard to sit after a
    docstring; loops do not (a ``continue`` guard is only meaningful in a loop,
    and a function body cannot contain a top-level ``continue``).
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.body, True
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            yield node.body, False


def _det_add_guard(ctx: _Ctx) -> OpSpec | None:
    ins = ctx.pure_insert()
    if ins is None:
        return None
    tree = ctx.new_tree
    if tree is None:
        return None
    p, text = ins
    j1, j2 = ctx.regions[0][2], ctx.regions[0][3]

    for body, allow_docstring in _guard_bodies(tree):
        for idx in (0, 1):
            if idx >= len(body):
                break
            stmt = body[idx]
            if not isinstance(stmt, ast.If):
                continue
            if stmt.lineno - 1 != j1 or stmt.end_lineno != j2:
                continue
            if idx == 1 and not (allow_docstring and _is_docstring(body[0])):
                continue
            if len(stmt.body) != 1:
                continue
            inner = stmt.body[0]
            if isinstance(inner, ast.Return):
                guard = "return"
            elif isinstance(inner, ast.Raise):
                guard = "raise"
            elif isinstance(inner, ast.Continue):
                guard = "continue"
            else:
                continue
            return OpSpec(
                "add_guard",
                ctx.path,
                (p, p),
                {
                    "line": str(p),
                    "text": text,
                    "condition": _seg(ctx.new_src, stmt.test),
                    "guard": guard,
                },
            )
    return None


def _det_wrap_try_except(ctx: _Ctx) -> OpSpec | None:
    tree = ctx.new_tree
    if tree is None:
        return None
    new_lines = ctx.new_lines
    n_new = len(new_lines)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Try) or not node.handlers or not node.body:
            continue
        try_idx = node.lineno - 1
        body_last = node.body[-1].end_lineno
        end = node.end_lineno
        if body_last is None or end is None:
            continue
        if not (0 <= try_idx < n_new) or end > n_new:
            continue
        if node.body[0].lineno - 1 != try_idx + 1:
            continue
        if node.handlers[0].lineno != body_last + 1:
            continue

        try_line = new_lines[try_idx]
        pad = _indent_of(try_line)
        body_lines = new_lines[try_idx + 1 : body_last]
        dedented: list[str] = []
        ok = True
        for ln in body_lines:
            if ln.strip() == "":
                dedented.append(ln)
            elif pad and ln.startswith(pad):
                dedented.append(ln[len(pad) :])
            else:
                ok = False
                break
        if not ok:
            continue

        handler = "".join(new_lines[node.handlers[0].lineno - 1 : end])
        candidate = "".join(new_lines[:try_idx] + dedented + new_lines[end:])
        if candidate != ctx.old_src:
            continue

        a = try_idx + 1
        b = a + len(dedented) - 1
        return OpSpec(
            "wrap_try_except",
            ctx.path,
            (a, b),
            {
                "try_line": try_line,
                "pad": pad,
                "handler": handler,
                "n_body": str(len(dedented)),
            },
        )
    return None


def _det_rename_symbol(ctx: _Ctx) -> OpSpec | None:
    # A rename is a claim about identifiers, and identifiers only exist in a
    # language we can lex.  ``tokenize`` is a Python lexer; on a CSS file it still
    # reports ``red``/``blue`` as NAME tokens, which used to turn a colour change
    # into a symbol rename.  Wherever it is only a stand-in, the honest answer is
    # to decline and let the literal / line detectors pick the change up.
    if not has_python_lexer(ctx.path):
        return None
    old_t = _significant(ctx.old_toks)
    new_t = _significant(ctx.new_toks)
    if old_t is None or new_t is None:
        return None
    a = [(t.type, t.string) for t in old_t]
    b = [(t.type, t.string) for t in new_t]
    diffs = _diff_indices(a, b)
    if not diffs:
        return None
    if not all(a[i][0] == tokenize.NAME and b[i][0] == tokenize.NAME for i in diffs):
        return None
    old_names = {a[i][1] for i in diffs}
    new_names = {b[i][1] for i in diffs}
    if len(old_names) != 1 or len(new_names) != 1:
        return None
    old_name = old_names.pop()
    new_name = new_names.pop()
    if old_name == new_name:
        return None
    if not old_name.isidentifier() or not new_name.isidentifier():
        return None

    old_count = sum(1 for t in old_t if t.type == tokenize.NAME and t.string == old_name)
    new_count = sum(1 for t in new_t if t.type == tokenize.NAME and t.string == old_name)
    # "renamed consistently": every occurrence went, none was left behind.
    if old_count != len(diffs) or new_count != 0:
        return None
    # A name that already exists in the old file makes this a merge, not a rename.
    if any(t.type == tokenize.NAME and t.string == new_name for t in old_t):
        return None

    rows = [old_t[i].start[0] for i in diffs]
    return OpSpec(
        "rename_symbol",
        ctx.path,
        (min(rows), max(rows)),
        {"old": old_name, "new": new_name, "count": str(old_count)},
    )


def _det_change_return(ctx: _Ctx) -> OpSpec | None:
    line = ctx.single_replace_line()
    if line is None:
        return None
    a, old_text, new_text, p, q = line
    old_tree, new_tree = ctx.old_tree, ctx.new_tree
    if old_tree is None or new_tree is None:
        return None

    def find(tree: ast.Module) -> ast.Return | None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Return) and node.lineno == a and node.value is not None:
                return node
        return None

    old_ret = find(old_tree)
    new_ret = find(new_tree)
    if old_ret is None or new_ret is None:
        return None
    if not _inside_node(a, p, q, old_ret):
        return None
    return OpSpec(
        "change_return",
        ctx.path,
        (a, a),
        {
            "line": str(a),
            "old_text": old_text,
            "new_text": new_text,
            "old_value": _seg(ctx.old_src, old_ret.value),
            "new_value": _seg(ctx.new_src, new_ret.value),
        },
    )


def _det_change_condition(ctx: _Ctx) -> OpSpec | None:
    line = ctx.single_replace_line()
    if line is None:
        return None
    a, old_text, new_text, p, q = line
    old_tree = ctx.old_tree
    if old_tree is None:
        return None

    for node in ast.walk(old_tree):
        if not isinstance(node, (ast.If, ast.While, ast.Assert)):
            continue
        test = node.test
        if not _inside_node(a, p, q, test):
            continue
        if isinstance(node, ast.If):
            keyword = "if"
        elif isinstance(node, ast.While):
            keyword = "while"
        else:
            keyword = "assert"
        return OpSpec(
            "change_condition",
            ctx.path,
            (a, a),
            {
                "line": str(a),
                "old_text": old_text,
                "new_text": new_text,
                "keyword": keyword,
            },
        )
    return None


def _det_modify_string(ctx: _Ctx) -> OpSpec | None:
    line = ctx.single_replace_line()
    if line is None:
        return None
    a, old_text, new_text, _p, _q = line
    old_t, new_t = ctx.region_tokens(a, a, a, a)
    if old_t is None or new_t is None:
        return None
    a_toks = [(t.type, t.string) for t in old_t]
    b_toks = [(t.type, t.string) for t in new_t]
    diffs = _diff_indices(a_toks, b_toks)
    # "only a message string changes": exactly one, so that props['old'] and
    # props['new'] describe the whole edit rather than the first of several.
    if not diffs or len(diffs) != 1:
        return None
    if not all(_is_string_tok(*a_toks[i]) and _is_string_tok(*b_toks[i]) for i in diffs):
        return None
    return OpSpec(
        "modify_string",
        ctx.path,
        (a, a),
        {
            "line": str(a),
            "old_text": old_text,
            "new_text": new_text,
            "old": a_toks[diffs[0]][1],
            "new": b_toks[diffs[0]][1],
        },
    )


def _det_modify_call_args(ctx: _Ctx) -> OpSpec | None:
    line = ctx.single_replace_line()
    if line is None:
        return None
    a, old_text, new_text, p, q = line
    old_tree = ctx.old_tree
    toks = _significant(ctx.old_toks)
    if old_tree is None or toks is None:
        return None

    for node in ast.walk(old_tree):
        if not isinstance(node, ast.Call) or not (node.args or node.keywords):
            continue
        func_end = (node.func.end_lineno or node.func.lineno, node.func.end_col_offset or 0)
        call_end = (node.end_lineno or node.lineno, node.end_col_offset or 0)
        if not (node.lineno <= a <= call_end[0]):
            continue
        open_idx = _first_at_or_after(toks, *func_end)
        close_idx = _last_before(toks, *call_end)
        if open_idx is None or close_idx is None or open_idx >= close_idx:
            continue
        if not _inside_arg_region(a, p, q, toks[open_idx], toks[close_idx]):
            continue
        return OpSpec(
            "modify_call_args",
            ctx.path,
            (a, a),
            {
                "line": str(a),
                "old_text": old_text,
                "new_text": new_text,
                "func": _seg(ctx.old_src, node.func),
            },
        )
    return None


def _det_replace_literal(ctx: _Ctx) -> OpSpec | None:
    rep = ctx.single_replace()
    if rep is None:
        return None
    a, b, old_text, new_text = rep
    i1, i2, j1, j2 = ctx.regions[0]
    if (i2 - i1) != (j2 - j1):
        return None  # same line count only
    old_t, new_t = ctx.region_tokens(a, b, j1 + 1, j2)
    if old_t is None or new_t is None:
        return None
    a_toks = [(t.type, t.string) for t in old_t]
    b_toks = [(t.type, t.string) for t in new_t]
    diffs = _diff_indices(a_toks, b_toks)
    # Exactly one literal, so that props['old'] / props['new'] are the whole edit.
    if not diffs or len(diffs) != 1:
        return None
    if not all(_is_literal_tok(*a_toks[i]) and _is_literal_tok(*b_toks[i]) for i in diffs):
        return None
    return OpSpec(
        "replace_literal",
        ctx.path,
        (a, b),
        {
            "line": str(a),
            "old_text": old_text,
            "new_text": new_text,
            "old": a_toks[diffs[0]][1],
            "new": b_toks[diffs[0]][1],
        },
    )


def _complete_statements(
    tree: ast.Module, lo: int, hi: int, lines: list[str]
) -> list[tuple[int, int]] | None:
    """Maximal statement spans covering 1-based lines ``lo..hi``.

    The statements must tile the region *modulo blank lines*: whitespace between
    or around them is not content, so a block inserted with a trailing blank line
    (the common ``def`` + blank shape) is still a complete-statement insertion.
    A line that is neither blank nor the start of a maximal statement span ends
    the walk with ``None`` — a comment-only insertion is therefore still
    ``freeform_patch``, on purpose.
    """
    spans = {
        (node.lineno, node.end_lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.stmt) and node.end_lineno is not None
    }
    inside = sorted(s for s in spans if s[0] >= lo and s[1] <= hi)
    if not inside:
        return None
    maximal = sorted(
        {
            s
            for s in inside
            if not any(o != s and o[0] <= s[0] and s[1] <= o[1] for o in inside)
        }
    )
    cursor = lo
    used: list[tuple[int, int]] = []
    while cursor <= hi:
        if lines[cursor - 1].strip() == "":
            cursor += 1
            continue
        span = next((s for s in maximal if s[0] == cursor), None)
        if span is None:
            return None
        used.append(span)
        cursor = span[1] + 1
    return used or None


def _det_insert_statement(ctx: _Ctx) -> OpSpec | None:
    ins = ctx.pure_insert()
    if ins is None:
        return None
    tree = ctx.new_tree
    if tree is None:
        return None
    p, text = ins
    j1, j2 = ctx.regions[0][2], ctx.regions[0][3]
    stmts = _complete_statements(tree, j1 + 1, j2, ctx.new_lines)
    if stmts is None:
        return None
    return OpSpec(
        "insert_statement",
        ctx.path,
        (p, p),
        {"line": str(p), "text": text, "n_stmts": str(len(stmts))},
    )


def _det_delete_statement(ctx: _Ctx) -> OpSpec | None:
    dele = ctx.pure_delete()
    if dele is None:
        return None
    tree = ctx.old_tree
    if tree is None:
        return None
    a, b, text = dele
    stmts = _complete_statements(tree, a, b, ctx.old_lines)
    if stmts is None:
        return None
    return OpSpec(
        "delete_statement",
        ctx.path,
        (a, b),
        {"text": text, "n_stmts": str(len(stmts))},
    )


# --------------------------------------------------------------------------- #
# detectors added with the sequence decomposition
# --------------------------------------------------------------------------- #


def _single_call_arg(text: str) -> str | None:
    """Parse an inserted/deleted line as exactly one call argument, or ``None``."""
    body = text.strip()
    if body.endswith(","):
        body = body[:-1]
    if not body.strip():
        return None
    src = "f(" + body + ")"
    tree = _parse(src)
    if tree is None or len(tree.body) != 1:
        return None
    stmt = tree.body[0]
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return None
    call = stmt.value
    if len(call.args) + len(call.keywords) != 1:
        return None
    if call.keywords:
        return ast.get_source_segment(src, call.keywords[0])
    return ast.get_source_segment(src, call.args[0])


def _single_dict_entry(text: str) -> tuple[str, str] | None:
    """Parse an inserted/deleted line as exactly one ``key: value`` entry."""
    body = text.strip()
    if body.endswith(","):
        body = body[:-1]
    if not body.strip():
        return None
    src = "{" + body + "}"
    tree = _parse(src)
    if tree is None or len(tree.body) != 1:
        return None
    stmt = tree.body[0]
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Dict):
        return None
    node = stmt.value
    if len(node.keys) != 1 or node.keys[0] is None:
        return None
    key = ast.get_source_segment(src, node.keys[0])
    value = ast.get_source_segment(src, node.values[0])
    if key is None or value is None:
        return None
    return key, value


def _enclosing_call(tree: ast.Module, line: int) -> ast.Call | None:
    """Innermost call whose parentheses strictly enclose ``line``."""
    best: ast.Call | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        hi = node.end_lineno or node.lineno
        if not (node.lineno <= line <= hi):
            continue
        func_end = node.func.end_lineno or node.func.lineno
        if line <= func_end or line >= hi:
            continue
        if best is None or node.lineno >= best.lineno:
            best = node
    return best


def _enclosing_dict(tree: ast.Module, line: int) -> ast.Dict | None:
    best: ast.Dict | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        hi = node.end_lineno or node.lineno
        if node.lineno <= line <= hi and (best is None or node.lineno >= best.lineno):
            best = node
    return best


def _enclosing_list(tree: ast.Module, line: int):
    best = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Set, ast.Tuple)):
            continue
        hi = node.end_lineno or node.lineno
        if node.lineno <= line <= hi and (best is None or node.lineno >= best.lineno):
            best = node
    return best


def _single_list_element(text: str) -> str | None:
    """Parse an inserted/deleted line as exactly one list/set/tuple element."""
    body = text.strip()
    if body.endswith(","):
        body = body[:-1]
    if not body.strip():
        return None
    src = "[" + body + "]"
    tree = _parse(src)
    if tree is None or len(tree.body) != 1:
        return None
    stmt = tree.body[0]
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.List):
        return None
    node = stmt.value
    if len(node.elts) != 1 or isinstance(node.elts[0], ast.Starred):
        return None
    return ast.get_source_segment(src, node.elts[0])


def _det_add_list_element(ctx: _Ctx) -> OpSpec | None:
    ins = ctx.pure_insert()
    tree = ctx.new_tree
    if ins is None or tree is None:
        return None
    p, text = ins
    value = _single_list_element(text)
    if value is None:
        return None
    if _enclosing_list(tree, p) is None:
        return None
    return OpSpec(
        "add_list_element",
        ctx.path,
        (p, p),
        {"line": str(p), "text": text, "value": value},
    )


def _det_remove_list_element(ctx: _Ctx) -> OpSpec | None:
    dele = ctx.pure_delete()
    tree = ctx.old_tree
    if dele is None or tree is None:
        return None
    a, b, text = dele
    value = _single_list_element(text)
    if value is None:
        return None
    if _enclosing_list(tree, a) is None:
        return None
    return OpSpec(
        "remove_list_element",
        ctx.path,
        (a, b),
        {"text": text, "value": value},
    )


def _det_add_call_arg(ctx: _Ctx) -> OpSpec | None:
    ins = ctx.pure_insert()
    tree = ctx.new_tree
    if ins is None or tree is None:
        return None
    p, text = ins
    arg = _single_call_arg(text)
    if arg is None:
        return None
    call = _enclosing_call(tree, p)
    if call is None:
        return None
    return OpSpec(
        "add_call_arg",
        ctx.path,
        (p, p),
        {"line": str(p), "text": text, "func": _seg(ctx.new_src, call.func), "arg": arg},
    )


def _det_remove_call_arg(ctx: _Ctx) -> OpSpec | None:
    dele = ctx.pure_delete()
    tree = ctx.old_tree
    if dele is None or tree is None:
        return None
    a, b, text = dele
    arg = _single_call_arg(text)
    if arg is None:
        return None
    call = _enclosing_call(tree, a)
    if call is None:
        return None
    return OpSpec(
        "remove_call_arg",
        ctx.path,
        (a, b),
        {"text": text, "func": _seg(ctx.old_src, call.func), "arg": arg},
    )


def _det_add_dict_entry(ctx: _Ctx) -> OpSpec | None:
    ins = ctx.pure_insert()
    tree = ctx.new_tree
    if ins is None or tree is None:
        return None
    p, text = ins
    entry = _single_dict_entry(text)
    if entry is None:
        return None
    if _enclosing_dict(tree, p) is None:
        return None
    key, value = entry
    return OpSpec(
        "add_dict_entry",
        ctx.path,
        (p, p),
        {"line": str(p), "text": text, "key": key, "value": value},
    )


def _det_remove_dict_entry(ctx: _Ctx) -> OpSpec | None:
    dele = ctx.pure_delete()
    tree = ctx.old_tree
    if dele is None or tree is None:
        return None
    a, b, text = dele
    entry = _single_dict_entry(text)
    if entry is None:
        return None
    if _enclosing_dict(tree, a) is None:
        return None
    key, value = entry
    return OpSpec(
        "remove_dict_entry",
        ctx.path,
        (a, b),
        {"text": text, "key": key, "value": value},
    )


_WRAPPABLE = (ast.With, ast.AsyncWith, ast.For, ast.AsyncFor, ast.While, ast.If)


def _det_wrap_block(ctx: _Ctx) -> OpSpec | None:
    """A region is indented under a newly inserted ``with`` / ``if`` / loop header.

    The generic sibling of :func:`_det_wrap_try_except`, which is kept separate
    because a ``try`` carries a handler payload and this one does not.  The
    header is identified from the *new* tree and the wrap is proven by removing
    it and dedenting the body back to ``old_src`` exactly.
    """
    tree = ctx.new_tree
    if tree is None:
        return None
    new_lines = ctx.new_lines
    n_new = len(new_lines)
    for node in ast.walk(tree):
        if not isinstance(node, _WRAPPABLE) or getattr(node, "orelse", None):
            continue
        if not node.body:
            continue
        header_idx = node.lineno - 1
        body_last = node.body[-1].end_lineno
        if body_last is None or node.end_lineno != body_last:
            continue
        if node.body[0].lineno - 1 != header_idx + 1:
            continue
        if not (0 <= header_idx < n_new) or body_last > n_new:
            continue
        header = new_lines[header_idx]
        body_lines = new_lines[header_idx + 1 : body_last]
        if not body_lines or not body_lines[0].strip():
            continue
        header_indent = _indent_of(header)
        first_indent = _indent_of(body_lines[0])
        if not first_indent.startswith(header_indent):
            continue
        pad = first_indent[len(header_indent) :]
        if pad == "":
            continue
        dedented: list[str] = []
        ok = True
        for ln in body_lines:
            if ln.strip() == "":
                dedented.append(ln)
            elif ln.startswith(pad):
                dedented.append(ln[len(pad) :])
            else:
                ok = False
                break
        if not ok:
            continue
        candidate = "".join(new_lines[:header_idx] + dedented + new_lines[body_last:])
        if candidate != ctx.old_src:
            continue
        a = header_idx + 1
        b = a + len(dedented) - 1
        return OpSpec(
            "wrap_block",
            ctx.path,
            (a, b),
            {"header": header, "pad": pad, "n_body": str(len(dedented))},
        )
    return None


def _det_modify_string_multi(ctx: _Ctx) -> OpSpec | None:
    """Two or more string literals change inside one hunk.

    ``modify_string`` requires exactly one changed string so that its ``old`` /
    ``new`` props describe the whole edit; this kind exists for the case where
    several do (a path prefix applied to a list of names, a reworded pair of
    messages) and carries the whole region text instead of pretending to name
    a single literal.
    """
    rep = ctx.single_replace()
    if rep is None:
        return None
    a, b, old_text, new_text = rep
    i1, i2, j1, j2 = ctx.regions[0]
    if (i2 - i1) != (j2 - j1):
        return None
    old_t, new_t = ctx.region_tokens(a, b, j1 + 1, j2)
    if old_t is None or new_t is None:
        return None
    a_toks = [(t.type, t.string) for t in old_t]
    b_toks = [(t.type, t.string) for t in new_t]
    diffs = _diff_indices(a_toks, b_toks)
    if not diffs or len(diffs) < 2:
        return None
    if not all(_is_string_tok(*a_toks[i]) and _is_string_tok(*b_toks[i]) for i in diffs):
        return None
    return OpSpec(
        "modify_string_multi",
        ctx.path,
        (a, b),
        {
            "line": str(a),
            "old_text": old_text,
            "new_text": new_text,
            "n_strings": str(len(diffs)),
        },
    )


def _det_reindent(ctx: _Ctx) -> OpSpec | None:
    """Only leading whitespace changes; every line's content is identical."""
    rep = ctx.single_replace()
    if rep is None:
        return None
    a, b, old_text, new_text = rep
    old_lines = _split(old_text)
    new_lines = _split(new_text)
    if not old_lines or len(old_lines) != len(new_lines):
        return None
    changed = False
    for old_ln, new_ln in zip(old_lines, new_lines):
        if old_ln == new_ln:
            continue
        if old_ln.lstrip(" \t") != new_ln.lstrip(" \t"):
            return None
        changed = True
    if not changed:
        return None
    return OpSpec(
        "reindent",
        ctx.path,
        (a, b),
        {"line": str(a), "old_text": old_text, "new_text": new_text},
    )


def _det_replace_line(ctx: _Ctx) -> OpSpec | None:
    """One whole line replaced by exactly one other whole line.

    The last-resort structured detector: it fires on any single-line replacement
    the more specific detectors declined, which is exactly the ``+1/-1`` diff
    opcode that made up a large share of the old freeform failures.  A change
    that only moves whitespace is *not* a line replacement — that is
    ``reindent`` (indentation) or ``freeform_patch`` (anything else) — so this
    stays conservative about what counts as a real edit.
    """
    line = ctx.single_replace_line()
    if line is None:
        return None
    a, old_text, new_text, _p, _q = line
    i1, i2, j1, j2 = ctx.regions[0]
    if (i2 - i1) != 1 or (j2 - j1) != 1:
        return None
    if old_text.split() == new_text.split():
        return None
    return OpSpec(
        "replace_line",
        ctx.path,
        (a, a),
        {"line": str(a), "old_text": old_text, "new_text": new_text},
    )


_DETECTORS = (
    _det_add_import,
    _det_remove_import,
    _det_add_decorator,
    _det_add_dict_entry,
    _det_remove_dict_entry,
    _det_add_list_element,
    _det_remove_list_element,
    _det_add_call_arg,
    _det_remove_call_arg,
    _det_change_default,
    _det_add_guard,
    _det_wrap_block,
    _det_wrap_try_except,
    _det_rename_symbol,
    _det_change_return,
    _det_change_condition,
    _det_modify_string_multi,
    _det_modify_string,
    _det_modify_call_args,
    _det_replace_literal,
    _det_reindent,
    _det_replace_line,
    _det_insert_statement,
    _det_delete_statement,
)

# --------------------------------------------------------------------------- #
# detectors for the member-form languages (see :mod:`sylphae.ops.forms`)
# --------------------------------------------------------------------------- #


def _member_props(members, base: dict[str, str]) -> dict[str, str]:
    """``base`` plus per-member props, but only when the props describe it all.

    One ``key``/``value`` pair is emitted for a single member line.  For two or
    more it would under-describe the edit -- the same standard the literal
    detectors are held to -- so it is dropped and ``text`` speaks for itself.
    """
    props = dict(base)
    props["n_stmts"] = str(len(members))
    if len(members) == 1:
        props["key"] = members[0].key
        props["value"] = members[0].value
    return props


def _det_insert_member(ctx: _Ctx) -> OpSpec | None:
    """A member line inserted into a block of a non-Python, member-form file.

    The language-general counterpart of :func:`_det_insert_statement`, which needs
    Python ``ast`` to know the inserted lines tile complete statements.  The
    evidence here is structural instead: the line reads as ``key <sep> value`` for
    its extension, and its indentation is proven to belong to the block it lands
    in.  The key and the value are read off the source, not assumed.
    """
    ins = ctx.pure_insert()
    if ins is None:
        return None
    p, text = ins
    members = inserted_members(ctx.path, ctx.old_lines, p, text)
    if members is None:
        return None
    return OpSpec(
        "insert_statement",
        ctx.path,
        (p, p),
        _member_props(members, {"line": str(p), "text": text}),
    )


def _det_delete_member(ctx: _Ctx) -> OpSpec | None:
    """A member line removed from a block of a non-Python, member-form file."""
    dele = ctx.pure_delete()
    if dele is None:
        return None
    a, b, text = dele
    members = deleted_members(ctx.path, ctx.old_lines, a, b)
    if members is None:
        return None
    return OpSpec(
        "delete_statement",
        ctx.path,
        (a, b),
        _member_props(members, {"text": text}),
    )


#: Run ahead of the Python suite, and only for files whose lines are members.
_MEMBER_DETECTORS = (
    _det_insert_member,
    _det_delete_member,
)


def _detectors_for(path: str) -> tuple:
    """The detector order for ``path``.

    Python files get the frozen order.  A member-form file gets the two member
    detectors in front, because on those files the Python suite is an
    approximation and the member test is the real one; the Python suite still
    follows, since its literal-level evidence remains sound there.
    """
    if syntax_for(path) is not None:
        return _MEMBER_DETECTORS + _DETECTORS
    return _DETECTORS


_APPLY_FALLBACK_ERRORS = (
    OpApplyError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    AttributeError,
    SyntaxError,
    RecursionError,
    tokenize.TokenError,
)


def _freeform_span(ctx: _Ctx) -> tuple[int, int]:
    n_old = len(ctx.old_lines)
    if not ctx.regions:
        return (1, max(1, n_old))
    i1 = min(r[0] for r in ctx.regions)
    i2 = max(r[1] for r in ctx.regions)
    if i1 == i2:  # pure insertion: point before old line i1 + 1
        return (min(i1 + 1, n_old + 1), min(i1 + 1, n_old + 1))
    return (i1 + 1, max(i2, i1 + 1))


def _reject_reason(ctx: _Ctx) -> str:
    region = ctx.single_region()
    if region is None:
        return f"multiple_change_regions:{len(ctx.regions)}"
    i1, i2, j1, j2 = region
    if i1 == i2:
        return "insert_not_a_known_statement"
    if j1 == j2:
        return "delete_not_a_known_statement"
    return "replace_not_a_known_edit"


def detect_op(old_src: str, new_src: str, path: str) -> OpSpec | None:
    """Classify one file's change.

    Returns ``None`` when ``old_src == new_src``.
    Returns an :class:`OpSpec` of kind ``freeform_patch`` when no structured op
    reproduces ``new_src``.

    Guarantees, for every non-``freeform_patch`` result, that
    ``apply_op(old_src, result) == new_src`` exactly — the invariant is enforced
    here, by replay, not by convention.
    """
    if not isinstance(old_src, str) or not isinstance(new_src, str):
        raise TypeError("detect_op requires two str sources")
    if not isinstance(path, str) or not path:
        raise ValueError("detect_op requires a non-empty str path")

    if old_src == new_src:
        return None

    ctx = _Ctx(old_src, new_src, path)
    for detector in _detectors_for(path):
        try:
            spec = detector(ctx)
        except _APPLY_FALLBACK_ERRORS:
            spec = None
        if spec is None:
            continue
        try:
            if apply_op(old_src, spec) == new_src:
                return spec
        except _APPLY_FALLBACK_ERRORS:
            continue

    return OpSpec(
        FREE_FORM_KIND,
        path,
        _freeform_span(ctx),
        {
            "reason": _reject_reason(ctx),
            "n_regions": str(len(ctx.regions)),
            "old_sha256": _sha256(old_src),
            "new_sha256": _sha256(new_src),
        },
    )


# --------------------------------------------------------------------------- #
# sequence decomposition
# --------------------------------------------------------------------------- #


def _apply_opcodes(
    old_lines: list[str], new_lines: list[str], opcodes
) -> str:
    """``old_lines`` with exactly ``opcodes``' changes applied (others untouched)."""
    out: list[str] = []
    cursor = 0
    for _tag, i1, i2, j1, j2 in opcodes:
        out.extend(old_lines[cursor:i1])
        out.extend(new_lines[j1:j2])
        cursor = i2
    out.extend(old_lines[cursor:])
    return "".join(out)


def _hunks(opcodes, gap: int) -> list[list]:
    """Group non-equal opcodes, merging across equal runs shorter than ``gap``.

    Each returned hunk is the list of *all* opcodes in its range (equal ones
    included) so that a hunk can later be split at those equal runs.
    """
    hunks: list[list] = []
    cur: list = []
    pending: list = []
    for op in opcodes:
        if op[0] == "equal":
            if cur:
                pending.append(op)
            continue
        if cur:
            if sum(o[2] - o[1] for o in pending) >= gap:
                hunks.append(cur)
                cur = []
            else:
                cur.extend(pending)
        pending = []
        cur.append(op)
    if cur:
        hunks.append(cur)
    return hunks


def _split_equal_replace(hunk: list) -> list[list]:
    """Split same-length ``replace`` opcodes into per-line replace opcodes.

    Two independently edited adjacent lines arrive from ``difflib`` as a single
    ``replace`` opcode, so there is no equal run to split on.  When the opcode
    replaces ``L`` old lines with ``L`` new ones, each line pair is a candidate
    edit in its own right — this is how "two ops touching adjacent lines" becomes
    expressible.  Anything that is not a same-length multi-line replace is passed
    through untouched.
    """
    out: list[list] = []
    for tag, i1, i2, j1, j2 in hunk:
        if tag == "replace" and (i2 - i1) == (j2 - j1) and (i2 - i1) >= 2:
            for k in range(i2 - i1):
                out.append([("replace", i1 + k, i1 + k + 1, j1 + k, j1 + k + 1)])
        else:
            out.append([(tag, i1, i2, j1, j2)])
    return out


def _split_replace_head(hunk: list) -> list[list]:
    """Peel the first line pair of a replace so its tail can be an insertion.

    Some edits are genuinely two ops but arrive as one ``replace`` opcode: adding
    a JSON object member edits the previous line (it must gain its comma) *and*
    inserts a line, and ``difflib`` sees a single ``replace`` of one old line by
    two new ones.  No single op can express that, so the sequence decomposer has
    to -- this splitter says "the first new line belongs with the first old line;
    the remaining new lines were inserted".

    Only applies when the replace grows (``j2 - j1 > i2 - i1``), and only for the
    first pair, so a same-length replace and a shrinking one are untouched.  Every
    part is still offered to the full detector suite against the *original*
    source and the whole sequence is still replay-verified, so a bad split costs
    coverage rather than correctness.
    """
    out: list[list] = []
    for tag, i1, i2, j1, j2 in hunk:
        if tag == "replace" and i2 - i1 >= 1 and j2 - j1 > i2 - i1:
            out.append([("replace", i1, i1 + 1, j1, j1 + 1)])
            out.append([("insert", i1 + 1, i1 + 1, j1 + 1, j2)])
        else:
            out.append([(tag, i1, i2, j1, j2)])
    return out


def _explain_hunk(
    old_src: str, old_lines: list[str], new_lines: list[str], hunk: list, path: str
) -> list[OpSpec] | None:
    """Explain one hunk as a sequence of structured ops, or ``None``.

    First the whole hunk is offered to the single-op detector; if that fails the
    hunk is split at its internal equal runs, then at its same-length replace
    opcodes, then by peeling a growing replace's head -- and each part is
    explained recursively.  Merging (in :func:`_hunks`) is what lets a wrap --
    which shows up as a header insert *plus* a body reindent -- be recognised as
    one op.
    """
    mid = _apply_opcodes(old_lines, new_lines, hunk)
    if mid != old_src:
        spec = detect_op(old_src, mid, path)
        if spec.kind != FREE_FORM_KIND:
            return [spec]
    for parts in (_hunks(hunk, 1), _split_equal_replace(hunk), _split_replace_head(hunk)):
        if len(parts) < 2:
            continue
        out: list[OpSpec] = []
        for part in parts:
            sub = _explain_hunk(old_src, old_lines, new_lines, part, path)
            if sub is None:
                out = []
                break
            out.extend(sub)
        if out:
            return out
    return None


def detect_sequence(old_src: str, new_src: str, path: str) -> list[OpSpec]:
    """Decompose one file's change into an ordered sequence of ops.

    This is the general form of :func:`detect_op`: a change is not forced to be
    *one* op.  The algorithm is per-hunk classification:

    1. If a single op already reproduces ``new_src``, return ``[that op]``.
    2. Otherwise compute the line-level diff opcodes, merge non-equal opcodes
       separated by fewer than three equal lines into hunks, and try the whole
       detector suite on each hunk in isolation (each hunk's edit alone, against
       the *original* source).
    3. If every hunk is explained, return the concatenation; if any hunk is not,
       return a single ``freeform_patch`` for the file.

    Ordering / application contract
    -------------------------------

    Ops are emitted **top-to-bottom** (ascending original line number), and each
    op's ``span`` is in the **original** source, exactly like :func:`detect_op`
    — the sequence is *not* expressed in intermediate coordinates.  Applying it
    therefore means applying **back-to-front** (``apply_sequence``, or
    ``functools.reduce(apply_op, reversed(seq), old_src)``): the lowest hunk goes
    first, so it cannot shift the line numbers of the hunks above it.  Applying
    the sequence forward would be wrong for any change that alters the line
    count, which is why the direction is part of the contract and not an
    implementation detail.

    Returns ``[]`` when ``old_src == new_src``.  Guarantees, for any returned
    sequence whose ops are not all ``freeform_patch``, that
    ``apply_sequence(old_src, seq) == new_src`` exactly — enforced by replay, not
    by convention.  A failed decomposition returns ``[freeform_patch]``.
    """
    if not isinstance(old_src, str) or not isinstance(new_src, str):
        raise TypeError("detect_sequence requires two str sources")
    if not isinstance(path, str) or not path:
        raise ValueError("detect_sequence requires a non-empty str path")

    if old_src == new_src:
        return []

    spec = detect_op(old_src, new_src, path)
    if spec is not None and spec.kind != FREE_FORM_KIND:
        return [spec]

    old_lines = _split(old_src)
    new_lines = _split(new_src)
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    hunks = _hunks(matcher.get_opcodes(), 3)
    seq: list[OpSpec] = []
    explained = bool(hunks)
    for hunk in hunks:
        part = _explain_hunk(old_src, old_lines, new_lines, hunk, path)
        if part is None:
            explained = False
            break
        seq.extend(part)
    if explained and seq:
        try:
            if apply_sequence(old_src, seq) == new_src:
                return seq
        except _APPLY_FALLBACK_ERRORS:
            pass

    if spec is None:  # pragma: no cover - detect_op returns None only if equal
        ctx = _Ctx(old_src, new_src, path)
        spec = OpSpec(
            FREE_FORM_KIND,
            path,
            _freeform_span(ctx),
            {
                "reason": _reject_reason(ctx),
                "n_regions": str(len(ctx.regions)),
                "old_sha256": _sha256(old_src),
                "new_sha256": _sha256(new_src),
            },
        )
    return [spec]
