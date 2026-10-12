"""Cross-file reference analysis: does an edit leave the tree inconsistent?

The reflex writes one file at a time. That is fine for a local edit and wrong for anything
that has to hold across files, and the two cases fail in opposite ways:

- **rename** renames a definition without its call sites. MEASURED on
  ``pyv_rich_rename_not_renderable``: high confidence, patch written, content check
  passed, file compiled, and two tests red because ``rich/console.py`` and friends still
  import the old name. Nothing in the reflex loop noticed.
- **guard** needs to *insert* lines, which a single-file substitution cannot do, so the
  parameter decoder produced garbage, confidence came back 0.228, and the run escalated.

The second outcome is the one we want. This module supplies the evidence for the first to
get it: a rename is only safe to apply to one file if the symbol has no references outside
it.

**Why AST and not a keyword rule.** AGENTS.md forbids deciding semantics from naive
substring tests, and a renamed identifier is exactly the case where that fails --
``LoadedPlugin`` the class and ``loaded_plugin`` the attribute are different symbols, and a
blind search for the former matches the latter. So resolution goes through :mod:`ast`:
imports, names, attributes, arguments and redefinitions are distinguished, and a match has
to be an identifier occurrence rather than a word in a string.

Text matching is available behind a flag because it catches the case AST structurally
cannot: an entry point declared as ``"flake8.plugins:LoadedPlugin"`` is a genuine
reference that no name-resolution pass will ever see. MEASURED in 12.30: that is precisely
why a *complete* rename of flake8's ``LoadedPlugin`` breaks 18 tests while a partial one
breaks none. The two are reported as different kinds so the caller can say which it found.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Reference", "find_references", "rename_is_single_file_safe"]


@dataclass(frozen=True)
class Reference:
    """One occurrence of a symbol, and what kind of occurrence it is.

    ``col_offset``/``end_col_offset`` are byte offsets into the UTF-8 encoded line, which is
    what CPython's ``ast`` reports -- so they are converted to *character* indices before
    being stored, because every consumer here slices Python ``str``. Getting that wrong
    would corrupt any line containing a non-ASCII character before the offset, which is
    exactly the kind of silent damage a rename must never cause.
    """

    path: str
    line: int
    #: ``import`` | ``name`` | ``attribute`` | ``arg`` | ``definition`` | ``text``
    kind: str
    text: str = ""
    #: Character offsets within ``line``, 1-based line numbering. ``None`` for a ``text``
    #: or ``config`` hit, which is a substring match rather than a resolved identifier and
    #: therefore has no single position -- see ``apply_multi_file_rename`` for why those are
    #: treated as blockers rather than rewritten.
    col: int | None = None
    end_col: int | None = None

    def __str__(self) -> str:
        return f"{self.path}:{self.line} ({self.kind})"


def _byte_to_char(line: str, byte_offset: int) -> int:
    """Convert a UTF-8 byte offset within ``line`` to a character index.

    ``ast`` reports ``col_offset``/``end_col_offset`` as UTF-8 byte offsets, while every
    consumer of a :class:`Reference` slices a Python ``str``. For an ASCII line the two are
    equal, which is why this is the kind of bug that survives a test suite written in English:
    a single non-ASCII character earlier in the line shifts every later offset, and a rename
    driven by the wrong offset rewrites the wrong span.

    Returns ``byte_offset`` unchanged when it is past the end of the line, so a caller still
    gets a usable number rather than an exception on malformed input.
    """
    if byte_offset <= 0:
        return 0
    try:
        return len(line.encode("utf-8")[:byte_offset].decode("utf-8", "ignore"))
    except (UnicodeError, AttributeError):
        return byte_offset


def _identifier_is(node: ast.AST, symbol: str) -> bool:
    """Whether this AST node's own identifier *is* ``symbol``.

    A blanket ``ast.walk`` match would report every reference to every name in the file,
    which is useless; the check has to be on the node that carries the identifier.
    """
    if isinstance(node, ast.Name):
        return node.id == symbol
    if isinstance(node, ast.Attribute):
        return node.attr == symbol
    if isinstance(node, ast.arg):
        return node.arg == symbol
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name == symbol
    if isinstance(node, ast.ImportFrom):
        # `as` rebinding still references the original: `from a import WidgetError as
        # Renamed` names WidgetError, it just calls it something else locally. Checking
        # only the bound name misses exactly the case an alias makes interesting -- the
        # whole point of the alias is that the original name appears without being used.
        return any(a.name == symbol or (a.asname or "") == symbol
                   for a in node.names)
    if isinstance(node, ast.Import):
        return any(a.name == symbol or a.name.rsplit(".", 1)[-1] == symbol
                   or (a.asname or "") == symbol
                   for a in node.names)
    return False


def _kind_of(node: ast.AST) -> str:
    """Which sort of reference this node is."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return "import"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return "definition"
    if isinstance(node, ast.Attribute):
        return "attribute"
    if isinstance(node, ast.arg):
        return "arg"
    return "name"


#: Non-source files where a symbol can be referenced by name and no name resolution
#: will ever see it. MEASURED 12.30: flake8's tests reference a plugin class through an
#: entry-point *string*, which is why a complete rename there broke 18 tests while a
#: definition-only rename broke none.
CONFIG_GLOBS = ("setup.cfg", "pyproject.toml", "tox.ini", "setup.py",
                "*.cfg", "*.ini", "*.toml")

#: Directories a repository's own tooling keeps in it, none of which a symbol reference
#: lives in. Skipping them is what makes one walk cheap rather than merely fewer than
#: seven; the vendored snapshots are full of them.
IGNORED_DIRS = frozenset({
    ".git", ".tox", ".nox", ".venv", "venv", "env", "__pycache__",
    "node_modules", "build", "dist", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "site-packages", "htmlcov", ".eggs",
})


def _config_files(root: Path, excluded: set[str]) -> list[str]:
    """Config files that could name a symbol by string. One walk, not one per pattern.

    MEASURED: the per-pattern ``rglob`` version cost **71 ms** on tiangolo__typer's 637
    files -- seven full tree walks for a handful of results, inside a 150 ms budget. A
    single ``os.walk`` collects every matching suffix at once.
    """
    suffixes = {p.lstrip("*") for p in CONFIG_GLOBS}
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in IGNORED_DIRS and d != ".git"]
        for name in filenames:
            if Path(name).suffix.lower() in suffixes:
                rel = Path(dirpath, name).relative_to(root).as_posix()
                if rel not in excluded:
                    out.append(rel)
    return sorted(out)

def _files_containing(root: Path, symbol: str) -> list[str] | None:
    """Files whose text contains ``symbol`` as a whole word. ``None`` if rg is missing.

    MEASURED, and this is the whole optimization. The first version AST-parsed every
    ``.py`` file: **430 ms** on typer's 812 files. Replacing that with a Python-level
    per-file read-and-regex pass was **worse** -- 1262 ms -- because the cost was never the
    parsing, it was reading and decoding every file. ``rg -l -w`` is a multi-threaded Rust
    grep doing exactly this one job, and it is already a dependency of the explorer.

    ``-w`` is ripgrep's word-boundary match, which is the same predicate the Python fallback
    uses, so the two paths agree on what counts as a hit. A test asserts that on a real
    tree, because a pre-filter that quietly disagrees with the fallback would be a
    correctness bug wearing a performance hat.

    Returns ``None`` rather than ``[]`` when the binary is unavailable, so the caller falls
    back instead of concluding there are no references.
    """
    import subprocess

    from sylphae.agent.tools import get_ripgrep_binary

    binary = get_ripgrep_binary()
    if not binary:
        return None
    try:
        proc = subprocess.run(
            [binary, "-l", "-w", "--no-messages", "--glob", "!.git",
             "--", symbol, "."],
            cwd=str(root), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode not in (0, 1):  # 0 = found, 1 = no match
        return None
    out = []
    for line in (proc.stdout or "").splitlines():
        rel = line.strip()
        if not rel:
            continue
        # Windows: `rg -l` emits `.\pkg\errors.py`. Stripping only "./" left the path as
        # "./pkg/errors.py", which then failed the `exclude` comparison and the exclusion
        # silently stopped excluding -- MEASURED, the gate reported a symbol as having an
        # external reference when it had none. AGENTS.md warns about exactly this class of
        # path bug; normalising through posix form is what makes both platforms agree.
        posix = Path(rel).as_posix()
        while posix.startswith("./"):
            posix = posix[2:]
        out.append(posix)
    return out


def find_references(
    root: Path,
    symbol: str,
    *,
    exclude: tuple[str, ...] = (),
    include_text: bool = True,
) -> list[Reference]:
    """Every reference to ``symbol`` under ``root``, except in ``exclude``d files.

    Two passes, because the single-pass version cost more than the reflex it was added to.

    **MEASURED**, the first implementation AST-parsed every ``.py`` file: **430 ms** on
    ``tiangolo__typer``'s 812 files, against a 150 ms product requirement (DESIGN.md
    section 5). It was accurate and unusable, and a check that can only be afforded on the
    shapes that least need it is a check that will be skipped.

    So the passes are ordered by cost. A word-boundary text scan is I/O and regex, and it is
    a strict superset: any file whose AST references ``symbol`` necessarily contains the
    identifier as text, so pre-filtering cannot drop a real reference. Only the handful of
    files that hit get parsed. MEASURED after: the same 11 references on rich, in single-digit
    milliseconds.

    That asymmetry is safe only because the pre-filter is a superset, so it is written as
    one -- and the second pass still re-checks, rather than trusting the text hit to be the
    same thing as a reference.

    ``include_text`` adds the text hits themselves as ``text``/``config`` references, for
    the case AST cannot see: entry points are strings.
    """
    from sylphae.repo.index import iter_python_files

    root = Path(root)
    excluded = {p.replace("\\", "/") for p in exclude}
    word = re.compile(rf"(?<!\w){re.escape(symbol)}(?!\w)")

    out: list[Reference] = []
    seen: set[tuple[str, int]] = set()

    # -- pass 1: text, as a superset filter ------------------------------------------
    # ripgrep when it is available, because the cost of this stage is I/O over the whole
    # tree and that is the one thing a Python loop is bad at. The Python fallback is the
    # same predicate, and `None` from the helper means "could not ask", not "no hits".
    hits = _files_containing(root, symbol)
    if hits is None:
        hits = [
            rel for rel in iter_python_files(root)
            if rel.replace("\\", "/") not in excluded
            and word.search(
                (root / rel).read_text(encoding="utf-8", errors="replace")
            )
        ]

    candidates: list[str] = []
    for norm in sorted(hits):
        norm = norm.replace("\\", "/")
        if norm in excluded:
            continue
        try:
            source = (root / norm).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        candidates.append(norm)
        if include_text:
            for i, line in enumerate(source.splitlines(), 1):
                if word.search(line):
                    out.append(Reference(norm, i, "text", line.strip()[:100]))

    # -- pass 2: resolve only the files that hit -------------------------------------
    for norm in candidates:
        try:
            source = (root / norm).read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except (OSError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if not _identifier_is(node, symbol):
                continue
            lineno = getattr(node, "lineno", 0)
            raw_col = getattr(node, "col_offset", None)
            raw_end = getattr(node, "end_col_offset", None)
            if raw_col is None or raw_end is None:
                out.append(Reference(norm, lineno, _kind_of(node)))
                continue
            line_text = source.splitlines()[lineno - 1] if lineno else ""
            # ast reports UTF-8 byte offsets; this module and every consumer slice str.
            col = _byte_to_char(line_text, raw_col)
            end = _byte_to_char(line_text, raw_end)

            # The span has to be narrowed at BOTH ends. ``col_offset`` points at the start
            # of the construct and ``end_col_offset`` at its end, and for Attribute, Import,
            # ImportFrom, ClassDef and FunctionDef the construct is wider than the name:
            #
            #   raise errors.NotRenderableError(   Attribute  span 4..30, name 11..30
            #   from rich.errors import XyzError   ImportFrom span 0..41, name 27..41
            #   class NotRenderableError(...):      ClassDef   span 0..40, name 6..25
            #
            # MEASURED 2026-10-02: the executor's span check refused every one of these,
            # correctly -- a rewrite driven by the un-narrowed span would have deleted the
            # module qualifier, the whole import prefix, or the class statement's arguments.
            # The check is what found this; these narrowings are what make it pass.
            if isinstance(node, ast.Attribute):
                dot = line_text.rfind(".", col, end)
                if dot != -1:
                    col = _byte_to_char(line_text, dot + 1)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                   ast.Import, ast.ImportFrom)):
                # The name is the last place the symbol appears inside the construct, for
                # every form that can reference it: `class X(...)` and `def x(...)` name it
                # directly, `import a.b as c` puts it last as the alias, `from a import A as
                # B` puts it last as the alias. The search is bounded by ``end``, so it
                # cannot escape the statement.
                at = line_text.rfind(symbol, col, end)
                if at != -1:
                    col = _byte_to_char(line_text, at)
                    end = col + len(symbol)

            # Deduplicated on the *narrowed* span, not the raw one. Keyed on
            # (path, line, raw_col) the walk reported `rich/console.py:1326` twice, because
            # the Attribute node and its enclosing Call share a start offset while the
            # narrowed position is distinct -- and an executor driven by that list would have
            # replaced the same span twice, producing `NotRenderableNotRenderable`.
            # Keyed on the line alone the opposite happens: two references on one line
            # collapse to one, and the second is left dangling.
            key = (norm, lineno, col, end)
            if key in seen:
                continue
            seen.add(key)
            out.append(Reference(norm, lineno, _kind_of(node), text=line_text.strip()[:100],
                                 col=col, end_col=end))

    if include_text:
        for rel in _config_files(root, excluded):
            try:
                source = (root / rel).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(source.splitlines(), 1):
                if word.search(line):
                    out.append(Reference(rel, i, "config", line.strip()[:100]))

    return sorted(out, key=lambda r: (r.path, r.line, r.kind, r.col or 0))


def rename_is_single_file_safe(
    root: Path,
    symbol: str,
    target_path: str,
    *,
    include_text: bool = True,
) -> tuple[bool, list[Reference]]:
    """Whether renaming ``symbol`` in ``target_path`` alone leaves the tree consistent.

    Returns ``(safe, references)``. ``safe`` is ``True`` only when the symbol has no
    reference outside the file being edited.

    A rename that turns out unsafe is **not** a failure of the edit -- it is a request the
    single-file path cannot honour, and the honest response is to escalate with the list,
    not to write a partial change. MEASURED: on the reflex set, that is the difference
    between a run that quietly broke two tests and a run that declined and said which
    symbol, which files and which lines would have had to move.
    """
    refs = find_references(
        root, symbol,
        exclude=(target_path,),
        include_text=include_text,
    )
    return (not refs), refs