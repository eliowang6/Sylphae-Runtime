"""Repository world state: file inventory, AST symbol index, import graph, retrieval.

Implements INTERFACES.md section 3.

Design constraints that shaped this module:

* **Standard library only.** ``ast`` is the parser. ``libcst``/``tree-sitter`` are not
  installed and are not going to be.
* **Parse failures are data, not exceptions.** A file that does not parse is recorded with
  ``symbols=[]`` and kept in the inventory. A repository with one broken file must still be
  indexable, because a broken file is exactly the kind of thing an editing agent is asked
  to fix.
* **Retrieval is lexical + structural, never learned.** No embeddings, no GPU, no
  nondeterminism. For a given ``(index, request, k)`` the output is byte-identical across
  processes. This is what makes the recall benchmark in ``bench_recall.py`` a ceiling
  rather than a sample.

The ranker is a weighted sum of five signals plus a mild prior. Every weight is a module
constant, so the ablation in the module docstring can be reproduced:

===================  =====================================================
``W_SYM``            request token appears in one of the file's symbol tokens
``W_SYM_EXACT``      request token equals a whole symbol name (lowercased)
``W_PATH``           request token appears in a path token
``W_STEM``           request token equals the file stem
``W_IMPORT``         import-graph proximity to files that already matched
``W_DEPTH``          per-directory penalty (mild prior favouring shallow paths)
``W_TEST_PENALTY``   flat penalty for test files (mild prior)
``W_CENTRALITY``     import-graph in-degree, normalised (mild prior favouring core files)
===================  =====================================================

Token weights are IDF over the index itself, so a token that appears in every file
(``self``, ``init``) contributes almost nothing and a token that appears in one file
contributes a lot. The IDF map is a free by-product of the index and costs no downloads.

Measured ablation (500 real commits, 5 repositories, ``bench_recall.py``)
------------------------------------------------------------------------

================================================================  ======  ======
configuration                                                     r@3     r@8
================================================================  ======  ======
shipped (all signals)                                             0.670   0.828
``W_CENTRALITY = 0``                                              0.680   0.802
``W_IMPORT = 0``                                                  0.658   0.820
priors gated on lexical signal (rejected)                         0.590   0.762
================================================================  ======  ======

Two findings worth keeping. The priors must stay **ungated**: gating them on lexical
signal was measurably worse, because on a request that matches nothing the depth prior is
the only information available and it is right more often than chance. And import-graph
**centrality** buys recall@8 (+2.6pp) while costing recall@3 (-1.0pp); it is what lets a
``pkg/__init__.py`` target be reached at all when the subject is a bare version string.
Per-repository effects are larger than the aggregate: centrality takes markupsafe from
0.71 to 0.90 and itsdangerous from 0.95 down to 0.82. The aggregate difference is about
1.4 standard errors, so treat it as suggestive rather than settled.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from sylphae.repo.polyglot import (
    SYLPHAE_ARTIFACT_PREFIX,
    POLYGLOT_EXTENSIONS,
    extract_polyglot_symbols,
    is_ignored_file,
    iter_project_files,
)

__all__ = [
    "Candidate",
    "FileRecord",
    "ImportRef",
    "REPO_INDEX_FILE_NAME",
    "RepoIndex",
    "SNIPPET_MAX_CHARS",
    "collect_imports",
    "collect_symbols",
    "collect_polyglot_symbols",
    "is_test_path",
    "iter_python_files",
    "iter_indexable_files",
    "module_name_of",
    "package_name_of",
    "parse_source",
    "resolve_import_candidates",
    "tokenize_text",
]

#: Bumped whenever ``collect_symbols`` changes *what* it collects, not merely how.
#:
#: A cache written by an older build is a valid file with a valid ``root`` and a
#: plausible symbol list, so nothing about it looks wrong -- it simply describes a
#: different repository than the one on disk, and ``load`` would happily serve it.  The
#: version is the only thing that can tell the two apart.
#:
#: History: 1 -> 2 when constant bindings became symbols.  MEASURED on 2026-09-23: that
#: change moved constant-request localisation from 9/86 to 86/86, so a stale cache would
#: silently keep the old number.
#:
#: History: 2 -> 3 when the index stopped being Python-only.  Before it, a Go, Rust, TSX
#: or CSS repository indexed *zero* files -- ``ast_search`` returned nothing and the only
#: tool that could locate a target was ``ripgrep_search``, whose process-spawn cost
#: (MEASURED 400-660 ms on this machine) exceeds the 0.5 s exploration budget on its own.
#: Four real-world scenarios (``rust_max_workers``, ``rust_retry_count``,
#: ``go_read_timeout``, ``go_server_port``) therefore located nothing at all and reported
#: a budget expiry as the failure.  A v2 cache holds an empty polyglot symbol list, so
#: serving it would silently keep the pre-change behaviour.
SCHEMA_VERSION = 3

#: Where the cached index is written inside the repo it describes.
#:
#: Named from ``SYLPHAE_ARTIFACT_PREFIX`` rather than written out, because the writer and the
#: readers used to each hold their own copy of this string.  A rename would then have left
#: the readers loading a stale file while the writer populated a new one -- and the file
#: was only excluded from project scanning by one of the discovery paths, so the copies
#: also decided which paths saw it as source.  One binding, imported by all of them.
REPO_INDEX_FILE_NAME = f"{SYLPHAE_ARTIFACT_PREFIX}_index.json"

#: Hard cap on indexed files. Beyond this the inventory is truncated (sorted, so the
#: truncation is deterministic).
MAX_FILES = 5000

#: Directories deeper than this (relative to the tree root) are not descended into.
MAX_DEPTH = 8

#: ``Candidate.snippet`` is never longer than this.
SNIPPET_MAX_CHARS = 600

#: At most this many symbol names are attached to a candidate, matched ones first.
MAX_CANDIDATE_SYMBOLS = 12

SKIP_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        "build",
        "dist",
        ".tox",
        ".eggs",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".idea",
        ".vscode",
        "work",
        "scratch",
        "runs",
        "corpus",
        "tmp",
        "temp",
    }
)

# --------------------------------------------------------------------------------------
# Ranker weights
# --------------------------------------------------------------------------------------

W_SYM = 3.0
W_SYM_EXACT = 4.0
W_PATH = 2.0
W_STEM = 2.5
W_IMPORT = 1.0
W_DEPTH = 0.15
W_TEST_PENALTY = 0.5

#: Structural prior: how central a file is in the import graph (normalised in-degree, i.e.
#: how many indexed files import it). ``setup.py``, ``docs/conf.py`` and ``bench/*`` are
#: imported by nobody; ``pkg/__init__.py`` and core modules are imported by everybody. Core
#: modules are edited more often than leaf scripts, and without this signal a request that
#: matches nothing lexically can never reach a file the depth prior has already penalised.
W_CENTRALITY = 0.3

#: Function words plus the generic commit verbs that appear in a large fraction of commit
#: subjects and therefore carry no file signal. ``test``/``tests`` are deliberately *not*
#: here: a request that mentions tests should be able to reach a test file.
STOPWORDS = frozenset(
    """
a an the and or but if then else when while for in on at by to of from into with without
is are was were be been being do does did done have has had will would shall should can
could may might must not no yes it its this that these those we you they he she i our your
their as so than too very just only also more most less least all any some none each every
up out over under again further once here there now new old
add adds added adding fix fixes fixed fixing
update updates updated updating change changes changed changing
remove removes removed removing support supports supported supporting
use uses used using make makes made making allow allows allowed
""".split()
)


# --------------------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z]+|[0-9]+")
#: Whole identifier-ish words, underscores included.  ``_WORD_RE`` deliberately drops
#: underscores, which is right for subtoken expansion and wrong for exact symbol
#: matching: ``_get_text_stderr`` has to survive as one token to be compared against
#: ``sym_names`` at all.
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def _subtokens(word: str) -> Iterator[str]:
    for part in word.split("_"):
        if not part:
            continue
        for m in _CAMEL_RE.finditer(part):
            yield m.group(0).lower()


def tokenize_text(text: str) -> list[str]:
    """Split text into lowercase identifier-ish tokens.

    Handles ``snake_case``, ``camelCase``, ``PascalCase``, dotted paths and dashes.
    Tokens shorter than two characters and stopwords are dropped.
    """
    out: list[str] = []
    for word in _WORD_RE.findall(text or ""):
        for tok in _subtokens(word):
            if len(tok) >= 2 and tok not in STOPWORDS:
                out.append(tok)
    return out


# --------------------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------------------

_TEST_DIR_NAMES = frozenset({"test", "tests", "testing"})


def is_test_path(path: str) -> bool:
    """True for files that look like test files.

    Deliberately generous: a false positive only changes a ranking prior or a candidate
    filter, and a false negative would let a test file outrank the source it tests.
    """
    parts = path.lower().split("/")
    name = parts[-1]
    if name == "conftest.py":
        return True
    if any(p in _TEST_DIR_NAMES for p in parts[:-1]):
        return True
    return name.startswith("test_") or name.endswith("_test.py") or name == "tests.py"


def module_name_of(path: str) -> str:
    """``src/pkg/mod.py`` -> ``src.pkg.mod``; ``src/pkg/__init__.py`` -> ``src.pkg``."""
    p = path[:-3] if path.endswith(".py") else path
    if p.endswith("/__init__"):
        p = p[: -len("/__init__")]
    return p.replace("/", ".")


def package_name_of(path: str) -> str:
    """The package a file lives in, as a dotted name relative to the tree root."""
    p = path[:-3] if path.endswith(".py") else path
    if p.endswith("/__init__"):
        return p[: -len("/__init__")].replace("/", ".")
    head, _, _tail = p.rpartition("/")
    return head.replace("/", ".")


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ImportRef:
    """One import edge as written in the source.

    ``module`` is the dotted module *as written* (``""`` only for ``from . import x``-style
    statements, which are recorded once per imported name instead). ``level`` is the number
    of leading dots for relative imports, ``0`` for absolute ones.
    """

    module: str
    is_relative: bool
    level: int


@dataclass
class FileRecord:
    """Everything the index knows about one source file."""

    path: str  # repo-relative posix path
    symbols: list[str]  # qualified names, top-level first
    symbol_spans: dict[str, tuple[int, int]]  # qualname -> 1-based inclusive (start, end)
    imports: list[ImportRef]
    n_lines: int
    parse_error: str | None  # None when the file parsed

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "symbols": list(self.symbols),
            "symbol_spans": {k: [int(v[0]), int(v[1])] for k, v in self.symbol_spans.items()},
            "imports": [[i.module, bool(i.is_relative), int(i.level)] for i in self.imports],
            "n_lines": int(self.n_lines),
            "parse_error": self.parse_error,
        }

    @staticmethod
    def from_dict(d: dict) -> "FileRecord":
        return FileRecord(
            path=str(d["path"]),
            symbols=[str(s) for s in d.get("symbols", [])],
            symbol_spans={
                str(k): (int(v[0]), int(v[1])) for k, v in (d.get("symbol_spans") or {}).items()
            },
            imports=[
                ImportRef(str(m), bool(r), int(end)) for m, r, end in (d.get("imports") or [])
            ],
            n_lines=int(d.get("n_lines", 0)),
            parse_error=d.get("parse_error"),
        )


@dataclass
class Candidate:
    """A ranked retrieval result. See INTERFACES.md section 3."""

    path: str
    score: float
    symbols: list[str]
    snippet: str
    snippet_span: tuple[int, int]


# --------------------------------------------------------------------------------------
# AST extraction
# --------------------------------------------------------------------------------------


def parse_source(source: str, filename: str = "<unknown>") -> ast.Module:
    """``ast.parse`` without the ``SyntaxWarning`` firehose.

    CPython warns on every invalid escape sequence in a string literal (``"\\s"`` inside a
    regex is the common case). A repository-wide index hits thousands of them and the
    warnings go to stderr, drowning the caller's own output and costing real time to
    format. The warning is about a string literal, not about the syntax tree we want, so
    it is suppressed here -- a genuine ``SyntaxError`` still propagates to the caller.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source, filename)


def _statement_lists(node: ast.AST) -> Iterator[list[ast.stmt]]:
    for attr in ("body", "orelse", "finalbody"):
        block = getattr(node, attr, None)
        if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
            yield block
    for handler in getattr(node, "handlers", None) or []:
        body = getattr(handler, "body", None)
        if isinstance(body, list) and body:
            yield body


def _bound_constant_names(node: ast.AST) -> list[str]:
    """Names bound by a module-level/class-level statement to a literal, if any.

    Only literal-valued bindings count.  ``x = f()`` is a computed value, not something a
    request can name a new value for, and indexing it would make every module-level
    temporary a symbol.
    """
    if isinstance(node, ast.Assign):
        value, targets = node.value, node.targets
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        value, targets = node.value, [node.target]
    else:
        return []
    if not isinstance(value, ast.Constant):
        return []
    return [t.id for t in targets if isinstance(t, ast.Name)]


def collect_symbols(tree: ast.AST) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """Qualified names of every function/class/method **and every constant binding**, with
    1-based inclusive spans.

    Descends through non-definition blocks (``if``/``try``/``with`` at module level) so a
    conditionally defined function is still indexed, but a definition inside a function
    gets the enclosing function's prefix (``outer.inner``).

    Constant bindings are indexed as well, and that is a measured requirement rather than
    a completeness gesture.  MEASURED on 2026-09-23 across four real repositories: a
    request naming a module-level constant ("把 DEFAULT_TIMEOUT 修改为 60") located its
    defining file through the symbol index **9 times out of 86**, because the index held
    only functions and classes.  That request shape is the reflex agent's core domain --
    the whole point of a *parameterised* op is choosing a literal value -- and its target
    was invisible to the most expensive tool in the stack.

    Function bodies are deliberately not scanned: a local variable is not addressable by a
    request that names it, and indexing locals would add noise to every symbol match.
    """
    names: list[str] = []
    spans: dict[str, tuple[int, int]] = {}

    def record(qual: str, node: ast.AST) -> None:
        if qual in spans:
            return
        names.append(qual)
        end = getattr(node, "end_lineno", None) or node.lineno
        spans[qual] = (int(node.lineno), int(end))

    def walk(node: ast.AST, prefix: str, in_function: bool) -> None:
        for child in getattr(node, "body", []) or []:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = prefix + child.name
                record(qual, child)
                walk(child, qual + ".", True)
            elif isinstance(child, ast.ClassDef):
                qual = prefix + child.name
                record(qual, child)
                walk(child, qual + ".", False)
            else:
                if not in_function:
                    for target in _bound_constant_names(child):
                        record(prefix + target, child)
                walk(child, prefix, in_function)

    walk(tree, "", False)
    return names, spans


def collect_imports(tree: ast.AST) -> list[ImportRef]:
    """Every import statement anywhere in the file, deduplicated, source order preserved."""
    out: list[ImportRef] = []
    seen: set[tuple[str, bool, int]] = set()

    def add(module: str, rel: bool, level: int) -> None:
        if not module:
            return
        key = (module, rel, level)
        if key not in seen:
            seen.add(key)
            out.append(ImportRef(module, rel, level))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name, False, 0)
        elif isinstance(node, ast.ImportFrom):
            rel = node.level > 0
            mod = node.module or ""
            add(mod, rel, node.level)
            for alias in node.names:
                if alias.name == "*":
                    continue
                # `from pkg import submodule` is a real edge to `pkg.submodule`, and
                # `from . import x` carries the module name in the alias rather than in
                # `node.module`. Both forms are recorded.
                add(f"{mod}.{alias.name}" if mod else alias.name, rel, node.level)
    return out


#: The non-Python extensions the index treats as source.  ``POLYGLOT_EXTENSIONS`` is the
#: shared definition of "source", and Python is handled by the AST path above, so this is
#: that table minus ``.py``.
_POLYGLOT_NON_PY = frozenset(POLYGLOT_EXTENSIONS - {".py"})

#: Upper bound on the number of polyglot symbols recorded per file.  ``collect_symbols``
#: is bounded by the AST, but the polyglot extractor's general fallback ("every identifier
#: of three or more characters") can yield hundreds of names from a single long file, and
#: the index pays for each one in the token sets built at construction.  The extractor
#: sorts longest-name-first, and the long names are the specific ones a request can name,
#: so truncation discards the generic tail rather than the signal.
_MAX_POLYGLOT_SYMBOLS = 96


def collect_polyglot_symbols(file_path: str, content: str) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """Names and approximate 1-based spans for a non-Python source file.

    ``extract_polyglot_symbols`` is regex-based and language-aware (Go ``func``/``type``,
    Rust ``fn``/``struct``/``const``, CSS selectors and variables, TSX components and
    class names), so this is the counterpart of :func:`collect_symbols` for the files the
    Python AST cannot read.  It never raises: a file whose content is unreadable is
    recorded with no symbols, exactly as a file that does not parse is.

    Spans are *approximate* -- the first line the name occurs on, as a whole word -- rather
    than exact AST extents.  They are used only to choose a snippet, and a snippet anchored
    on the defining line is what the caller needs; the span's end being the start line is
    honest about what it measured rather than pretending to a precision the extractor does
    not have.
    """
    names = extract_polyglot_symbols(file_path, content)
    if not names:
        return [], {}

    lines = content.splitlines()
    spans: dict[str, tuple[int, int]] = {}
    for name in names:
        pattern = re.compile(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])")
        for lineno, line in enumerate(lines, 1):
            if pattern.search(line):
                spans[name] = (lineno, lineno)
                break
        if len(spans) >= _MAX_POLYGLOT_SYMBOLS:
            break
    kept = list(spans)
    return kept, spans


def resolve_import_candidates(path: str, imp: ImportRef) -> list[str]:
    """Dotted module names an :class:`ImportRef` may denote, relative to the tree root.

    The tree root is not necessarily the package root (``src/`` layouts are common), so
    these are matched against *suffixes* of indexed module names rather than compared
    literally. See :meth:`RepoIndex._build_edges`.
    """
    if not imp.is_relative:
        return [imp.module] if imp.module else []
    base = package_name_of(path)
    parts = base.split(".") if base else []
    up = max(0, imp.level - 1)
    if up:
        parts = parts[: max(0, len(parts) - up)]
    base = ".".join(parts)
    cand = f"{base}.{imp.module}" if imp.module else base
    return [cand] if cand else []


# --------------------------------------------------------------------------------------
# Tree walking
# --------------------------------------------------------------------------------------


def iter_python_files(root: Path) -> list[str]:
    """Sorted repo-relative posix paths of every ``.py`` file under ``root``."""
    found: list[str] = []
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        depth = 0 if rel_dir == "." else len(Path(rel_dir).parts)
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in SKIP_DIRS and not d.endswith(".egg-info") and not d.startswith(".git")
        )
        if depth >= MAX_DEPTH:
            dirnames[:] = []
        for name in sorted(filenames):
            if not name.endswith(".py") or is_ignored_file(name):
                continue
            rel = name if rel_dir == "." else f"{rel_dir}/{name}"
            found.append(rel.replace(os.sep, "/"))
    found.sort()
    return found


def _read_source(root: Path, rel: str) -> str | None:
    try:
        return (Path(root) / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _record_for(root: Path, rel: str) -> FileRecord:
    src = _read_source(root, rel)
    if src is None:
        return FileRecord(rel, [], {}, [], 0, "read error")
    n_lines = len(src.splitlines())
    try:
        tree = parse_source(src, rel)
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        return FileRecord(rel, [], {}, [], n_lines, f"{type(exc).__name__}: {exc}")
    symbols, spans = collect_symbols(tree)
    return FileRecord(rel, symbols, spans, collect_imports(tree), n_lines, None)


def _record_for_polyglot(root: Path, rel: str) -> FileRecord:
    """A non-Python source file, recorded the same way a Python one is.

    Symbols come from the regex extractor rather than the AST, and imports are empty --
    there is no import graph to build across languages without a per-language parser, and
    a wrong graph is worse than none.  The file is still a full member of the inventory:
    it can be ranked, enriched with a snippet, and chosen as an edit target, which is the
    capability that was missing.
    """
    src = _read_source(root, rel)
    if src is None:
        return FileRecord(rel, [], {}, [], 0, "read error")
    symbols, spans = collect_polyglot_symbols(rel, src)
    return FileRecord(rel, symbols, spans, [], len(src.splitlines()), None)


def _symbol_kind(rec: "FileRecord", qual: str, lines: list[str]) -> str:
    """Coarse classification of one indexed name: what kind of thing does it name?

    Used only as a tie-break among symbols that matched the request equally, to prefer the
    one the request can actually *act on*.  Classification is deliberately conservative.
    Anything uncertain returns ``"other"`` and gets no boost, which is the safe default --
    the tie-break must never *invert* a genuine token win, only order symbols the request
    matched identically.

    ``lines`` are the record's own source lines, supplied by the caller (which already has
    them for snippet selection) rather than re-read here.
    """
    if rec.path.endswith(".py"):
        # ``collect_symbols`` records module/class-level constant bindings as plain names
        # (``DEFAULT_TIMEOUT``) and definitions as qualified ones (``Cls.method``).  A
        # bare name with no dot is therefore a constant or an imported alias, and a dotted
        # one is a definition -- the distinction the tie-break needs.
        return "value" if "." not in qual else "definition"
    # Polyglot: the extractor yields qualified names for CSS selectors and class
    # attributes, and bare identifiers for constants and struct members.  A bare
    # identifier whose defining line binds a literal is a value binding.
    span = rec.symbol_spans.get(qual)
    if span and _LINE_BINDS_A_VALUE(lines, span[0] - 1, len(lines)):
        return "value"
    return "other"


#: The shapes a line can take when it *binds a value*: an ``=`` before a literal, in any
#: of the polyglot languages the extractor covers.  Deliberately a shape test on the line
#: rather than a parse, because the extractor is regex-based and no parser is available;
#: the test is used only to break token ties, never to create a match.
_VALUE_LINE_RE = re.compile(
    r"^\s*(?:const\s+|let\s+|var\s+|static\s+|pub\s+(?:const|static)\s+)?"
    r"[A-Za-z0-9_]+\s*(?::\s*[^=]+)?=\s*(?:\d|[\"'`\[#]|[A-Za-z_][\w]*\s*[(<])"
)


def _LINE_BINDS_A_VALUE(lines: list[str], index: int, n: int) -> bool:
    """Whether ``lines[index]`` binds a literal value, with the const-block case.

    A Rust ``const (`` block opens on one line and its bindings sit on the following lines,
    so a span may point at ``DefaultPort`` on a line that holds no ``=`` of its own while
    the block header above does.  A Go ``const (`` block is the same shape.  The search
    walks up a few lines to the block opener and accepts either.
    """
    if not (0 <= index < n):
        return False
    if _VALUE_LINE_RE.match(lines[index]):
        return True
    for back in range(1, 4):
        if index - back < 0:
            break
        above = lines[index - back]
        if _VALUE_LINE_RE.match(above) and "(" in above:
            return True
        if above.strip() and not above.strip().startswith("//"):
            break
    return False


#: Tie-break boost for a symbol that *holds a value* the request can name a new one for.
#: See :meth:`RepoIndex._make_candidate` for the measured failure it addresses.  Kept
#: small and non-negative so it can only reorder an exact token tie, never outweigh one.
_VALUE_BINDING_BOOST: dict[str, float] = {"value": 1.0}


def iter_indexable_files(root: Path) -> list[str]:
    """Every file the index may record, Python first then the rest, sorted deterministically.

    Python files come from :func:`iter_python_files` (the AST path) and everything else
    from :func:`iter_project_files` filtered to the polyglot extensions.  The two walkers
    apply the same exclusion tables, so a directory skipped for one is skipped for both.

    Order matters only for which files survive ``MAX_FILES`` truncation, and Python-first
    is deliberate: an index that has to truncate is indexing a large polyglot tree, where
    the Python files are the ones with an import graph and exact spans, so keeping them is
    worth more than the hundred-and-first CSS file.
    """
    py = iter_python_files(root)
    py_set = set(py)
    rest = [
        rel
        for rel in iter_project_files(root, extensions=set(_POLYGLOT_NON_PY))
        if rel not in py_set
    ]
    return py + rest


# --------------------------------------------------------------------------------------
# Snippet selection
# --------------------------------------------------------------------------------------


def _clip(lines: Sequence[str], start: int, end: int) -> tuple[str, tuple[int, int]]:
    """First <= SNIPPET_MAX_CHARS characters of lines ``start..end`` (1-based inclusive).

    Returns the text and the 1-based inclusive span of the lines actually included, so the
    span always describes the snippet the caller received rather than the symbol it came
    from.
    """
    if not lines:
        return "", (1, 1)
    start = max(1, min(int(start), len(lines)))
    end = max(start, min(int(end), len(lines)))
    parts: list[str] = []
    total = 0
    last = start
    for ln in range(start, end + 1):
        text = lines[ln - 1]
        extra = len(text) + (1 if parts else 0)
        if total + extra > SNIPPET_MAX_CHARS:
            if not parts:  # a single monster line: hard truncate
                parts.append(text[:SNIPPET_MAX_CHARS])
                last = ln
            break
        parts.append(text)
        total += extra
        last = ln
    return "\n".join(parts), (start, last)


# --------------------------------------------------------------------------------------
# The index
# --------------------------------------------------------------------------------------


class RepoIndex:
    """Parsed world state of one source tree.

    Source text is *not* stored: the index keeps the tree root and re-reads files on
    demand when a snippet is requested (cached for the lifetime of the index). That keeps
    ``save``/``load`` small, at the cost that snippets are empty if the tree has moved.
    Retrieval itself never touches disk.
    """

    def __init__(self, root: Path, records: list[FileRecord], edges: list[tuple[int, int]]):
        self.root = Path(root)
        self._records: list[FileRecord] = list(records)
        self._edges: list[tuple[int, int]] = [tuple(e) for e in edges]
        self.files: list[str] = [r.path for r in self._records]

        self._path_tokens: list[set[str]] = []
        self._sym_tokens: list[set[str]] = []
        self._sym_names_lower: list[set[str]] = []
        self._stems: list[str] = []
        for rec in self._records:
            self._path_tokens.append(set(tokenize_text(rec.path)))
            sym_tokens: set[str] = set()
            for qual in rec.symbols:
                sym_tokens |= set(tokenize_text(qual))
            self._sym_tokens.append(sym_tokens)
            self._sym_names_lower.append({q.lower() for q in rec.symbols})
            self._stems.append(Path(rec.path).stem.lower())

        self._src_cache: dict[str, list[str]] = {}
        self._idf_cache: dict[str, float] | None = None
        self._centrality_cache: list[float] | None = None

    # -- construction ------------------------------------------------------------------

    @classmethod
    def build(cls, root: Path) -> "RepoIndex":
        """Walk ``root``, parse every Python file and record every other source file.

        Never raises for a malformed file: the file is recorded with ``symbols=[]`` and a
        ``parse_error`` reason and stays in the inventory.  Non-Python source is recorded
        through the regex extractor, which cannot fail on a language it does not know --
        it simply yields fewer names.

        The index is not Python-only.  Before polyglot recording existed, a Go, Rust or
        TSX repository indexed *zero* files: ``ast_search`` returned nothing and the only
        tool that could locate a target was ``ripgrep_search``, whose process-spawn cost
        (MEASURED 400-660 ms on this machine) exceeds the 0.5 s exploration budget on its
        own.  Four real-world scenarios then located nothing at all and reported a budget
        expiry as the failure, which reads as a timeout and is a missing index.
        """
        root = Path(root)
        records: list[FileRecord] = []
        for rel in iter_indexable_files(root):
            if len(records) >= MAX_FILES:
                break
            if rel.endswith(".py"):
                records.append(_record_for(root, rel))
            else:
                records.append(_record_for_polyglot(root, rel))
        return cls(root, records, cls._build_edges(records))

    @staticmethod
    def _build_edges(records: list[FileRecord]) -> list[tuple[int, int]]:
        """Import edges between indexed files as ``(importer, imported)`` index pairs.

        Import targets are matched against every dotted *suffix* of an indexed module name,
        because the tree root is not necessarily the package root. That over-matches
        (``import utils`` reaches every ``*/utils.py``), which is the safe direction here:
        the graph is only used to spread score to plausible neighbours and to rank files by
        how central they are.
        """
        by_suffix: dict[str, list[int]] = {}
        for idx, rec in enumerate(records):
            mod = module_name_of(rec.path)
            if not mod:
                continue
            parts = mod.split(".")
            for cut in range(len(parts)):
                by_suffix.setdefault(".".join(parts[cut:]), []).append(idx)

        edges: set[tuple[int, int]] = set()
        for idx, rec in enumerate(records):
            for imp in rec.imports:
                for target in resolve_import_candidates(rec.path, imp):
                    for other in by_suffix.get(target, ()):
                        if other != idx:
                            edges.add((idx, other))
        return sorted(edges)

    # -- persistence -------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": SCHEMA_VERSION,
            "root": str(self.root),
            "files": [r.to_dict() for r in self._records],
            "edges": [[int(a), int(b)] for a, b in self._edges],
        }

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RepoIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = payload.get("version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"RepoIndex schema version {version!r} != {SCHEMA_VERSION}; rebuild the index"
            )
        records = [FileRecord.from_dict(d) for d in payload.get("files", [])]
        edges = [(int(a), int(b)) for a, b in payload.get("edges", [])]
        return cls(Path(payload["root"]), records, edges)

    # -- introspection -----------------------------------------------------------------

    @property
    def records(self) -> list[FileRecord]:
        return list(self._records)

    #: The enrichment the trained target head expects, kept in one place so the
    #: training corpus provider and the production index cannot drift apart.
    ENRICH_SYMBOLS = 16
    ENRICH_SNIPPET_CHARS = 400

    def enrich(self, paths: Sequence[str], request: str = ""
               ) -> list[tuple[str, list[str], str]]:
        """Attach real file content to arbitrary paths, in the training format.

        ``neural_search`` reranks paths the lexical layer picked, but the head was
        trained on candidates carrying def/class names and a file prefix -- scoring
        it on bare paths measures a model that no longer exists and caps it at the
        lexical ceiling.  This returns ``(path, symbols, snippet)`` triples shaped
        exactly like ``SourceMapProvider``'s: up to ``ENRICH_SYMBOLS`` qualified
        names (already top-level first, see ``FileRecord``) and the first
        ``ENRICH_SNIPPET_CHARS`` characters of the file.

        A path the index does not hold comes back empty, so an unknown file is
        scored as no-content rather than guessed at.
        """
        by_path = {rec.path: rec for rec in self._records}
        out: list[tuple[str, list[str], str]] = []
        for path in paths:
            rec = by_path.get(path)
            if rec is None:
                out.append((path, [], ""))
                continue
            syms = list(rec.symbols)[: self.ENRICH_SYMBOLS]
            snippet = ""
            if self.ENRICH_SNIPPET_CHARS > 0:
                lines = self._source_lines(rec.path)
                if lines:
                    snippet = "\n".join(lines)[: self.ENRICH_SNIPPET_CHARS]
            out.append((path, syms, snippet))
        return out

    @property
    def edges(self) -> list[tuple[int, int]]:
        """Import edges as ``(importer_index, imported_index)`` pairs."""
        return list(self._edges)

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"RepoIndex(root={str(self.root)!r}, files={len(self._records)})"

    # -- scoring -----------------------------------------------------------------------

    def _idf(self) -> dict[str, float]:
        """Token -> IDF, computed from this index. Cached; the index is immutable."""
        if self._idf_cache is not None:
            return self._idf_cache
        n = max(1, len(self._records))
        df: dict[str, int] = {}
        for i in range(len(self._records)):
            for tok in self._path_tokens[i] | self._sym_tokens[i]:
                df[tok] = df.get(tok, 0) + 1
        self._idf_cache = {tok: math.log(1.0 + n / (1.0 + c)) for tok, c in df.items()}
        return self._idf_cache

    def _weight(self, tok: str, idf: dict[str, float], default: float) -> float:
        return idf.get(tok, default)

    def _centrality(self) -> list[float]:
        """Per-file normalised in-degree in the import graph, in ``[0, 1]``. Cached."""
        if self._centrality_cache is None:
            in_degree = [0] * len(self._records)
            for _importer, imported in self._edges:
                in_degree[imported] += 1
            top = max(in_degree) if in_degree else 0
            self._centrality_cache = [d / top if top else 0.0 for d in in_degree]
        return self._centrality_cache

    def _source_lines(self, rel: str) -> list[str]:
        cached = self._src_cache.get(rel)
        if cached is not None:
            return cached
        src = _read_source(self.root, rel)
        lines = src.splitlines() if src is not None else []
        self._src_cache[rel] = lines
        return lines

    def _make_candidate(self, idx: int, score: float, request_tokens: set[str]) -> Candidate:
        rec = self._records[idx]
        idf = self._idf()
        max_idf = max(idf.values()) if idf else 1.0

        # Which symbol does the request point at? Highest token overlap wins; ties break on
        # (start line, name) so the choice is deterministic.
        #
        # The tie-break is load-bearing for the reflex domain, and it is a measured failure
        # rather than an aesthetic preference.  ``把服务端口修改为9000`` matched the
        # ``Port`` *field* of a Go struct and the ``DefaultPort`` *constant* on the token
        # ``port``, and the request names a *value to change*, which only the constant
        # holds.  Both scored equally on token overlap, the (start line, name) tie-break
        # preferred whichever came first in the file, and the candidate anchored on the
        # struct declaration -- a line the parameterised path cannot edit.  A request that
        # names a value is answered by the binding that *holds* one.
        lines = self._source_lines(rec.path)
        scored: list[tuple[float, float, int, str, tuple[int, int]]] = []
        for qual, span in rec.symbol_spans.items():
            toks = set(tokenize_text(qual))
            sc = sum(self._weight(t, idf, max_idf) for t in request_tokens & toks)
            if request_tokens and qual.lower() in request_tokens:
                sc += self._weight(qual.lower(), idf, max_idf)
            scored.append(
                (sc, _VALUE_BINDING_BOOST.get(_symbol_kind(rec, qual, lines), 0.0),
                 span[0], qual, span)
            )
        scored.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))

        if scored and scored[0][0] > 0.0:
            span = scored[0][4]
            snippet, used = _clip(lines, span[0], span[1])
        else:
            snippet, used = _clip(lines, 1, max(1, len(lines)))

        ordered = [item[3] for item in scored] or list(rec.symbols)
        return Candidate(
            path=rec.path,
            score=float(score),
            symbols=ordered[:MAX_CANDIDATE_SYMBOLS],
            snippet=snippet,
            snippet_span=used,
        )

    def candidates(
        self, request: str, k: int = 8, exclude_tests: bool = False
    ) -> list[Candidate]:
        """Rank files against a natural-language request. Deterministic.

        Returns ``min(k, eligible)`` candidates, so a short list is only ever caused by a
        small index (or by ``exclude_tests``), never by silent truncation.
        """
        if k <= 0 or not self._records:
            return []

        request_tokens = set(tokenize_text(request))
        eligible = [
            i
            for i, rec in enumerate(self._records)
            if not (exclude_tests and is_test_path(rec.path))
        ]
        if not eligible:
            return []

        # A prompt that names an identifier has to be matchable against the *whole*
        # identifier, not only its subtokens.  ``tokenize_text`` splits on underscores,
        # so ``_get_text_stderr`` arrives as get/text/stderr and the exact-symbol path
        # (``W_SYM_EXACT``) can never fire -- the identifier that made the request
        # specific is discarded by the query.  MEASURED on
        # ``把 _get_text_stderr 重命名为 _get_text_stderr_v2``: the sole definition,
        # src/click/_winconsole.py, scored 20.787 against src/click/_compat.py's 25.635,
        # a file that defines ``get_text_stderr`` and ``_make_text_stream`` and matched
        # ``text``/``stderr`` more often without ever defining the name asked for.
        for ident in _IDENTIFIER_RE.findall(request):
            low = ident.lower()
            if len(low) >= 3:
                request_tokens.add(low)
        # And when a symbol *is* named exactly, its subtokens stop being evidence about
        # anything else.  The request asked for that symbol, so the ``text`` and
        # ``stderr`` inside its name refer to it -- crediting them to every file that
        # happens to share them is exactly what lets a pile of same-subtoken relatives
        # outrank the one file that defines the name.
        exact_names = {
            tok for tok in request_tokens
            if any(tok in self._sym_names_lower[i] for i in eligible)
        }
        explained: set[str] = set()
        for name in exact_names:
            explained.update(_subtokens(name))

        idf = self._idf()
        max_idf = max(idf.values()) if idf else 1.0

        base: dict[int, float] = {}
        for i in eligible:
            rec = self._records[i]
            sym_tokens = self._sym_tokens[i]
            path_tokens = self._path_tokens[i]
            sym_names = self._sym_names_lower[i]
            score = 0.0
            for tok in request_tokens:
                w = self._weight(tok, idf, max_idf)
                if tok in sym_tokens and tok not in explained:
                    score += W_SYM * w
                if tok in sym_names:
                    score += W_SYM_EXACT * w
                if tok in path_tokens and tok not in explained:
                    score += W_PATH * w
            stem = self._stems[i]
            if stem in request_tokens:
                score += W_STEM * self._weight(stem, idf, max_idf)
            base[i] = score

        # Import-graph proximity: score spreads along edges, scaled by the best raw score so
        # the bonus can never outrank a direct lexical hit.
        peak = max(base.values()) if base else 0.0
        if peak > 0.0:
            spread: dict[int, float] = {i: 0.0 for i in eligible}
            for a, b in self._edges:
                if a in spread and b in spread:
                    spread[a] += base[b]
                    spread[b] += base[a]
            for i in eligible:
                if spread[i]:
                    base[i] += W_IMPORT * spread[i] / peak

        # Structural priors.
        #
        # An earlier revision gated these on how much the request matched lexically, on the
        # theory that a prior should only break ties among matched candidates. Measured on
        # 500 real commits that was clearly worse -- recall@3 fell 0.680 -> 0.590 and
        # recall@8 fell 0.802 -> 0.762 -- because on a request that matches nothing (release
        # and version-bump subjects, a large slice of real history) the depth prior is not
        # noise, it is the *only* information available, and it is right more often than
        # chance. 33.8% of targets sit at the repository root and 44.4% at depth 1. The
        # prior is left ungated.
        centrality = self._centrality()
        for i in eligible:
            rec = self._records[i]
            prior = -W_DEPTH * rec.path.count("/")
            if is_test_path(rec.path):
                prior -= W_TEST_PENALTY
            base[i] += prior + W_CENTRALITY * centrality[i]

        ranked = sorted(eligible, key=lambda i: (-base[i], self._records[i].path))
        return [self._make_candidate(i, base[i], request_tokens) for i in ranked[:k]]
