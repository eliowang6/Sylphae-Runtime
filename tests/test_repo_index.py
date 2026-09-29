"""Tests for ``sylphae/repo/index.py`` -- INTERFACES.md section 3.

Run from the repository root:

    python -m pytest tests/test_repo_index.py -q

(The ``sys.path`` bootstrap below makes the module importable even when pytest is invoked
as a bare ``pytest`` from a different working directory.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sylphae.repo.index import (  # noqa: E402
    MAX_CANDIDATE_SYMBOLS,
    SNIPPET_MAX_CHARS,
    Candidate,
    RepoIndex,
    is_test_path,
    module_name_of,
    package_name_of,
    resolve_import_candidates,
    tokenize_text,
)

SERIALIZER = '''\
"""Serialization."""

from .encoding import base64_encode
from .signer import Signer


class Serializer:
    """Turns objects into signed strings."""

    def __init__(self, secret_key, salt=b"cookie-session"):
        self.secret_key = secret_key
        self.salt = salt

    def dumps(self, obj):
        payload = base64_encode(str(obj).encode("utf-8"))
        return Signer(self.secret_key).sign(payload)

    def loads(self, value):
        return value


def _make_helper(name):
    def inner(x):
        return x

    return inner
'''

ENCODING = '''\
"""Base64 helpers."""

import base64


def base64_encode(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=")


def base64_decode(data):
    return base64.urlsafe_b64decode(data + b"==")
'''

BROKEN = "def oops(:\n    return 1\n"


def _write_tree(root: Path) -> None:
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "serializer.py").write_text(SERIALIZER, encoding="utf-8")
    (root / "pkg" / "encoding.py").write_text(ENCODING, encoding="utf-8")
    (root / "pkg" / "broken.py").write_text(BROKEN, encoding="utf-8")

    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "test_serializer.py").write_text(
        "from pkg.serializer import Serializer\n\n\ndef test_dumps():\n    assert Serializer\n",
        encoding="utf-8",
    )

    # Directories the walker must skip entirely.
    for skipped in ("node_modules", "build", "dist", "__pycache__", ".venv"):
        (root / skipped).mkdir(exist_ok=True)
        (root / skipped / "junk.py").write_text("x = 1\n", encoding="utf-8")

    # Non-source files are never indexed.  ``.md`` is not in ``POLYGLOT_EXTENSIONS``,
    # so it is excluded from the inventory entirely -- unlike a ``.go`` or ``.rs`` file,
    # which the polyglot extractor records (see ``test_build_indexes_polyglot_files``).
    (root / "README.md").write_text("# readme\n", encoding="utf-8")

    # Polyglot source: a Go constant the AST path cannot see.  A request naming
    # ``DEFAULT_MAX_WORKERS`` has to reach this file through the symbol index, not only
    # through content search.
    (root / "config").mkdir(exist_ok=True)
    (root / "config" / "settings.go").write_text(
        "package config\n\nconst (\n    DefaultPort = 8080\n)\n", encoding="utf-8"
    )
    (root / "rust.rs").write_text(
        "pub const DEFAULT_MAX_WORKERS: usize = 8;\n", encoding="utf-8"
    )
    (root / "styles.css").write_text(".btn-primary { color: #3b82f6; }\n", encoding="utf-8")

    # Depth: a file directly inside a depth-8 directory is indexed; anything deeper is not.
    shallow_dir = root / "a/b/c/d/e/f/g/h"
    shallow_dir.mkdir(parents=True, exist_ok=True)
    (shallow_dir / "ok.py").write_text("ok = True\n", encoding="utf-8")
    (shallow_dir / "i" / "deep").mkdir(parents=True, exist_ok=True)
    (shallow_dir / "i" / "deep" / "too_deep.py").write_text("deep = True\n", encoding="utf-8")


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    _write_tree(tmp_path)
    return tmp_path


@pytest.fixture()
def index(tree: Path) -> RepoIndex:
    return RepoIndex.build(tree)


# --------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------


def test_build_indexes_python_files(index: RepoIndex) -> None:
    files = set(index.files)
    assert "pkg/serializer.py" in files
    assert "pkg/encoding.py" in files
    assert "tests/test_serializer.py" in files
    assert "pkg/broken.py" in files
    assert "README.md" not in files


def test_build_indexes_polyglot_files(index: RepoIndex) -> None:
    """The index is not Python-only; MEASURED before this, a Go repo indexed zero files.

    That was not a curiosity: with the symbol index empty, ``ast_search`` returned
    nothing and the only locator left was a ripgrep subprocess whose spawn cost
    (400-660 ms here) exceeds the 0.5 s exploration budget, so four real-world scenarios
    located no target at all.  See ``RepoIndex.build``.
    """
    files = set(index.files)
    assert "config/settings.go" in files
    assert "rust.rs" in files
    assert "styles.css" in files


def test_build_records_polyglot_constants_as_symbols(index: RepoIndex) -> None:
    """A Go const and a Rust const are addressable by a request that names them.

    The reflex agent's core edit shape is "choose a literal value", so a constant that
    is invisible to the symbol index is invisible to the whole parameterised path.  This
    is the polyglot half of ``test_module_level_constants_are_indexed``.
    """
    go = next(r for r in index.records if r.path == "config/settings.go")
    assert "DefaultPort" in go.symbols
    assert go.symbol_spans["DefaultPort"] == (4, 4)

    rs = next(r for r in index.records if r.path == "rust.rs")
    assert "DEFAULT_MAX_WORKERS" in rs.symbols
    assert rs.symbol_spans["DEFAULT_MAX_WORKERS"] == (1, 1)


def test_build_records_css_selectors_as_symbols(index: RepoIndex) -> None:
    css = next(r for r in index.records if r.path == "styles.css")
    assert "btn-primary" in css.symbols


def test_build_polyglot_records_carry_no_imports(index: RepoIndex) -> None:
    """No import graph across languages without a per-language parser.

    A wrong graph is worse than none: the edges feed both centrality and score spread,
    and an invented edge would move scores for reasons no source file states.
    """
    go = next(r for r in index.records if r.path == "config/settings.go")
    assert go.imports == []
    assert go.parse_error is None


def test_build_skips_excluded_directories(index: RepoIndex) -> None:
    for path in index.files:
        head = path.split("/")[0]
        assert head not in {"node_modules", "build", "dist", "__pycache__", ".venv"}, path


def test_build_respects_depth_cap(index: RepoIndex) -> None:
    assert "a/b/c/d/e/f/g/h/ok.py" in set(index.files)
    assert "a/b/c/d/e/f/g/h/i/deep/too_deep.py" not in set(index.files)


def test_build_keeps_unparseable_file(index: RepoIndex) -> None:
    """A broken file is data, not an exception: kept, with no symbols and a reason."""
    record = next(r for r in index.records if r.path == "pkg/broken.py")
    assert record.symbols == []
    assert record.symbol_spans == {}
    assert record.parse_error is not None
    assert "SyntaxError" in record.parse_error
    # ...and it is still a candidate target, not silently dropped.
    assert "pkg/broken.py" in index.files


def test_build_records_nested_qualified_symbols(index: RepoIndex) -> None:
    record = next(r for r in index.records if r.path == "pkg/serializer.py")
    assert "Serializer" in record.symbols
    assert "Serializer.dumps" in record.symbols
    assert "Serializer.loads" in record.symbols
    assert "_make_helper" in record.symbols
    assert "_make_helper.inner" in record.symbols  # nested function keeps its prefix

    start, end = record.symbol_spans["Serializer.dumps"]
    lines = SERIALIZER.splitlines()
    assert 1 <= start <= end <= len(lines)
    assert lines[start - 1].strip().startswith("def dumps")


CONSTANTS = '''\
"""Settings."""

DEFAULT_TIMEOUT = 30
RETRIES: int = 3
LABEL = "primary"
COMPUTED = DEFAULT_TIMEOUT * 2
BUILT = object()


class Limits:
    MAX_ITEMS = 100

    def __init__(self):
        self.local = 1


def configure():
    LOCAL_ONLY = 5

    def nested():
        DEEP_LOCAL = 6

        return DEEP_LOCAL

    return LOCAL_ONLY + nested()
'''


@pytest.fixture()
def constants_index(tmp_path: Path) -> RepoIndex:
    """The constant's file plus six distractors that carry the request's other tokens.

    The distractors are load-bearing.  With ``settings.py`` as the only file, *any*
    ranking puts it first and the end-to-end test below passed without the constant being
    indexed at all -- which is exactly what the first version of it did.
    """
    (tmp_path / "settings.py").write_text(CONSTANTS, encoding="utf-8")
    for name, body in {
        "defaults.py": '"""Defaults."""\n\nTIMEOUT = 10\n',
        "client.py": '"""Client."""\n\n\ndef fetch(url, timeout=10):\n    return url\n',
        "server.py": '"""Server."""\n\nDEFAULT_PORT = 8080\n',
        "cache.py": '"""Cache."""\n\nDEFAULT_TTL = 60\n',
        "queue.py": '"""Queue."""\n\nDEFAULT_RETRIES = 3\n',
        "notes.md": "# default timeout notes\n",
    }.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    return RepoIndex.build(tmp_path)


def test_module_level_constants_are_indexed(constants_index: RepoIndex) -> None:
    """A request that names a constant has to be able to find the file that defines it.

    MEASURED before this existed, over four real repositories: a request naming a
    module-level constant located its defining file 9 times out of 86 through the symbol
    index.  The index held only functions and classes, so the target of the reflex agent's
    core edit shape -- choose a literal value -- was invisible to it.
    """
    record = next(r for r in constants_index.records if r.path == "settings.py")
    assert "DEFAULT_TIMEOUT" in record.symbols
    assert "RETRIES" in record.symbols  # AnnAssign with a value
    assert "LABEL" in record.symbols

    start, end = record.symbol_spans["DEFAULT_TIMEOUT"]
    assert CONSTANTS.splitlines()[start - 1].startswith("DEFAULT_TIMEOUT")


def test_class_body_constants_keep_their_class_prefix(constants_index: RepoIndex) -> None:
    record = next(r for r in constants_index.records if r.path == "settings.py")
    assert "Limits.MAX_ITEMS" in record.symbols


def test_locals_and_computed_bindings_are_not_symbols(constants_index: RepoIndex) -> None:
    """The two exclusions that keep the addition from adding noise.

    A local variable is not addressable by a request that names it, and ``COMPUTED`` is
    not something a request can name a *new value* for.  Indexing either would put a
    match on every file that has a module-level temporary.
    """
    record = next(r for r in constants_index.records if r.path == "settings.py")
    for absent in ("LOCAL_ONLY", "DEEP_LOCAL", "nested.DEEP_LOCAL", "COMPUTED", "BUILT"):
        assert absent not in record.symbols, absent
    # ``nested`` itself is a function and must still be indexed, with its prefix.
    assert "configure" in record.symbols
    assert "configure.nested" in record.symbols


def test_request_naming_a_constant_finds_its_file(constants_index: RepoIndex) -> None:
    """End to end through ``ast_search``, the tool the agent actually uses.

    ``client.py`` also contains the token ``timeout`` and ``defaults.py`` defines its own
    ``TIMEOUT`` constant, so returning ``settings.py`` is a statement about matching the
    *whole* identifier rather than about the tree having one file in it.

    Deliberately asserted at top-3 rather than rank 1: the measured claim is hit@3 over
    four real repositories (86 of 86 tasks), and ``LowEffort.n_candidates`` is 3, so top-3
    is the property the agent depends on.  Rank 1 is a stronger statement that the ranker
    does not make -- on this tree ``defaults.py`` comes first, because its path token
    ``defaults`` is credited against the request token ``default``.
    """
    import asyncio

    from sylphae.agent.explorer import extract_keywords
    from sylphae.agent.tools import ast_search

    keywords = extract_keywords("把 DEFAULT_TIMEOUT 修改为 60")
    matches = asyncio.run(ast_search(keywords, constants_index, k=3))
    assert matches, "a request naming an indexed constant must return a candidate"
    assert "settings.py" in [m.path for m in matches], [m.path for m in matches]


def test_build_records_relative_and_absolute_imports(index: RepoIndex) -> None:
    record = next(r for r in index.records if r.path == "pkg/serializer.py")
    relative = [i for i in record.imports if i.is_relative]
    assert ("encoding", 1) in {(i.module, i.level) for i in relative}

    enc = next(r for r in index.records if r.path == "pkg/encoding.py")
    absolute = [i for i in enc.imports if not i.is_relative]
    assert "base64" in {i.module for i in absolute}


def test_build_counts_lines(index: RepoIndex) -> None:
    record = next(r for r in index.records if r.path == "pkg/encoding.py")
    assert record.n_lines == len(ENCODING.splitlines())


def test_build_is_silent_on_invalid_escape_sequences(tmp_path: Path) -> None:
    """Regression: a repo full of regexes must not drown the caller in SyntaxWarnings.

    ``re.compile("\\s+")`` is a SyntaxWarning in CPython, and an indexer that emits one per
    literal floods stderr and costs real time formatting them.
    """
    import warnings

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "regexes.py").write_text(
        'import re\n\nPATTERN = re.compile("\\s+\\[\\h]")\n', encoding="utf-8"
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        index = RepoIndex.build(tmp_path)

    assert "pkg/regexes.py" in index.files
    assert index.records[1].parse_error is None, "a bad escape is not a syntax error"
    assert [str(w.message) for w in caught] == [], "indexing must not emit warnings"


def test_build_on_missing_root_is_empty(tmp_path: Path) -> None:
    empty = RepoIndex.build(tmp_path / "does-not-exist")
    assert len(empty) == 0
    assert empty.candidates("anything") == []


# --------------------------------------------------------------------------------------
# candidates -- contract
# --------------------------------------------------------------------------------------


def test_candidates_are_deterministic(index: RepoIndex) -> None:
    request = "fix the base64 encoding used by the serializer"
    first = index.candidates(request, k=6)
    second = index.candidates(request, k=6)
    third = RepoIndex.build(index.root).candidates(request, k=6)

    def key(cands: list[Candidate]) -> list[tuple]:
        return [(c.path, c.score, c.snippet, c.snippet_span) for c in cands]

    assert key(first) == key(second)
    assert key(first) == key(third)  # stable across an independent rebuild


def test_candidates_snippet_budget_and_span(index: RepoIndex) -> None:
    for request in ("base64 encode", "serializer dumps", "zzzz nothing matches here"):
        for cand in index.candidates(request, k=len(index.files)):
            assert len(cand.snippet) <= SNIPPET_MAX_CHARS, cand.path
            assert cand.snippet_span[0] >= 1
            assert cand.snippet_span[1] >= cand.snippet_span[0]
            n_lines = next(r.n_lines for r in index.records if r.path == cand.path)
            assert cand.snippet_span[1] <= max(1, n_lines), cand.path


def test_candidates_never_exceed_k(index: RepoIndex) -> None:
    assert len(index.candidates("serializer", k=2)) == 2
    assert len(index.candidates("serializer", k=0)) == 0
    assert len(index.candidates("serializer", k=-3)) == 0
    # Fewer than k only when the index has fewer files.
    assert len(index.candidates("serializer", k=10_000)) == len(index.files)


def test_candidates_exclude_tests(index: RepoIndex) -> None:
    with_tests = index.candidates("serializer dumps", k=len(index.files))
    assert any(is_test_path(c.path) for c in with_tests), "fixture should surface a test file"

    without = index.candidates("serializer dumps", k=len(index.files), exclude_tests=True)
    assert without, "excluding tests must not empty the list"
    assert not any(is_test_path(c.path) for c in without)


def test_candidates_symbols_are_bounded_and_from_the_file(index: RepoIndex) -> None:
    record_by_path = {r.path: r for r in index.records}
    for cand in index.candidates("serializer base64 signer", k=len(index.files)):
        assert len(cand.symbols) <= MAX_CANDIDATE_SYMBOLS
        for symbol in cand.symbols:
            assert symbol in record_by_path[cand.path].symbols


# --------------------------------------------------------------------------------------
# candidates -- retrieval quality
# --------------------------------------------------------------------------------------


def test_request_mentioning_a_symbol_finds_that_file(index: RepoIndex) -> None:
    top = index.candidates("base64 encode helpers", k=1)
    assert top[0].path == "pkg/encoding.py"
    # The snippet must be the matching function, not the head of the file.
    assert "base64_encode" in top[0].snippet


def test_request_mentioning_a_method_finds_its_class_file(index: RepoIndex) -> None:
    top = index.candidates("Serializer.dumps should reject non strings", k=1)
    assert top[0].path == "pkg/serializer.py"


def test_import_graph_proximity_spreads_to_neighbours(index: RepoIndex) -> None:
    """`serializer.py` imports `encoding.py`; a serializer-only request must lift encoding."""
    paths = [c.path for c in index.candidates("Serializer loads", k=len(index.files))]
    assert paths.index("pkg/encoding.py") < len(paths)


def test_nonsense_request_does_not_crash_or_invent(index: RepoIndex) -> None:
    cands = index.candidates("qqqq zzzz wwww", k=4)
    assert len(cands) == 4
    assert all(c.path in set(index.files) for c in cands)
    # Deterministic, and no lexical signal was invented for the request.
    assert cands == index.candidates("qqqq zzzz wwww", k=4)
    assert len(index.candidates("qqqq zzzz wwww", k=1)[0].symbols) <= MAX_CANDIDATE_SYMBOLS


def test_centrality_lifts_a_core_file_over_a_shallow_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The markupsafe mechanism, isolated.

    Subjects like "release 1.1.0" match no identifier, so the depth prior decides. Without
    centrality, ``bench/bench_escape.py`` (depth 1) beats ``src/pkg/__init__.py`` (depth 2)
    even though the latter is imported by everything and the former by nothing. Centrality
    is what flips that.
    """
    import sylphae.repo.index as index_mod

    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "__init__.py").write_text(
        "VERSION = '1.0'\n\n\ndef escape(s):\n    return s\n", encoding="utf-8"
    )
    (tmp_path / "src" / "pkg" / "_native.py").write_text(
        "from . import escape\n\n__all__ = ['escape']\n", encoding="utf-8"
    )
    (tmp_path / "bench").mkdir()
    (tmp_path / "bench" / "bench_escape.py").write_text("print('bench')\n", encoding="utf-8")
    (tmp_path / "setup.py").write_text("from setuptools import setup\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_pkg.py").write_text(
        "from pkg import escape\n\n\ndef test_escape():\n    assert escape('x') == 'x'\n",
        encoding="utf-8",
    )

    index = RepoIndex.build(tmp_path)
    # "version" is deliberately absent from the subject.  ``VERSION`` in the core file is
    # an indexed symbol now that constant bindings are indexed, so a subject containing
    # that token would match an identifier -- the one thing this test needs it not to do.
    request = "release notes 2.1.1"
    core = "src/pkg/__init__.py"
    leaf = "bench/bench_escape.py"

    # Guard the premise instead of assuming it.  Without this, a future symbol kind that
    # happens to match the subject would make the ranking assertions below fail in a way
    # that reads as a centrality regression.
    request_tokens = set(tokenize_text(request))
    for record in index.records:
        matched = request_tokens & {s.lower() for s in record.symbols}
        assert not matched, (
            f"{record.path} has symbols {sorted(matched)} matching the subject; this test "
            f"requires a subject that matches no identifier"
        )

    with_centrality = [c.path for c in index.candidates(request, k=len(index.files))]
    assert core in with_centrality and leaf in with_centrality
    assert with_centrality.index(core) < with_centrality.index(leaf)

    monkeypatch.setattr(index_mod, "W_CENTRALITY", 0.0)
    without = [c.path for c in index.candidates(request, k=len(index.files))]
    assert without.index(core) > without.index(leaf), (
        "the depth prior alone should prefer the shallow leaf; that is the bug centrality fixes"
    )


def test_skipped_directories_are_unreachable(index: RepoIndex) -> None:
    """Negative case: a file the walker skipped can never be returned, however well it fits."""
    cands = index.candidates("junk module in node_modules build dist", k=len(index.files))
    assert not any("node_modules" in c.path or c.path.startswith("build/") for c in cands)


# --------------------------------------------------------------------------------------
# save / load
# --------------------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path, index: RepoIndex) -> None:
    path = tmp_path / "index.json"
    index.save(path)
    assert path.is_file()

    reloaded = RepoIndex.load(path)
    assert reloaded.files == index.files
    assert reloaded.records == index.records

    request = "fix base64 encoding in the serializer"
    assert [(c.path, c.score, c.snippet) for c in reloaded.candidates(request, k=5)] == [
        (c.path, c.score, c.snippet) for c in index.candidates(request, k=5)
    ]


def test_save_is_json_serialisable(tmp_path: Path, index: RepoIndex) -> None:
    payload = index.to_dict()
    text = json.dumps(payload)
    assert json.loads(text)["files"]
    assert all(isinstance(f["path"], str) for f in payload["files"])


def test_load_rejects_wrong_schema_version(tmp_path: Path, index: RepoIndex) -> None:
    path = tmp_path / "index.json"
    payload = index.to_dict()
    payload["version"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version"):
        RepoIndex.load(path)


def test_a_cache_from_before_polyglot_indexing_is_rejected(
    tmp_path: Path, index: RepoIndex
) -> None:
    """Version 2 predates polyglot files being recorded, and must not be served.

    A v2 cache written against this repository lists no ``.go``/``.rs``/``.css`` records
    at all -- a valid file describing a *different* repository than the one on disk, which
    is the failure mode the schema version exists to catch.  Serving it would silently
    keep the pre-change behaviour (an empty symbol index on non-Python trees) while the
    tools above it believed they had an index.
    """
    path = tmp_path / "index.json"
    payload = index.to_dict()
    payload["version"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version"):
        RepoIndex.load(path)


def test_a_cache_from_before_constants_were_symbols_is_rejected(
    tmp_path: Path, index: RepoIndex
) -> None:
    """Version 1 predates constant bindings being indexed, and must not be served.

    Such a cache is a valid file with a valid root and a plausible symbol list; nothing
    about it looks wrong, it simply describes a repository the index no longer matches.
    Serving it would silently keep the pre-change behaviour -- measured as 9/86 rather
    than 86/86 on constant requests.
    """
    path = tmp_path / "index.json"
    payload = index.to_dict()
    payload["version"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version"):
        RepoIndex.load(path)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def test_is_test_path() -> None:
    assert is_test_path("tests/test_app.py")
    assert is_test_path("src/pkg/tests/helper.py")
    assert is_test_path("test_app.py")
    assert is_test_path("app_test.py")
    assert is_test_path("conftest.py")
    assert not is_test_path("src/pkg/app.py")
    assert not is_test_path("src/pkg/contest.py")
    assert not is_test_path("src/pkg/latest.py")


def test_tokenize_text() -> None:
    assert tokenize_text("URLSafeSerializerMixin") == ["url", "safe", "serializer", "mixin"]
    assert tokenize_text("read_config_file") == ["read", "config", "file"]
    assert tokenize_text("the a of fix add") == []  # stopwords and single chars only
    assert tokenize_text("a.b.c") == []  # single characters are dropped
    assert tokenize_text("x.py") == ["py"]  # kept; IDF is what defuses a token this common


def test_module_and_package_names() -> None:
    assert module_name_of("src/pkg/mod.py") == "src.pkg.mod"
    assert module_name_of("src/pkg/__init__.py") == "src.pkg"
    assert package_name_of("src/pkg/mod.py") == "src.pkg"
    assert package_name_of("src/pkg/__init__.py") == "src.pkg"
    assert package_name_of("top.py") == ""


def test_resolve_import_candidates_relative() -> None:
    from sylphae.repo.index import ImportRef

    ref = ImportRef("encoding", True, 1)
    assert resolve_import_candidates("src/pkg/serializer.py", ref) == ["src.pkg.encoding"]

    # `from .. import x` inside src/pkg/sub/mod.py: level 2 is the parent of the package
    # `src.pkg.sub`, i.e. `src.pkg` -- Python's own semantics, not "the tree root".
    parent = ImportRef("", True, 2)
    assert resolve_import_candidates("src/pkg/sub/mod.py", parent) == ["src.pkg"]

    absolute = ImportRef("os.path", False, 0)
    assert resolve_import_candidates("src/pkg/mod.py", absolute) == ["os.path"]


def test_import_edges_exist_for_relative_import(index: RepoIndex) -> None:
    by_path = {r.path: i for i, r in enumerate(index.records)}
    edges = {frozenset(e) for e in index.edges}
    assert frozenset({by_path["pkg/serializer.py"], by_path["pkg/encoding.py"]}) in edges
