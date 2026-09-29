"""Parallel exploration tools for Sylphae agent.

Implements candidate search across multiple modalities:
1. ripgrep_search: regex/literal content search in files
2. ast_search: AST symbol name and span search via RepoIndex
3. git_history_search: recent commit log & modified file search
4. path_lexical_search: path token scoring
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from sylphae.repo.ignore import IgnoreRules
from sylphae.repo.index import (
    REPO_INDEX_FILE_NAME,
    RepoIndex,
    is_test_path,
    tokenize_text,
)
from sylphae.repo.polyglot import (
    SYLPHAE_ARTIFACT_PREFIX,
    IGNORED_DIRS,
    IGNORED_FILE_SUFFIXES,
    POLYGLOT_EXTENSIONS,
    is_sylphae_artifact,
    load_ignore_rules,
)

SKIP_DIRS: frozenset[str] = frozenset(IGNORED_DIRS) | {".ruff_cache", ".mypy_cache"}

#: How many distinct files a single ripgrep invocation may contribute before the scan
#: stops collecting.  It has to be generous, because ripgrep emits matches in filesystem
#: walk order: the smaller this bound, the more the *walk* decides which files are even
#: eligible, and the ranking below can only reorder what it was given.  The previous bound
#: was ``max_matches * 3`` (45 files for the agent's cap of 15), and MEASURED on starlette
#: the gold file for a constant request sat at median rank 56 -- outside the pool entirely.
_RIPGREP_MAX_FILES: int = 400

#: Wall-clock bound on a single ripgrep invocation.  It exists because the call runs in
#: a worker thread (see ``ripgrep_search``), where a cancellation cannot reach it: the
#: thread has to be allowed to finish on its own.  A timeout here means the process was
#: pathological, so the caller falls back to the pure-Python search rather than treating
#: an unfinished search as an empty one.
_RIPGREP_TIMEOUT_S: float = 10.0


def _content_match_terms(keywords: Sequence[str]) -> list[str]:
    """Regex fragments for content search, one per keyword.

    Deliberately a plain substring test, in both the ripgrep path and the Python
    fallback: this is the *discovery* stage, where finding too much is cheap (the
    results are ranked and capped) and finding too little is not.  Narrowing it to word
    boundaries looks tidier and is wrong -- it drops ``ReadTimeout`` for the keyword
    ``timeout``, which is the match the request actually wanted, and three benchmark
    scenarios stopped resolving because of it.

    Precision belongs in anchor selection instead: see ``_placement_rank``, which picks
    the line to edit from among a file's matches.  Substring discovery plus a ranked
    anchor handles the ``port``/``import`` collision correctly without giving up
    ``timeout``/``ReadTimeout``, and neither rule can do it alone.
    """
    return [re.escape(kw) for kw in keywords]


def _ripgrep_exclude_globs() -> list[str]:
    """``--glob`` exclusions derived from the shared ignore tables.

    The executable path in ``ripgrep_search`` does not go through
    ``iter_project_files``, so it needs the same exclusions spelled as globs.  Building
    them from the tables rather than writing them out keeps the two discovery paths from
    disagreeing.

    The ``**/`` prefix is load-bearing.  ripgrep matches a bare ``dir/**`` only against
    paths relative to the search root, and this call always passes an absolute root, so
    the un-prefixed form silently matches nothing -- measured: searching an absolute root
    holding ``work/a.py`` still returned it under ``--glob '!work/**'`` and only dropped it
    under ``--glob '!**/work/**'``.  The previous hardcoded list used the bare form, so
    every one of those exclusions was inert.  A leading ``**/`` also matches ``dir/`` at
    any depth, which is what ``iter_project_files`` does when it prunes by name.
    """
    globs: list[str] = []
    for directory in sorted(IGNORED_DIRS):
        globs += ["--glob", f"!**/{directory}/**"]
    globs += ["--glob", f"!**/{SYLPHAE_ARTIFACT_PREFIX}*"]
    for suffix in IGNORED_FILE_SUFFIXES:
        globs += ["--glob", f"!*{suffix}"]
    # Not project source anywhere, and large enough to dominate a match budget.  The
    # bare ``*.min.*`` form is wider than ``IGNORED_FILE_SUFFIXES``, which enumerates the
    # two extensions that exist today; both are kept so content search does not widen back.
    globs += [
        "--glob", "!*.min.*",
        "--glob", "!*.log",
        "--glob", "!*.jsonl",
        "--glob", "!*.safetensors",
    ]
    return globs


#: A line whose value is a literal -- a number or a quoted string -- rather than a
#: reference to another name.  Changing "the port" means changing ``DefaultPort = 8080``,
#: not ``Port: DefaultPort,``, which only names it.
_LITERAL_VALUE = re.compile(r"[:=]\s*(?:\d|[\"'])")
_DEFINITION_SHAPE = re.compile(r"^[A-Za-z0-9_]+\s*[:=]")


def _placement_rank(snippet: str) -> tuple[int, int]:
    """How good a placement anchor a line makes, among one file's matches.

    The first line a content search hits is frequently the least useful one.  A request
    about a service's port matched ``Port int`` inside a struct while the value to change
    was ``DefaultPort = 8080`` in a const block further down; anchoring there leaves the
    placement guard with nothing to agree with, so the request could not be completed at
    all even though the right file had been found.

    Ranked on the property the patch builder keys off -- a line holding a literal value
    beats one that only references a name -- so the anchor and the edit agree about
    which line the request is about.
    """
    text = snippet.strip()
    return (
        int(bool(_LITERAL_VALUE.search(text))),
        int(bool(_DEFINITION_SHAPE.match(text)) or "const " in text),
    )


def _is_source_path(rel_path: str) -> bool:
    """Whether a path is source code, using the shared extension table.

    Not a new list: ``POLYGLOT_EXTENSIONS`` is already the project's definition of what
    source is, and a second one would drift from it.
    """
    return os.path.splitext(rel_path)[1].lower() in POLYGLOT_EXTENSIONS


def _rank_content_matches(
    best_per_file: dict[str, "ToolMatch"],
    file_kws: dict[str, set[str]],
    position: dict[str, int],
    max_matches: int,
) -> list["ToolMatch"]:
    """Order content matches by how *specific* their evidence is, then by source-ness.

    The keyword set a request expands into is deliberately broad: ``extract_keywords``
    adds an identifier *and* its subtokens, so ``DEFAULT_TIMEOUT`` contributes ``default``
    and ``timeout`` as well.  Scoring every match the same lets the file that happens to
    contain the common subtoken compete with the file that contains the whole identifier,
    and on volume the common one wins.

    Weights are inverse document frequencies over the matches actually observed, so they
    are calibrated by the repository rather than by a constant: a keyword that matched
    every file carries no information and is worth almost nothing, and no magic number
    decides where that cutoff sits.

    Source files sort ahead of non-source ones outright rather than being multiplied by a
    penalty.  MEASURED before this existed: the top three files for a request about a
    service's port were ``uv.lock``, ``pyproject.toml`` and ``README.md``.
    ``is_test_path`` is the same idea one level down: ``_is_source_path`` admits every
    ``.py`` file, so a test and the source it tests tie on the primary key.  Sorting the
    test second keeps the promise in ``is_test_path``'s own docstring ("a false negative
    would let a test file outrank the source it tests") at the point where it was being
    broken.
    """
    n_files = len(best_per_file) or 1
    df: dict[str, int] = {}
    for kws in file_kws.values():
        for kw in kws:
            df[kw] = df.get(kw, 0) + 1
    total_weight = sum(math.log(1.0 + n_files / d) for d in df.values()) or 1.0

    scored: list[tuple[int, float, int, str]] = []
    for path, match in best_per_file.items():
        weight = sum(math.log(1.0 + n_files / df[kw]) for kw in file_kws.get(path, ()))
        # A file *named* the concept is the concept's home; a file that merely mentions
        # the concept is a reference to it, and a reference is not worth more than the
        # definition.  MEASURED on ``把导航栏的高度缩短一些`` against a two-file repo:
        # the rarest keyword ``navbar`` (df = 1, the highest-information term in the
        # whole request) matched only ``tests/test_navbar.py``, because that file spells
        # ``src.navbar`` inside an import.  ``src/navbar.py`` never writes the word
        # "navbar" anywhere in its own body -- it *is* the word.  So the defining file
        # lost the single most informative keyword to the file that imports it, ripgrep
        # scored the importer 1.000 and the definition 0.608, and the reflex agent then
        # edited the test.  Creditting a keyword against the file whose own name it is
        # restores the definition's share.  Whole path components only: ``tokenize_text``
        # would also split ``test_navbar`` into ``test``+``navbar`` and hand the credit
        # right back to the reference.
        lower_parts = path.lower().split("/")
        self_names = set(lower_parts[:-1]) | {os.path.splitext(lower_parts[-1])[0]}
        weight += sum(
            math.log(1.0 + n_files / d) for kw, d in df.items() if kw in self_names
        )
        score = weight / total_weight
        # A line that declares something beats a line that only mentions it -- the same
        # property the anchor choice uses, applied to ranking.
        score += 0.05 * _placement_rank(match.snippet)[0]
        match.score = max(0.05, min(1.0, score))
        scored.append(
            (
                (0 if _is_source_path(path) else 1),
                1 if is_test_path(path) else 0,
                -match.score,
                position[path],
                path,
            )
        )

    scored.sort()
    return [best_per_file[p] for _, _, _, _, p in scored[:max_matches]]


@dataclass
class ToolMatch:
    """A candidate match discovered by an exploration tool."""
    path: str
    source_tool: str
    score: float
    snippet: str = ""
    line: int | None = None
    symbols: list[str] = field(default_factory=list)


def get_ripgrep_binary() -> str | None:
    """Locate ripgrep executable in PATH or standard development tool locations."""
    bin_path = shutil.which("rg")
    if bin_path:
        return bin_path
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    prog_files = os.environ.get("PROGRAMFILES", "")
    candidates = [
        os.path.join(local_app_data, "copilot", "pkg", "win32-x64", "1.0.73", "ripgrep", "bin", "win32-x64", "rg.exe"),
        os.path.join(prog_files, "raccoon-ai", "resources", "rg", "rg.exe"),
        os.path.join(prog_files, "Microsoft VS Code", "resources", "app", "node_modules.asar.unpacked", "@vscode", "ripgrep", "bin", "rg.exe"),
        os.path.join(prog_files, "WorkBuddy", "resources", "app.asar.unpacked", "cli", "vendor", "ripgrep", "x64-win32", "rg.exe"),
        os.path.join(prog_files, "Unity Hub", "resources", "app.asar.unpacked", "node_modules", "@vscode", "ripgrep-win32-x64", "bin", "rg.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


# ---------------------------------------------------------------------------
# 1. ripgrep search (with pure-Python fallback)
# ---------------------------------------------------------------------------

def _ignored_by_repo_rules(rules: IgnoreRules, rel_path: str) -> bool:
    """True when ``rel_path``, or any directory containing it, is declared ignored.

    A directory rule such as ``work/`` excludes everything *under* ``work``, but it
    matches only the directory itself: ``ignores_file("work/a.py", "a.py")`` is False
    for it.  A directory walk gets that for free by pruning; applying the same rules to
    a search *result* has to walk the ancestors explicitly.
    """
    parts = [p for p in rel_path.split("/") if p]
    for i in range(len(parts) - 1):
        if rules.ignores_dir("/".join(parts[: i + 1]), parts[i]):
            return True
    return bool(parts) and rules.ignores_file(rel_path, parts[-1])


#: A keyword specific enough to be searched on its own.  The point is not the
#: length but the shape: an ``_``-bearing snake_case token (``http_502_bad_gateway``)
#: names one identifier, while a bare fragment (``http``) names a concept that
#: occurs in nearly every file of a web framework.  ``camelCase`` of this length is
#: also specific; the bound is deliberately conservative because a short camelCase
#: name like ``getUser`` is still a common word.
_IDENTIFIER_KEYWORD_RE = re.compile(r"^(?=[A-Za-z0-9_]{6,}$)[a-z0-9_]*_[a-z0-9_]+$")


def _is_identifier_keyword(kw: str) -> bool:
    """Whether a keyword should be searched separately from the subtoken group.

    The split this enables is load-bearing, and the reason is a measured retrieval
    failure rather than a ranking intuition.  ``ripgrep_search`` used to pass every
    keyword to one ``rg`` alternation under ``--max-count 3``.  On
    ``work/repos/encode__starlette`` for ``把 HTTP_502_BAD_GATEWAY 修改为 503`` the
    first three matches inside ``starlette/status.py`` are the file's header
    comments -- ``HTTP codes``, ``See HTTP Status Code Registry``, the IANA URL --
    which contain only the common token ``http``.  The defining line
    ``HTTP_502_BAD_GATEWAY = 502`` sits 145 lines further down and is truncated away
    by the per-file budget, so the gold file entered ranking with the keyword set
    ``{'http'}``, tied at the same idf-degraded score as 97 unrelated files, and
    ranked 34th.  Raising ``--max-count`` does not help -- that file defines 60-odd
    ``HTTP_*`` constants, so any per-file line budget is spent on the quoted
    ``"HTTP_XXX",`` lines first; measured across max-count 3/5/8/12/20 the gold's
    keyword set stayed ``{'http'}`` and its rank stayed 34-42.

    Searching the identifier alone recovers it in one call: that alternation matches
    almost nothing, so the definition line is not competing for a budget.  With the
    two results merged the gold carries ``{'http', 'http_502_bad_gateway'}`` and
    ranks 1 of 98.
    """
    k = kw.strip().lower()
    return bool(_IDENTIFIER_KEYWORD_RE.match(k))


async def ripgrep_search(
    keywords: Sequence[str],
    root: Path,
    max_matches: int = 15,
) -> list[ToolMatch]:
    """Search file contents for keywords using ripgrep if available, or fallback."""
    if not keywords:
        return []
    root = Path(root).resolve()
    rg_bin = get_ripgrep_binary()

    valid_kw = [k.strip() for k in keywords if len(k.strip()) >= 2]
    if not valid_kw:
        return []

    # Try ripgrep executable first if available
    if rg_bin:
        try:
            # Two invocations, not one.  Identifiers are searched apart from
            # subtokens because a per-file ``--max-count`` budget is spent by
            # whichever keyword matches first, and a common subtoken matches
            # first everywhere: see ``_is_identifier_keyword`` for the measured
            # failure.  The identifier call matches almost nothing, so its lines
            # survive the budget and the gold file keeps the keyword that
            # actually distinguishes it.
            groups: list[list[str]] = []
            cmds: list[list[str]] = []
            id_kw = [k for k in valid_kw if _is_identifier_keyword(k)]
            rest_kw = [k for k in valid_kw if not _is_identifier_keyword(k)]
            if id_kw:
                groups.append(id_kw)
            if rest_kw:
                groups.append(rest_kw)

            best_per_file: dict[str, ToolMatch] = {}
            #: Every keyword that matched *anywhere* in the file, not just on the
            #: anchor line.  Ranking on the anchor's keywords alone measures which
            #: line happened to be kept, not what the file contains.
            file_kws: dict[str, set[str]] = {}
            position: dict[str, int] = {}
            # The repository's own declarations, applied to the *results* as well as
            # passed to ripgrep as globs.  ``iter_project_files`` reads these directly,
            # so without this the two discovery paths could disagree about what source
            # is -- which is how `work/` leaked back in when the search root was not a
            # git checkout.
            rules = load_ignore_rules(root)

            for group in groups:
                cmds.append(
                    [
                        rg_bin,
                        "-i",
                        "-n",
                        "--max-count", "3",
                        # Honor the repository's own `.gitignore` even when the search
                        # root is not a git checkout.  Without this, ripgrep applies
                        # gitignore rules only inside a repository, so the *same*
                        # directory produced different candidates depending on whether
                        # someone had run `git init` -- and ``iter_project_files``
                        # (which reads the rules directly via ``load_ignore_rules``)
                        # disagreed with this path about what source is.  Measured on
                        # ripgrep 14: with `.gitignore` holding `work/` and no
                        # repository, `work/a.py` was returned; with this flag it is not.
                        "--no-require-git",
                        *_ripgrep_exclude_globs(),
                        "|".join(_content_match_terms(group)),
                        str(root),
                    ]
                )

            # Run ripgrep in worker threads instead of through
            # ``create_subprocess_exec``.  MEASURED on this machine: spawning
            # ``rg.exe`` blocked the event loop for 441 ms while ripgrep itself ran in
            # 33 ms.  That block happens inside an explorer task, so nothing else can
            # be scheduled until it returns -- the "parallel" explorer was serial in
            # practice, and the exploration budget could not fire, because the timer
            # lives on the loop that was blocked.  Observed consequences on a 0.5 s
            # budget: a run reaching 1721 ms, and the same prompt yielding 4 different
            # candidate lists in 20 runs.  ``git_history_search`` already runs its
            # subprocess in a thread; ripgrep was the outlier.
            #
            # The two groups are awaited concurrently rather than in sequence.  The
            # cost of a call is dominated by process creation, not by ripgrep's own
            # work (441 ms to spawn vs 33 ms to search), so two sequential calls cost
            # twice the budget of one.  That is not a theoretical concern: the second
            # call pushed ``rust_taskrunner`` past its exploration budget and the run
            # truncated before reaching ``src/config.rs`` -- a correctness regression,
            # not merely a latency one.  Overlapped, both calls cost the budget of one.
            def _run_ripgrep(cmd: list[str]) -> str:
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=_RIPGREP_TIMEOUT_S,
                )
                return completed.stdout.decode("utf-8", errors="replace")

            outputs = await asyncio.gather(
                *(asyncio.to_thread(_run_ripgrep, c) for c in cmds)
            )

            # One anchor per file, chosen rather than taken.  ripgrep emits matches in
            # line order, and the caller wants a single line to anchor on, so keeping
            # whichever came first is a coin flip over which match was most useful.
            for lines in outputs:
                for line in lines.splitlines():
                    m = re.match(r"^((?:[a-zA-Z]:[\\/])?[^:]+):(\d+):(.*)$", line)
                    if m:
                        raw_path, line_no_str, snippet = m.group(1), m.group(2), m.group(3)
                    try:
                        file_path = os.path.relpath(raw_path, root).replace(os.sep, "/")
                    except ValueError:
                        file_path = raw_path.replace(os.sep, "/")
                    first_part = file_path.split("/")[0].lower()
                    if first_part in SKIP_DIRS:
                        continue
                    # Belt and braces: the glob list above is built from the shared
                    # tables, but a glob is a filter on the *search*, not a statement
                    # about the result.  Sylphae's own artifacts must never become
                    # candidates, so the decision is re-applied where the result enters.
                    if is_sylphae_artifact(file_path):
                        continue
                    if _ignored_by_repo_rules(rules, file_path):
                        continue
                    line_no = int(line_no_str)
                    snip_clean = snippet.strip()
                    snip_lower = snip_clean.lower()
                    hits = file_kws.setdefault(file_path, set())
                    for kw in valid_kw:
                        if kw.lower() in snip_lower:
                            hits.add(kw.lower())
                    candidate = ToolMatch(
                        path=file_path,
                        source_tool="ripgrep",
                        score=0.0,  # assigned by ``_rank_content_matches``
                        snippet=snip_clean,
                        line=line_no,
                    )
                    previous = best_per_file.get(file_path)
                    if previous is None:
                        best_per_file[file_path] = candidate
                        position[file_path] = len(position)
                        if len(position) >= _RIPGREP_MAX_FILES:
                            break
                    elif (
                        _placement_rank(candidate.snippet),
                        -(candidate.line or 0),
                    ) > (
                        _placement_rank(previous.snippet),
                        -(previous.line or 0),
                    ):
                        best_per_file[file_path] = candidate
            if best_per_file:
                return _rank_content_matches(best_per_file, file_kws, position, max_matches)
        except asyncio.CancelledError:
            # The subprocess runs in a worker thread, where a cancellation cannot reach
            # it.  It is bounded by ``_RIPGREP_TIMEOUT_S`` instead, and the thread is
            # left to finish on its own rather than being killed from the wrong thread.
            raise
        except Exception:
            pass  # Fall back to pure-Python search

    # Pure-Python ripgrep fallback
    return await _py_content_search(valid_kw, root, max_matches)


async def _py_content_search(
    keywords: Sequence[str],
    root: Path,
    max_matches: int,
) -> list[ToolMatch]:
    """Fast async pure-python keyword search across polyglot source files."""
    matches: list[ToolMatch] = []
    from sylphae.repo.polyglot import iter_project_files
    proj_files = iter_project_files(root)
    valid_kw = [k.strip() for k in keywords if len(k.strip()) >= 2]
    if not valid_kw:
        return []
    kw_lowers = [k.lower() for k in valid_kw]
    regexes = [re.compile(term, re.IGNORECASE) for term in _content_match_terms(valid_kw)]

    def _search_sync() -> list[ToolMatch]:
        res = []
        for rel in proj_files:
            full_path = root / rel
            try:
                st = full_path.stat()
                if st.st_size > 256 * 1024 or st.st_size == 0:
                    continue
                content = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            content_lower = content.lower()
            # Fast C-level Boyer-Moore substring pre-filter
            if not any(k in content_lower for k in kw_lowers):
                continue

            # Check distinct keyword hits in this file
            matched_kws = [rx for rx in regexes if rx.search(content)]
            if not matched_kws:
                continue

            lines = content.splitlines()
            best_line_no = None
            best_snippet = ""
            has_definition = False

            for ln, line in enumerate(lines, 1):
                if any(rx.search(line) for rx in matched_kws):
                    if not best_snippet:
                        best_line_no = ln
                        best_snippet = line.strip()
                    if re.search(r"^[A-Za-z0-9_]+\s*[:=]\s*[0-9A-Za-z_\"']+", line.strip()) or "const " in line:
                        has_definition = True
                    # Same anchor choice as the executable path, for the same reason:
                    # the first matching line is often a reference rather than the value.
                    if _placement_rank(line) > _placement_rank(best_snippet):
                        best_line_no = ln
                        best_snippet = line.strip()

            # Base score proportional to keyword coverage
            base_score = 0.70 + 0.15 * min(3, len(matched_kws) - 1)
            if has_definition:
                base_score += 0.25

            # Deprioritize test/benchmark/docs trees when searching
            rel_lower = rel.lower()
            if any(k in rel_lower for k in ("benchmarks/", "benchmark/", "tests/", "test/", "docs/", "examples/")):
                base_score *= 0.55

            res.append(ToolMatch(
                path=rel,
                source_tool="py_search",
                score=base_score,
                snippet=best_snippet,
                line=best_line_no,
            ))

            if len(res) >= max_matches * 2:
                break

        # Sort by score descending and return top matches
        res.sort(key=lambda m: -m.score)
        return res[:max_matches]

    return await asyncio.to_thread(_search_sync)


# ---------------------------------------------------------------------------
# 2. AST symbol search via RepoIndex
# ---------------------------------------------------------------------------

async def ast_search(
    query_tokens: Sequence[str],
    index_or_root: RepoIndex | Path,
    k: int = 10,
) -> list[ToolMatch]:
    """Search AST symbols, functions, classes and signatures."""
    if not query_tokens:
        return []

    def _search() -> list[ToolMatch]:
        if isinstance(index_or_root, RepoIndex):
            idx = index_or_root
        elif isinstance(index_or_root, Path) and index_or_root.is_file() and index_or_root.suffix == ".json":
            idx = RepoIndex.load(index_or_root)
        elif isinstance(index_or_root, Path):
            index_path = index_or_root / REPO_INDEX_FILE_NAME
            if index_path.exists():
                idx = RepoIndex.load(index_path)
            else:
                return []
        else:
            return []

        query = " ".join(query_tokens)
        cands = idx.candidates(query, k=k)
        # The index's own ranking signal has to survive the trip into [0, 1].  A fixed
        # ``score / 8`` collapses it: MEASURED on ``把 HTTP_500_INTERNAL_SERVER_ERROR ...``,
        # the defining file scored 48.34 raw and a test file that merely mentions 500 scored
        # 18.34, and both became exactly 1.000 -- the cap erased a 2.6x difference in
        # evidence, the two files then tied in the fusion, and the defining file lost the
        # tie-break.  A sigmoid ``raw/(raw+k)`` was tried and is worse: the peak file no
        # longer maps to 1.0, so ast's best (0.35*0.8) falls under ripgrep's best
        # (0.30*1.0) and the fuse stops dominating its own components.  Dividing by the
        # result set's peak is what makes the fuse domination-preserving: the peak file
        # always reaches the tool's full weight, and the remaining files keep their
        # proportion of it.
        #
        # The floor exists to stop a result set with no real signal from being inflated to
        # a full 1.0, and 1.0 is the right value for it rather than a guess: the smallest
        # weight the index can hand out is ``W_IMPORT = 1.0``, and a candidate only exists
        # at all because *something* matched, so any peak is already >= 1.0.  MEASURED on
        # the two-file reflex fixture (把导航栏的高度缩短一些): the peak was 6.27, and the
        # old floor of 8.0 held the definition at 0.784 -- term 0.2746 -- which lost to the
        # importing test file's ripgrep term of 0.30, and the agent edited the test instead
        # of the source it imports.
        peak = max((c.score for c in cands), default=0.0)
        scale = max(peak, 1.0)
        return [
            ToolMatch(
                path=c.path,
                source_tool="ast_index",
                score=min(1.0, max(0.0, c.score / scale)),
                snippet=c.snippet,
                line=c.snippet_span[0] if c.snippet_span else 1,
                symbols=c.symbols,
            )
            for c in cands
        ]

    return await asyncio.to_thread(_search)


# ---------------------------------------------------------------------------
# 3. Git history search
# ---------------------------------------------------------------------------

async def git_history_search(
    keywords: Sequence[str],
    root: Path,
    limit: int = 10,
) -> list[ToolMatch]:
    """Search git commit history for files recently modified with matching keywords."""
    if not keywords:
        return []
    root = Path(root).resolve()
    git_dir = root / ".git"
    if not git_dir.exists():
        return []

    def _git_log_sync() -> list[ToolMatch]:
        try:
            # Query recent 60 commits with commit message and changed files
            res = subprocess.run(
                ["git", "log", "-n", "60", "--name-only", "--pretty=format:commit:%s"],
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=2.0,
            )
            lines = res.stdout.splitlines()
            current_subject = ""
            matches: list[ToolMatch] = []
            seen: set[str] = set()

            kw_lower = [k.lower() for k in keywords]

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("commit:"):
                    current_subject = line[7:].strip().lower()
                else:
                    rel_path = line.replace("\\", "/")
                    if rel_path.endswith((".py", ".tsx", ".jsx", ".ts", ".js", ".css")):
                        # Score by match in commit message or file path
                        subj_hit = any(kw in current_subject for kw in kw_lower)
                        path_hit = any(kw in rel_path.lower() for kw in kw_lower)
                        if (subj_hit or path_hit) and rel_path not in seen:
                            seen.add(rel_path)
                            score = 0.75 if (subj_hit and path_hit) else 0.55
                            matches.append(ToolMatch(
                                path=rel_path,
                                source_tool="git_history",
                                score=score,
                                snippet=f"recent commit: {current_subject}",
                            ))
                            if len(matches) >= limit:
                                break
            return matches
        except Exception:
            return []

    return await asyncio.to_thread(_git_log_sync)


# ---------------------------------------------------------------------------
# 4. Path lexical search
# ---------------------------------------------------------------------------

async def path_lexical_search(
    keywords: Sequence[str],
    files: Sequence[str],
    k: int = 10,
) -> list[ToolMatch]:
    """Score file paths by token overlap."""
    return await asyncio.to_thread(_score_paths, keywords, files, k)


def _score_paths(
    keywords: Sequence[str],
    files: Sequence[str],
    k: int,
) -> list[ToolMatch]:
    """The synchronous core of :func:`path_lexical_search`.

    Kept module-level so the neural tool can build its pool without nesting an
    event loop inside a worker thread.
    """
    if not keywords or not files:
        return []

    kw_set = {k.lower() for k in keywords if len(k) >= 2}
    scored: list[tuple[float, str]] = []
    for f in files:
        path_tokens = set(tokenize_text(f))
        overlap = len(kw_set & path_tokens)
        if overlap > 0:
            # Prioritize direct name match
            stem = Path(f).stem.lower()
            stem_bonus = 1.0 if stem in kw_set else 0.0
            scored.append((overlap + stem_bonus, f))

    scored.sort(key=lambda item: -item[0])
    return [
        ToolMatch(
            path=path,
            source_tool="path_lexical",
            score=min(1.0, score * 0.4),
            snippet="",
        )
        for score, path in scored[:k]
    ]


#: Width of the pool the neural tool reranks.  Bounded because each batch of
#: ``n_candidates`` is one forward pass, and the exploration budget is sub-second.
_NEURAL_POOL = 32


async def neural_search(
    keywords: Sequence[str],
    files: Sequence[str],
    prompt: str,
    k: int = 16,
    run_dir: str | Path | None = None,
    index: Any = None,
) -> list[ToolMatch]:
    """Rerank a candidate pool with the trained target head.

    The lexical score is surface overlap between request and path, which is why a
    request naming a *concept* rather than an identifier falls through: the tokens
    shared with any English path are the empty set and every file ties at zero.
    The head was fit on (request, candidates) -> gold pairs, so it carries
    associations the lexical score cannot express.  It is a reranker, not a
    retriever -- it still needs a pool.

    The pool is the lexical layer's top paths, widened with a deterministic slice
    of the repository when the request shares no token with any path, so the head
    is given something to rank on the concept class at all.  Scores are softmaxed
    over the pool so they fuse with the other tools' [0, 1] terms.
    """
    from sylphae.agent.neural import NeuralRanker, NeuralUnavailable  # noqa: PLC0415 - lazy torch

    if not files or not prompt:
        return []

    def _rank() -> list[ToolMatch]:
        try:
            ranker = NeuralRanker.for_run(run_dir)
        except NeuralUnavailable:
            return []
        pool = [m.path for m in _score_paths(keywords, files, _NEURAL_POOL)]
        if len(pool) < _NEURAL_POOL:
            # No lexical signal -- the concept case.  Hand the head a broad,
            # deterministic slice instead of letting the pool come back empty.
            stride = max(1, len(files) // _NEURAL_POOL)
            for f in list(files)[::stride]:
                if f not in pool:
                    pool.append(f)
                if len(pool) >= _NEURAL_POOL:
                    break
        if not pool:
            return []
        # The head was trained on candidates carrying real file content, so score
        # the enriched form: bare paths cap it at the lexical ceiling (MEASURED
        # 0.248 hit@1) and the enrichment is what lifted it to 0.2556/0.5163.
        # Without an index the pool stays bare and the tool degrades honestly.
        entries: list[Any] = pool
        if index is not None:
            try:
                entries = index.enrich(pool, prompt)
            except Exception:
                entries = pool
        scored = ranker.score(prompt, entries)
        probs = _softmax([s.score for s in scored])
        ranked = sorted(zip(scored, probs), key=lambda pair: -pair[1])
        return [
            ToolMatch(path=s.path, source_tool="neural", score=float(p), snippet="")
            for (s, p) in ranked[:k]
        ]

    return await asyncio.to_thread(_rank)


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    hi = max(values)
    exps = [math.exp(v - hi) for v in values]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [e / total for e in exps]
