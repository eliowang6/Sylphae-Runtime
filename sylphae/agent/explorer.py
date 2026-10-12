"""Parallel exploration engine for Sylphae agent.

Implements concurrent tool execution, asynchronous result streaming,
early candidate locking, and mid-flight interruption.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Sequence

from sylphae.agent.effort import EffortConfig, LowEffort
from sylphae.agent.tools import (
    ToolMatch,
    _is_source_path,
    ast_search,
    git_history_search,
    neural_search,
    path_lexical_search,
    ripgrep_search,
)
from sylphae.agent.spatial import resolve_spatial_anchors
from sylphae.repo.index import (
    REPO_INDEX_FILE_NAME,
    RepoIndex,
    is_test_path,
    tokenize_text,
)
from sylphae.repo.polyglot import iter_project_files


@dataclass
class CandidateResult:
    path: str
    score: float
    contributing_tools: set[str] = field(default_factory=set)
    snippet: str = ""
    line: int | None = None
    symbols: list[str] = field(default_factory=list)
    #: True when this candidate came from the "no tool matched anything" fallback
    #: rather than from evidence about the request.  Such a candidate is a guess at
    #: where to *start looking*, not a finding: it carries no snippet, no line, and no
    #: relation to any keyword.  Downstream must not let it stand in for a located
    #: target -- see ``_fallback_confidence`` in the reflex agent.
    from_fallback: bool = False


# Mapping of common Chinese and English UI / coding concepts to candidate tokens
_CONCEPT_SYNONYMS: dict[str, list[str]] = {
    "导航栏": ["navbar", "nav", "navigation"],
    "导航": ["navbar", "nav", "navigation"],
    "页眉": ["header", "topbar"],
    "高度": ["height", "h-", "min_height", "max_height"],
    "宽度": ["width", "w-", "min_width", "max_width"],
    "尺寸": ["size", "height", "width", "dimension"],
    "颜色": ["color", "bg", "background", "foreground", "theme"],
    "背景": ["bg", "background", "bgcolor"],
    "内边距": ["padding", "pad", "p-"],
    "外边距": ["margin", "m-"],
    "间距": ["gap", "padding", "margin", "space"],
    "按钮": ["button", "btn", "Button", "btn-primary", "buy-btn"],
    "右上角": ["top-right", "header-actions", "navbar-end", "header", "nav"],
    "左上角": ["top-left", "navbar-start", "brand", "logo"],
    "溢出": ["overflow", "text-overflow", "ellipsis", "truncate", "white-space"],
    "控件": ["card", "component", "widget", "box", "container", "element"],
    "文字": ["text", "title", "label", "card", "content"],
    "超时": ["timeout", "read_timeout", "connect_timeout", "deadline"],
    "重试": ["retry", "retries", "max_retries", "backoff"],
    "端口": ["port", "defaultport", "listen"],
    "线程": ["worker", "workers", "thread", "threads"],
    "工作": ["worker", "workers", "task", "job"],
    "并发": ["worker", "workers", "concurrency"],
    "最大": ["max", "maximum", "limit"],
    "最小": ["min", "minimum"],
    "圆角": ["border-radius", "rounded", "radius"],
    "商品": ["product", "item"],
    "购物车": ["cart", "buy"],
    "购买": ["buy", "checkout", "cart"],
    "加入": ["add", "insert", "append"],
    "创建": ["create", "new", "add"],
    "项目": ["project", "item"],
    "搜索": ["search", "search-input", "filter", "query"],
    "指标": ["metric", "metrics", "stats", "dashboard"],
    "卡片": ["card", "panel", "box"],
    "操作": ["action", "actions", "btn", "button"],
    "标题": ["title", "heading"],
    "状态码": ["status", "status_code", "code", "http_status"],
    "配置": ["config", "settings", "options", "conf"],
    "缓存": ["cache", "caching", "ttl"],
    "样式": ["style", "styles", "css", "theme", "stylesheet"],
    "选项": ["option", "options", "choice", "choices"],
    "开关": ["switch", "toggle", "switches", "flag"],
    "用法": ["usage", "usage_text"],
    "全局": ["global", "globals", "root"],
    "用户": ["user", "account", "profile"],
    "权限": ["auth", "permission", "role", "token"],
    "注销": ["logout", "signout", "deactivate", "delete_account"],
    "登录": ["login", "signin", "auth", "authenticate"],
    "注册": ["register", "signup"],
    "认证": ["auth", "authenticate", "token", "jwt", "session"],
    "会话": ["session", "sessions", "cookie"],
    "版本": ["version", "__version__"],
}


#: A request that replaces one value with another.  ``从`` (or ``from``) marks the value
#: that exists in the file today; ``改为`` (or ``to``) marks its replacement.  Only the
#: existing value is a retrieval signal -- the replacement is not in the file yet.  See
#: ``extract_keywords``.
#:
#: Group 2 (the replacement) is *not* a retrieval signal, but it is still needed: its
#: words are the ones that leak into the AST symbol index as exact names -- see
#: ``_ast_query_keywords``.
_OLD_VALUE_RE = re.compile(
    r"(?:从|\bfrom\b)\s*(.+?)\s*"
    r"(?:改为|修改为|更改为|改成|变成|调整为|切换为|更新为|替换为|\bto\b)\s*(.+)",
    re.IGNORECASE,
)

#: Splits a value phrase into the word tokens the AST index would try to match as symbol
#: names.  Mirrors ``tokenize_text``'s alphabet runs: ``bold cyan`` -> {bold, cyan}.
_VALUE_TOKEN_RE = re.compile(r"[A-Za-z]+|[0-9]+")


def _value_tokens(prompt: str) -> set[str]:
    """Word tokens of the old and new values in a value-replacement request.

    ``把选项的样式从bold cyan改为bold red`` is about a string *literal*; ``cyan`` and
    ``red`` are its words.  ``typer/colors.py`` defines ``CYAN`` and ``RED`` as constants,
    so the AST index scores those words as ``W_SYM_EXACT * idf`` against ``colors.py``
    alone (df=1 -> maximal IDF), and ``ast``'s 0.35 tool weight is higher than
    ``ripgrep``'s 0.30 -- the fuse takes the strongest term, so the AST peak won no
    matter how well ripgrep scored the file that actually contains ``"bold cyan"``.
    MEASURED: ``colors.py`` was the AST peak at 46.1 for all four style prompts while
    ``rich_utils.py`` -- which holds the literal -- sat at 17-43.
    """
    out: set[str] = set()
    for m in _OLD_VALUE_RE.finditer(prompt):
        for grp in (m.group(1), m.group(2)):
            if not grp:
                continue
            old = grp.strip().strip("'\"")
            out.update(t.lower() for t in _VALUE_TOKEN_RE.findall(old))
    return out


def _ast_query_keywords(keywords: list[str], prompt: str) -> list[str]:
    """Drop value-literal words from the AST query, keeping every other signal.

    The AST index matches *symbol names*; the words of a replaced string literal are not
    names, and the one file that happens to define them as constants gets a maximal-IDF
    exact-name hit for free (``_value_tokens`` documents the measurement).  Dropping them
    here does not touch ripgrep, which still receives the whole phrase -- the phrase is
    how line 32 is located, and it is a signal the AST index cannot see either way.

    A keyword survives unless *all* of its word tokens come from a value: ``bold cyan``
    is pure value, ``style_option`` is pure name, and a mixed token would rather stay.
    CJK keywords (``样式``) have no word tokens under ``_VALUE_TOKEN_RE`` and are kept --
    ``all()`` over an empty set is true.
    """
    value_toks = _value_tokens(prompt)
    if not value_toks:
        return keywords
    kept: list[str] = []
    for kw in keywords:
        toks = [t.lower() for t in _VALUE_TOKEN_RE.findall(kw)]
        if toks and all(t in value_toks for t in toks):
            continue
        kept.append(kw)
    return kept


def extract_keywords(prompt: str) -> list[str]:
    """Extract candidate identifiers, keywords, and synonym expansions from prompt."""
    tokens = tokenize_text(prompt)
    kw_set: set[str] = set(tokens)

    #: A real symbol name, not a common word.  Length is the wrong test for this:
    #: ``green`` and ``yellow`` are five and six characters, and both are *values*, not
    #: names.  Structure is the tell -- an underscore, a digit, or a case transition that
    #: no plain word carries (``max_width``, ``LoadSettings_v2``, ``LoadConfig``).  A
    #: bare lowercase word has none: MEASURED on ``把开关的样式从bold green改为bold
    #: blue``, where ``green`` alone tripped this gate, closed the synonym branch, and
    #: left a request about 样式/开关 with three keywords -- ``bold``, ``green``, ``blue``
    #: -- none of them the concept it was about.  ``config`` no longer trips it either;
    #: the length rule it was written under already called that "a word as much as a
    #: symbol", so this is the rule that comment intended.
    _IDENTIFIER_RE = re.compile(
        r"[A-Za-z][A-Za-z0-9_]*"
        r"(?:[_\d][A-Za-z0-9_]*|[a-z][A-Z][A-Za-z0-9_]*|[A-Z][a-z][A-Za-z0-9_]*)"
    )
    has_identifier = bool(_IDENTIFIER_RE.search(prompt))

    # Add spatial & component anchors -- but only for a request that is not naming a
    # symbol.  ``resolve_spatial_anchors`` matches its aliases as *substrings of the raw
    # prompt*, so the words inside an identifier trigger it: MEASURED on
    # ``把 HTTP_431_REQUEST_HEADER_FIELDS_TOO_LARGE 修改为 432``, where ``header`` fired
    # the 顶部 position anchor and ``field`` fired the 输入框 component anchor and pulled
    # navbar/topbar/form-control/textarea/textbox into a request about an HTTP status
    # constant.  Those synonyms then outranked the constant's own definition.  The same
    # gate the concept synonyms use applies here for the same reason.
    if not has_identifier:
        spatial = resolve_spatial_anchors(prompt)
        for sk in spatial.search_keywords:
            kw_set.add(sk)

    # Concept synonyms are for *concept-only* requests.  When the prompt already names an
    # identifier the request is explicit (the ladder's class A/B), and expanding
    # "unauthorized" into authenticate/session/signin/token/login/jwt/... turns one exact
    # signal into eighteen noisy ones.  MEASURED: "把 HTTP_401_UNAUTHORIZED 修改为 402"
    # expanded to 18 keywords, 14 of them auth/session synonyms; ``ast`` alone ranked the
    # defining file first and the fused explorer ranked it outside the top six, because
    # every auth-adjacent file in the repository accumulated matches on the synonyms.
    if not has_identifier:
        # Check for Chinese and English synonym triggers
        lower_prompt = prompt.lower()
        for concept, synonyms in _CONCEPT_SYNONYMS.items():
            if concept in prompt or any(s in lower_prompt for s in synonyms):
                kw_set.add(concept)
                for s in synonyms:
                    kw_set.add(s)

    # Check for identifiers with underscores, dots, hyphens
    for match in re.finditer(r"[A-Za-z0-9_.-]{3,}", prompt):
        kw_set.add(match.group(0).lower())

    # A value-replacement request names the value it is replacing.  ``从 X 改为 Y``: X is
    # in the file *today*, Y is not in it yet, so X is the retrieval signal and Y is
    # noise.  MEASURED on ``把选项的样式从bold cyan改为bold red``, where the replacement
    # ``red`` matched ``typer/colors.py`` (RED = "red") while the file that had to be
    # found never writes a bare ``red`` -- and ``colors.py`` outranked it.  The phrase is
    # kept whole rather than tokenised: ``bold cyan`` occurs in exactly one file
    # (``typer/rich_utils.py:32``), but ``bold`` and ``cyan`` separately are the
    # vocabulary of every file that names a colour, so tokenising gave ``colors.py`` a
    # higher idf-weighted score than the file containing the string verbatim.  Unconditional:
    # a request can name a symbol *and* name the old value
    # (``把 STYLE_OPTION 的值从 bold cyan 修改为 bold red``), and the phrase is not a
    # synonym the identifier gate is there to suppress.
    for m in _OLD_VALUE_RE.finditer(prompt):
        old = m.group(1).strip().strip("'\"")
        if len(old) >= 2:
            kw_set.add(old.lower())
    return sorted(kw_set, key=lambda x: -len(x))


def _named_tokens(prompt: str) -> set[str]:
    """Tokens the user actually wrote, not the ones synonym expansion contributed.

    ``RepoIndex.candidates`` builds an explained-set that suppresses a token's
    subtoken-scoring everywhere once the token is an exact symbol name somewhere.  That is
    right when the user *named* the symbol and wrong when a concept synonym merely collided
    with one: ``option``/``style``/``css`` are exact names in typer, and for
    ``把选项的样式...`` they were all suppressed, costing the gold file three of its four
    scoring tokens.  The set below is every way a user-written identifier can appear --
    bare (``bold``), scored (``max_width``), dotted (``typer.rich_utils``).
    """
    return {t.lower() for t in re.findall(r"[A-Za-z0-9_.-]{2,}", prompt)}



#:
#: This is a *tie-breaker*, not a competitor: the bonus must stay small enough that a file
#: whose best term is clearly stronger still wins.  At 0.5 it did not -- two tools scoring
#: 0.9 each fused to 0.45 and beat an exact match worth 0.35, so the fuse was still
#: domination-breaking.  MEASURED against the localisation ladder; see
#: ``tests/test_explorer_fusion.py``.
CORROBORATION_DISCOUNT = 0.1


def fuse_tool_terms(
    terms: dict[str, float],
    *,
    discount: float = CORROBORATION_DISCOUNT,
) -> float:
    """Fuse a file's per-tool weighted scores into one ranking score.

    The fuse must not let several weak agreements outrank one exact match.  Under a plain
    weighted sum, three tools each scoring 0.4 on a near-miss contribute 0.30 together
    while a single tool scoring 1.0 on the right file contributes only its own weight -- so
    near-misses accumulate and the right file falls out of the top-k, and the fused explorer
    scores *worse than its own best component*.  The strongest term carries the score; the
    rest is added at ``discount``, so corroboration still helps but cannot drown a strong
    single hit.
    """
    if not terms:
        return 0.0
    vals = list(terms.values())
    best = max(vals)
    return best + (sum(vals) - best) * discount


def _candidate_sort_key(candidate: "CandidateResult") -> tuple[int, int, float, str]:
    """Order fused candidates: source before non-source, code before test, then score.

    The per-tool rankers already carry this structure -- ``_rank_content_matches`` sorts
    on ``(0 if source else 1, 1 if is_test else 0, -score)`` and documents why both keys
    exist. The cross-tool fusion threw all of it away and sorted on the summed score
    alone, so a test file that many tools each matched weakly could outrank the source
    file one tool matched exactly.

    MEASURED on ``runs/iter34``'s verified-Python reflex set: for
    ``把默认的溢出处理方式从fold改为crop`` the fused order was
    ``[tests/test_text.py, rich/text.py, ...]`` -- ripgrep on its own had ranked
    ``rich/text.py`` second of fifteen and the fusion promoted the test file to first,
    and the agent then edited the test. Editing the test is the wrong answer to almost
    any "change this constant" request, so the key that says so has to survive fusion.

    Score stays the third key, so this only decides cases the score alone gets wrong in
    the direction of editing a test or a config file, and it cannot promote a
    non-source file over a source one no matter how well the non-source file matches.
    """
    return (
        0 if _is_source_path(candidate.path) else 1,
        1 if is_test_path(candidate.path) else 0,
        -candidate.score,
        candidate.path,
    )


class ParallelExplorer:
    """Dispatches parallel tool invocations and streams candidate findings asynchronously."""

    def __init__(
        self,
        root: Path,
        repo_index: RepoIndex | None = None,
        effort: EffortConfig = LowEffort,
    ):
        self.root = Path(root).resolve()
        self.repo_index = repo_index
        self.effort = effort

        #: True when the exploration budget expired while tools were still running.
        #:
        #: The candidate list is then a *prefix* of an answer, not an answer.  This
        #: matters because the no-match fallback below cannot tell the difference: an
        #: empty ``cand_map`` means either "every tool searched and found nothing" or
        #: "no tool had returned yet when we stopped waiting", and it reports the
        #: first in both cases.  MEASURED on ``work/repos/tiangolo__typer`` (812 files)
        #: with ``LowEffort.timeout_seconds = 0.5``: 20 runs of the same prompt
        #: produced 4 different candidate lists, and 11 of the 20 had no tool answer
        #: at all -- latency ranged 436-1721 ms, straddling the budget.
        self.last_exploration_truncated: bool = False
        #: Names of the tools that had not returned when the budget expired.
        self.last_pending_tools: list[str] = []

    async def explore(
        self,
        prompt: str,
        additional_keywords: Sequence[str] | None = None,
        early_lock_threshold: float = 0.90,
        on_candidate_update: Callable[[list[CandidateResult]], None] | None = None,
    ) -> list[CandidateResult]:
        """Concurrently launch tools, stream results, and aggregate candidate rankings."""
        results: list[CandidateResult] = []
        async for updated_candidates in self.stream_explore(
            prompt=prompt,
            additional_keywords=additional_keywords,
            early_lock_threshold=early_lock_threshold,
        ):
            results = updated_candidates
            if on_candidate_update:
                on_candidate_update(results)
        return results

    async def stream_explore(
        self,
        prompt: str,
        additional_keywords: Sequence[str] | None = None,
        early_lock_threshold: float = 0.90,
    ) -> AsyncIterator[list[CandidateResult]]:
        """Asynchronously yield updated candidate rankings as each tool returns."""
        # Reset per call: a previous request's truncation must not be read as this
        # request's outcome.
        self.last_exploration_truncated = False
        self.last_pending_tools = []

        keywords = extract_keywords(prompt)
        if additional_keywords:
            keywords.extend(additional_keywords)
            keywords = sorted(set(keywords), key=lambda x: -len(x))

        tasks: list[asyncio.Task[list[ToolMatch]]] = []

        # 1. ripgrep content search
        if self.effort.include_ripgrep:
            tasks.append(asyncio.create_task(
                ripgrep_search(keywords, self.root, max_matches=15),
                name="ripgrep"
            ))

        # 2. AST symbol search (only when index is available to prevent blocking full-repo build)
        has_index = self.repo_index is not None or (self.root / REPO_INDEX_FILE_NAME).exists()
        resolved_index: Any = None
        if has_index:
            resolved_index = self.repo_index
            if resolved_index is None:
                from sylphae.repo.index import RepoIndex  # noqa: PLC0415

                resolved_index = RepoIndex.load(self.root / REPO_INDEX_FILE_NAME)
            idx = resolved_index
        if self.effort.include_ast and has_index:
            # Value-literal words are stripped from the AST query only: they are string
            # values, not symbol names, and the one file that defines them as constants
            # wins the AST peak with maximal-IDF exact-name hits.  ripgrep still receives
            # the full list -- the phrase ``bold cyan`` is how the literal's line is found.
            ast_keywords = _ast_query_keywords(keywords, prompt)
            tasks.append(asyncio.create_task(
                ast_search(
                    ast_keywords, idx,
                    k=self.effort.n_candidates * 2,
                    # Test files are demoted in the fusion sort key and excluded from the
                    # AST peak: a test at the peak divides every source file's normalized
                    # score, so gold arrives ordered-right but weighted-weak. MEASURED on
                    # typer style_option, where tests/test_types.py was the AST peak.
                    exclude_tests=True,
                    # Only the identifiers the user wrote may seed the explained-set; the
                    # concept synonyms in ``keywords`` are not symbols the request named,
                    # and letting them suppress ``option``/``style`` cost the gold file
                    # three of its four scoring tokens.
                    named=_named_tokens(prompt),
                ),
                name="ast"
            ))

        # 3. Path lexical search
        proj_files = iter_project_files(self.root)
        tasks.append(asyncio.create_task(
            path_lexical_search(keywords, proj_files, k=self.effort.n_candidates * 2),
            name="path_lexical"
        ))

        # 4. Git history search
        if self.effort.include_git_history:
            tasks.append(asyncio.create_task(
                git_history_search(keywords, self.root, limit=10),
                name="git_history"
            ))

        # 5. Neural rerank (the trained target head over a lexical pool).  Opt-in:
        # one forward pass per candidate batch, and it needs a trained checkpoint
        # to exist.  Missing, it returns no matches and the lexical tools stand.
        # The index is passed so the head scores candidates carrying real file
        # content -- it was trained on enriched candidates, and bare paths cap it
        # at the lexical ceiling.
        if self.effort.include_neural:
            tasks.append(asyncio.create_task(
                neural_search(keywords, proj_files, prompt,
                              k=self.effort.n_candidates * 2,
                              index=resolved_index),
                name="neural"
            ))

        # Aggregate candidate map: path -> CandidateResult.  ``terms`` keeps each tool's
        # weighted contribution per path, because the fused score is recomputed from them
        # after every tool returns (the ranking is streamed, not built once at the end).
        cand_map: dict[str, CandidateResult] = {}
        terms: dict[str, dict[str, float]] = {}
        locked = False

        # Tool weights
        tool_weights = {
            "ast_index": 0.35,
            "ripgrep": 0.30,
            "py_search": 0.30,
            "path_lexical": 0.25,
            "git_history": 0.20,
            # The trained head carries request->path associations the lexical scores
            # cannot express, so it is weighted with the strongest lexical tool.
            "neural": 0.35,
        }
        #: How much a second piece of evidence is worth once the strongest one is counted.
        #:
        #: The fusion used to be a plain weighted sum, which means N weak agreements outrank
        #: one exact match: a noise file seen by three tools at 0.4 each reached
        #: 0.4*(0.30+0.25+0.20)=0.30 while the gold file's single ast hit at 1.0 contributed
        #: only 1.0*0.35=0.35 -- so near-misses accumulated and the gold fell out of the
        #: top-k.  MEASURED: ``ast`` alone scores 86/86 on class A and 152/160 on class B,
        #: the fused explorer 81/86 and 137/160; the fuse is *worse than its own component*.
        #: The strongest evidence now carries the score, and corroboration is added at a
        #: discount so agreement still helps but cannot drown a strong single hit.
        #: ``fuse_tool_terms`` is module-level so the property is tested directly.

        def _fuse(path: str) -> float:
            return fuse_tool_terms(terms[path])

        try:
            for fut in asyncio.as_completed(tasks, timeout=self.effort.timeout_seconds):
                try:
                    matches = await fut
                except asyncio.CancelledError:
                    continue
                except asyncio.TimeoutError:
                    # ``as_completed`` raises the budget expiry from the *awaited result*
                    # as well as from the iteration, and on Python 3.13 the two are the
                    # same ``TimeoutError``.  Letting the broad handler below swallow it
                    # is how a truncated exploration came to look like a finished one:
                    # MEASURED on ``work/repos/tiangolo__typer``, 3 of 8 runs silently
                    # dropped ripgrep's matches while still reporting
                    # ``last_exploration_truncated = False``.
                    raise
                except Exception:
                    continue

                for m in matches:
                    weight = tool_weights.get(m.source_tool, 0.2)
                    if m.path not in cand_map:
                        cand_map[m.path] = CandidateResult(
                            path=m.path,
                            score=m.score * weight,
                            contributing_tools={m.source_tool},
                            snippet=m.snippet,
                            line=m.line,
                            symbols=list(m.symbols),
                        )
                        terms[m.path] = {m.source_tool: m.score * weight}
                    else:
                        existing = cand_map[m.path]
                        # keep the strongest per-tool term, so a tool that returns twice
                        # (streamed updates) does not double-count itself
                        terms[m.path][m.source_tool] = max(
                            terms[m.path].get(m.source_tool, 0.0), m.score * weight
                        )
                        existing.score = _fuse(m.path)
                        existing.contributing_tools.add(m.source_tool)
                        if m.source_tool in ("ripgrep", "py_search") and m.snippet:
                            existing.snippet = m.snippet
                            existing.line = m.line
                        elif not existing.snippet and m.snippet:
                            existing.snippet = m.snippet
                            existing.line = m.line
                        if m.symbols:
                            existing.symbols = list(set(existing.symbols + m.symbols))

                # Rank candidates
                ranked = sorted(cand_map.values(), key=_candidate_sort_key)[:self.effort.n_candidates]
                yield ranked

                # Early locking check
                if ranked and ranked[0].score >= early_lock_threshold and len(ranked[0].contributing_tools) >= 2:
                    locked = True
                    break

        except asyncio.TimeoutError:
            # The budget expired, not the search.  Record which tools were still
            # running, because everything below -- including the no-match fallback --
            # is about to reason as if the search had finished.
            self.last_exploration_truncated = True
            self.last_pending_tools = sorted(
                t.get_name() for t in tasks if not t.done()
            )
        finally:
            # Cancel any remaining pending tasks
            for t in tasks:
                if not t.done():
                    t.cancel()

        # Fallback if specific keywords did not yield candidates.  Nothing matched the
        # request -- or the budget expired before anything could answer, which
        # ``last_exploration_truncated`` distinguishes and this block cannot.  The best
        # that can be offered is somewhere to start looking.  The
        # score is a placeholder, not a confidence, and ``from_fallback`` is what says
        # so -- "no tool found anything" is a different state from "these are the best
        # three matches", and downstream cannot tell them apart from a number alone.
        if not cand_map and proj_files:
            ui_candidates = [f for f in proj_files if any(f.endswith(ext) for ext in (".css", ".scss", ".tsx", ".jsx", ".html", ".vue"))]
            fallback_list = ui_candidates if ui_candidates else proj_files
            for f in fallback_list[:self.effort.n_candidates]:
                cand_map[f] = CandidateResult(
                    path=f,
                    score=0.40,
                    contributing_tools={"fallback_project"},
                    from_fallback=True,
                )

        # Final yield
        ranked = sorted(cand_map.values(), key=_candidate_sort_key)[:self.effort.n_candidates]
        yield ranked
