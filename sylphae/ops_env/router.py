"""Route an ops symptom to a playbook by deciding what kind of problem it is.

ROUND 9, and the thing 2a asks for. MEASURED: ``ops_env_gate`` builds every symptom as
``Symptom(kind=task.category, ...)`` and calls ``select(task.category)``, which is a dict
lookup. So the eleven preset tasks never decide anything -- the ``category`` is written into
the task definition, the playbook is found by the same string, and the model is not
consulted at any point. The gate measures whether eight hand-assigned playbooks work, which
is a real result but is not a decision path.

**What this module adds is the missing decision, and only the missing decision.** The
playbooks, the safety checks and the trash-based undo stay exactly as they are; this decides
which one to call, refuses when the symptom does not clearly name a kind, and says why.

**Why refusal is the default and not a fallback.** ``select`` returns ``None`` for an
unknown kind and the docstring says the caller's job is then to escalate. That contract
already exists for a caller who knows what they do not know; what was missing was anything
that could *produce* the not-knowing. Routing a "port 8080 is occupied" symptom to
``missing-dependency`` would run a real command and return a real answer, and it would be
wrong -- which is why the decision below reports a confidence and a runner-up rather than
committing to the top match. A wrong playbook here is not a failed test, it is a command
executed against a machine on the strength of a guess.

**Why the decision is made from the symptom rather than from the task definition.** The
tasks' ``category`` is the label; the symptom's ``text`` is what an operator would actually
have. Keeping the label as the answer and the text as the input is what makes the gate able
to detect a mislabel -- and that detection is :func:`route`'s ``agrees_with_label`` field,
which is how this becomes a measurement rather than a feature.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "OpsRoute",
    "route",
    "KIND_VOCABULARY",
    "signals_for_kind",
]


@dataclass(frozen=True)
class _Signal:
    """A pattern that indicates a kind, and how much it is worth.

    ``weight`` exists because a symptom can carry several signals and the kinds are not
    mutually exclusive in the text: "no module named rich" and "cannot import rich" both
    point at ``missing-dependency``, while a traceback that says so *and* mentions a port
    does not. A single regex match therefore cannot decide, and neither can "first match
    wins" -- which is what an earlier version of this did and why it needed a confidence
    field to explain itself.
    """

    kind: str
    pattern: re.Pattern[str]
    weight: int = 1
    #: Human-readable form, for the report. ``None`` means "this matched a pattern, and the
    #: pattern is the explanation", which is the common case.
    note: str | None = None
    #: ROUND 46. Set when this signal identifies a specific env kind within its playbook.
    target_kind: str | None = None


def _c(kind: str, pattern: str, weight: int = 1, note: str | None = None,
       target_kind: str | None = None) -> _Signal:
    return _Signal(kind=kind, pattern=re.compile(pattern, re.I), weight=weight, note=note,
                   target_kind=target_kind)


#: Signals, in the order they are documented. Order does not decide anything -- the weights
#: and the margin do -- but it decides the tie-break, and a stable tie-break is what makes
#: the result reproducible across runs.
#: ROUND 46. A service *restart* request. The playbook is the same one a busy port routes to,
#: but the kind is the destructive ``service_restart`` rather than the advisory
#: ``port_release`` -- which is the distinction 12.79 found the map was making by fiat.
#:
#: Scored under the existing playbook so no existing route moves, and deliberately narrow: it
#: requires a restart *and* a service word, and must not match any of the eleven preset symptoms.
#: That is asserted in ``tests/test_ops_reachability.py`` rather than assumed.
KIND_SIGNALS: tuple[_Signal, ...] = (
    # ASCII and CJK are separate branches on purpose. ROUND 48: the first version wrote ``\\b`` after
    # the Chinese alternatives, and ``\\b`` between a CJK character and a digit is not a boundary --
    # so "重启 8000 端口的服务" did not match while "restart the api service" did. **A signal verified
    # in one language only looks verified.**
    _c("port-occupied",
       r"(restart|reboot)\b.*\b(service|worker|server)\b"
       r"|\b(service|worker|server)\b.*\b(restart|reboot)\b"
       r"|(重启|重新启动).*(服务|进程|worker|server)"
       r"|(服务|进程|worker|server).*(重启|重新启动)",
       1, "asks for a service to be restarted, not for a port to be released",
       target_kind="service_restart"),

    #: ROUND 47. The second destructive kind. Without this, "delete the virtualenv" routed to the
    #: ``venv`` playbook and resolved to ``venv_create`` -- so the request to remove an environment
    #: **created** one, and the bridge reported ``executed`` / ``reversible`` with no escalation.
    #: Naming the kind turns that into a refusal, which is what 2c asks for.
    #:
    #: Removal word **and** venv word, so "create a venv" and "the venv already exists" cannot match.
    # "nuke" and "purge" were missing until ROUND 48's sweep measured the miss -- and a removal word
    # list is exactly the kind of closed vocabulary that fails one word at a time.
    #: ROUND 48. The third and last shared-playbook site, and the sweep is what asked for it:
    #: "stop the service on port 8000" resolved to ``port_release``. Both are advisory and both get
    #: refused, so the harm was small -- but it was wrong, and it was wrong for a reason already
    #: solved twice. Same ASCII/CJK branch split as the restart signal, for the same reason.
    _c("port-occupied",
       r"\b(stop|kill|terminate|halt|shutdown|shut\s*down)\b.*\b(service|worker|server|process)\b"
       r"|\b(service|worker|server|process)\b.*\b(stop|kill|terminate|halt)\b"
       r"|(停止|停掉|关掉|关闭|终止).*(服务|进程|worker|server)"
       r"|(服务|进程|worker|server).*(停止|停掉|关掉|关闭)",
       1, "asks for a service to be stopped, not for a port to be released",
       target_kind="service_stop"),

    _c("venv",
       r"\b(delete|remove|destroy|drop|wipe|nuke|purge|rm|trash|discard)\b.*\b(venv|virtual\s*env)\b"
       r"|\b(venv|virtual\s*env)\b.*\b(delete|remove|destroy|drop|wipe|nuke|purge|rm|trash|discard)\b"
       r"|(删除|删掉|清理|销毁|清空).*(虚拟环境|venv)"
       r"|(虚拟环境|venv).*(删除|删掉|清理|销毁|清空)",
       2, "asks for a virtualenv to be removed, not created",
       target_kind="venv_remove"),
)

SIGNALS: tuple[_Signal, ...] = (
    # -- venv ---------------------------------------------------------------------
    _c("venv", r"virtual\s*env|\bvenv\b|虚拟环境", 2,
       "mentions a virtual environment"),
    _c("venv", r"create\s+(?:a\s+)?(?:new\s+)?(?:env|environment)|"
              r"(?:没有|缺少|未|帮我).{0,4}(?:虚拟环境|venv)", 2,
       "asks for an environment to be created"),
    # Reuse is a distinct ask from create, and the playbook's whole job is not rebuilding
    # one. Worth more than the generic venv signal so "reuse the existing venv" does not
    # tie with a bare mention.
    _c("venv", r"reuse|already\s+exists|re-?use|复用|已存在", 3,
       "asks to reuse rather than create"),

    # -- pip conflict --------------------------------------------------------------
    _c("pip-conflict", r"conflict|incompatible|version\s*\(?local\)?|"
                       r"依赖冲突|版本冲突|互斥", 2, "names a dependency conflict"),
    # MEASURED 2026-10-02: "requirements 里的两个包版本互斥，装不上" was refused because the
    # only Chinese signal was 依赖冲突/版本冲突 and this sentence says 互斥 instead. The
    # overfit run put it at 11/16, and this is one of the five. A paraphrase has to route the
    # same as the fixture or the router is a lookup table for the fixtures' own wording.
    _c("pip-conflict", r"装不上|装不了|install\s+fail|resolve|"
                       r"requirements", 2, "reports an install that cannot resolve"),
    # -- port occupied -------------------------------------------------------------
    _c("port-occupied", r"port\s+\d+|address\s+already\s+in\s+use|"
                        r"only\s+one\s+usage|EADDRINUSE|端口.{0,4}(?:占用|被占)", 3,
       "names a port in use"),
    # "8080 端口被别的进程占用了" matches neither the english form nor the previous chinese
    # one, because there is no 端口 word adjacent to 占用 -- the number leads. This one does.
    _c("port-occupied", r"\d{2,5}\s*(?:端口|port)", 3,
       "names a port number"),
    _c("port-occupied", r"who\s+(?:is|has)|哪个进程|占用进程|listening\s+on|"
                        r"哪个(?:程序|服务)|被谁占", 2,
       "asks which process holds something"),

    # -- missing dependency --------------------------------------------------------
    #
    # ROUND 17. The `cannot\s+import|ImportError` alternatives that used to sit here are gone,
    # and their removal is the fix. 12.51 measured the router at 13/20 on realistic rewrites,
    # and the one failure that was not a safe refusal was a misroute with consequences:
    #
    #     ImportError: cannot import name 'render' from 'app.views'
    #       -> missing-dependency, instead of import-error-locator
    #
    # A refusal wastes a turn. This sends the *wrong playbook* at the problem, and
    # missing-dependency is the one whose job is to consider installing a package. So the router
    # could hand an install-shaped tool a source-code problem.
    #
    # The reason the table could not tell them apart is that the two errors are not the same
    # error, and the old pattern treated them as one:
    #
    #   `ModuleNotFoundError: No module named X` -- the MODULE is absent. A package may be
    #     missing, and installing is a reasonable thing to consider. That is
    #     missing-dependency.
    #   `ImportError: cannot import name X from Y` -- the MODULE LOADED. The name is not in it.
    #     This is a source problem: the import statement, or the module's contents, are wrong.
    #     Installing anything cannot help. That is import-error-locator.
    #
    # So this entry now claims only the "module is absent" shapes, and the locator below claims
    # the "module loaded, name missing" shape. That is a missing distinction being supplied, not
    # a weight being moved: the weights are untouched, and the two patterns are disjoint by
    # construction rather than by preferring one.
    _c("missing-dependency", r"no\s+module\s+named|ModuleNotFoundError|"
                              r"缺少(?:模块|依赖)|未安装|找不到模块", 3,
       "is a missing module, which is the shape where a package may be absent"),
    # "import 的时候提示找不到模块 httpx" carries 找不到模块, which the pattern above
    # already covers -- but only because 找不到模块 was added for it. Without it this
    # sentence scored zero and was refused. Recorded because it is the second of five
    # rewrites that failed, and both were Chinese phrasings the English-first table missed.
    _c("missing-dependency", r"backport|version\s+gap|too\s+old|"
                              r"stdlib|版本(?:太老|过低)", 3,
       "is a stdlib version gap rather than a missing package"),
    _c("missing-dependency", r"import.{0,12}(?:失败|报错|提示|找不到)", 3,
       "reports an import that failed"),

    # -- affected tests -------------------------------------------------------------
    _c("affected-tests", r"which\s+tests?|what\s+tests?|"
                          r"affected\s+test|run\s+(?:the\s+)?tests|"
                          r"哪些测试|受影响.{0,2}测试|会受影响", 3, "asks which tests to run"),
    _c("affected-tests", r"imports?\s+the\s+changed|import\s+graph|"
                          r"改了.{0,20}(?:哪些|什么).{0,4}测试", 2,
       "asks about the import graph"),

    # -- flake retry ------------------------------------------------------------------
    # ``persistent-failure`` and ``flake-retry`` share a kind and differ in outcome, so the
    # signals deliberately do not separate them: the playbook checks whether the retry
    # passes. What the router must get right is that *both* are test failures, not venv or
    # dependency problems.
    _c("flake-retry", r"flaky|flake|intermittent|transient|"
                      r"passes?\s+(?:on\s+)?retry|时好时坏|偶发|不稳定", 3, "reports a flake"),
    _c("flake-retry", r"test\s+(?:failed|failure|failed\b)|failed\s+test|"
                      r"测试失败|persistent|一直失败|持续失败", 2, "reports a test failure"),

    # -- lint ------------------------------------------------------------------------
    _c("lint-fix", r"lint|flake8|ruff|black|format|"
                   r"格式(?:化|检查)|代码风格|按\s*pep8|规范化", 2,
       "asks about lint or formatting"),

    # -- import locator ----------------------------------------------------------------
    #
    # ROUND 17: the shape signal this entry lacked. The locator previously fired only when
    # someone *asked* to locate something -- "which file", "my code", "user's frame" -- and had
    # no signal for the error itself. So `ImportError: cannot import name 'render' from
    # 'app.views'`, which nobody asked about, scored zero here and fell through to
    # missing-dependency by default. That is how the one consequential 12.51 failure happened.
    #
    # The shapes claimed are the "module loaded, name missing" family, and they are disjoint from
    # the "module absent" family claimed above by construction:
    #
    #   cannot import NAME from MODULE   -- `from x import y` failed on the name
    #   ImportError: ... from 'x.y'      -- the same, once the traceback is collapsed
    #
    # `no module named` and `ModuleNotFoundError` appear in neither, so a genuinely absent module
    # cannot reach here. That is the property worth having, and a test asserts it rather than
    # trusting the two regexes to stay apart.
    _c("import-error-locator", r"cannot\s+import\s+name\b|"
                                r"ImportError[^\n]{0,80}\bfrom\s+['\"]|"
                                r"无法从.{0,20}导入|找不到名称", 3,
       "is a name that is missing from a module that loaded, which installing cannot fix"),
    _c("import-error-locator", r"locate|which\s+file|which\s+module\s+in\s+your|"
                               r"my\s+code|user(?:'s)?\s+(?:code|frame)|"
                               r"定位|哪个文件", 2,
       "asks to locate code in a traceback"),
    # "这个 ImportError 是我自己的代码引起的吗？" -- the question is whether the failure is
    # the caller's own code, which is exactly the locator's question and nothing to do with
    # naming a missing module. The english-first table had no signal for it.
    _c("import-error-locator", r"(?:我|自己|用户)(?:的)?(?:代码|项目)|"
                               r"是我.{0,6}(?:的|吗)|自己的代码", 3,
       "asks whether the failure is in the caller's own code"),
)

#: The kinds ``playbooks.select`` knows, copied rather than imported so this module has no
#: dependency on the executor and can be reasoned about -- and tested -- on its own. A test
#: asserts the two lists agree, because a vocabulary that drifts from the registry would
#: route to a kind that does not exist.
KIND_VOCABULARY: tuple[str, ...] = (
    "venv", "pip-conflict", "port-occupied", "missing-dependency",
    "affected-tests", "flake-retry", "lint-fix", "import-error-locator",
)

#: Below this margin between the top two kinds, the decision is a guess and is reported as
#: one. 0 means "any difference is enough", which is the same as not having the field.
DEFAULT_MARGIN = 1


@dataclass
class OpsRoute:
    """The routing decision, with enough of the reasoning attached to audit it."""

    kind: str | None = None
    #: ROUND 46. The specific env kind, when the symptom names one that the playbook serves.
    #: ``kind`` above is a *playbook*; several kinds share one playbook, which is why three of the
    #: eleven were unreachable until this field existed. ``None`` means "no kind named", and
    #: callers keep their playbook-to-kind fallback -- so every route that does not set this
    #: behaves exactly as before.
    target_kind: str | None = None
    confidence: float = 0.0
    #: The runner-up, when there is one. Present so a wrong route is diagnosable.
    runner_up: str | None = None
    margin: float = 0.0
    #: Every signal that fired, as ``(kind, weight, note)``, for the report.
    fired: list[tuple[str, int, str | None]] = field(default_factory=list)
    #: Set when the symptom does not clearly name a kind. The playbook layer's own contract
    #: is that an unknown kind escalates, so this is a refusal, not a failure.
    refused: str | None = None
    #: The task's declared category, when the caller had one, and whether the decision
    #: matched. This is what turns the router into a measurement.
    agrees_with_label: bool | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "confidence": round(self.confidence, 3),
            "runner_up": self.runner_up, "margin": round(self.margin, 3),
            "fired": [{"kind": k, "weight": w, "note": n} for k, w, n in self.fired],
            "refused": self.refused, "agrees_with_label": self.agrees_with_label,
        }


#: Signals read from the *structured* context rather than the text.
#:
#: MEASURED 2026-10-02, and this reshaped the module. Of the eleven preset tasks, only two
#: carry a symptom ``text`` at all -- the two whose symptom is a traceback. The other nine
#: hand the playbook structured facts and no prose: ``venv-create`` gives ``cwd`` and
#: ``venv_dir``, ``port-occupied`` gives ``port: 8695`` and an open socket,
#: ``pip-conflict`` gives ``requirements``. A router that reads only text refuses 8 of 11,
#: which is safe but useless: the information needed to route is present and is simply not
#: in the field the text would occupy.
#:
#: So a context key is a signal, and it is weighted *above* prose. ``port: 8695`` in a
#: context dict is a fact the operator's tooling already established; the string "port 8695"
#: in a free-text report is somebody's paraphrase. Both are named ``port-occupied`` here, but
#: only one of them is the machine telling us what it saw.
CONTEXT_SIGNALS: tuple[tuple[str, str, int, str], ...] = (
    # (context key, kind, weight, note)
    ("venv_dir", "venv", 4, "the caller already named a virtualenv directory"),
    ("port", "port-occupied", 4, "the caller already named a port number"),
    ("socket", "port-occupied", 3, "a socket is open"),
    ("_socket", "port-occupied", 3, "a socket is open"),
    ("requirements", "pip-conflict", 3, "the caller pointed at a requirements file"),
    ("traceback", "missing-dependency", 4, "the caller passed a traceback"),
    # A changed file is what "which tests" is a question *about*: the caller has already
    # localised the edit and is asking for its blast radius. Only affected-tests answers
    # that.
    ("changed", "affected-tests", 4, "the caller named files it changed"),
    # A test target means a test problem. ``target`` is deliberately *not* routed to
    # ``flake-retry`` unconditionally: MEASURED, ``lint-fix`` hands over ``target:
    # "mod.py"`` and was routed to the test playbook, because the key was treated as "a
    # file to act on" and a test problem was the only such kind. The file's own extension
    # says which problem it is, and the caller chose the key precisely because the extension
    # carries the distinction.
    ("target", None, 0, ""),  # handled by _target_kind below
)


#: ``target`` is a file to act on, and the extension says which playbook that is. A test file
#: is a test problem; a source file is a lint problem. MEASURED 2026-10-02: routing ``target``
#: unconditionally to ``flake-retry`` sent ``lint-fix`` (``target: "mod.py"``) to the test
#: playbook, which is wrong in a way no amount of downstream verification would catch --
#: ``FlaskRetryPlaybook`` would run a test, find none, and report something.
_TEST_FILE = re.compile(r"(^|[\\/])(tests?[\\/])|(^|[\\/])test_[^\\/]+\.py$|"
                        r"(^|[\\/])[^\\/]*_test\.[a-z]+$", re.I)


def _target_kind(path: str) -> tuple[str, int, str] | None:
    """Route a ``target`` path by what kind of file it is."""
    if _TEST_FILE.search(path):
        return ("flake-retry", 4, f"{path} is a test file, so this is a test problem")
    if re.search(r"\.(py|pyi|js|ts|tsx|jsx|go|rs|java|rb|cs)$", path, re.I):
        return ("lint-fix", 4, f"{path} is a source file, so this is a lint/format problem")
    return None

#: Keys whose presence argues *against* a kind rather than for it.
#:
#: ``import-error-locator`` and ``missing-dependency`` are different jobs on the same input:
#: a traceback. One asks "which module is missing", the other asks "which of these frames is
#: mine". MEASURED 2026-10-02: a plain import traceback routed to ``missing-dependency``,
#: which is the more common answer but the wrong one for ``import-locator``'s task.
#:
#: They are separated on what the *traceback says*, not on which task it came from. A
#: locator question contains a request to place the failure in user code -- "my code", "which
#: file", "locate" -- and that is a stronger signal than the import error it also contains.
#: Scored rather than ordered, because the first-match version of this put the two in an
#: if/elif and the import-error branch was always reachable first.
IMPORT_LOCATOR_TEXT = re.compile(
    r"which\s+(?:file|module|frame)|my\s+(?:code|project)|user(?:'s)?\s+(?:code|frame)|"
    r"locate|定位|哪个文件|我的代码", re.I)

#: The word "frame" is the tell. "which of these traceback frames is mine" is a locating
#: question with no other vocabulary in it -- no "code", no "file", no "locate" -- and the
#: overfit run refused it for exactly that reason while the fixture's own text (a two-frame
#: traceback) routed correctly. ``frame`` is specific enough that admitting it costs
#: nothing: a missing-dependency report names a module, never a frame.
_TRACE_FRAME_LINE = re.compile(r"^\s*File\s+\"", re.M)


def _frame_count(text: str) -> int:
    return len(_TRACE_FRAME_LINE.findall(text or ""))


def _looks_like_a_locator(text: str) -> bool:
    """Whether a traceback is asking to be localised rather than diagnosed."""
    if not text:
        return False
    if IMPORT_LOCATOR_TEXT.search(text):
        return True
    if re.search(r"\bframes?\b|\bstack\b", text, re.I):
        return True
    # More than one frame, and at least one of them outside the working tree, is the shape
    # of "somebody else's code called into mine and I cannot tell which file is mine".
    if _frame_count(text) >= 2 and re.search(r"site-packages|dist-packages|"
                                             r"node_modules|/\.venv/|\\Lib\\", text, re.I):
        return True
    return False

#: Frame counting, which is what actually separates the two import tasks.
#:
#: MEASURED 2026-10-02, from the two fixtures' own text:
#:
#:   missing-dependency   one frame, then ModuleNotFoundError
#:     Traceback (most recent call last):
#:       File ".../rich/console.py", line 1, in <module>
#:     ModuleNotFoundError: No module named 'rich'
#:
#:   import-locator       two frames -- a site-packages frame *and* a user frame -- then
#:                         ModuleNotFoundError
#:     Traceback (most recent call last):
#:       File "C:/site-packages/flask/app.py", line 20, in main
#:         from flask import request
#:       File ".../app.py", line 1, in <module>
#:         import utils
#:     ModuleNotFoundError: No module named 'utils'
#:
#: Both end in the same exception, so any signal on the exception routes them identically --
#: which is what the first version did, sending ``import-locator`` to
#: ``missing-dependency``. The question they answer is different, and it is visible in the
#: traceback's *shape*: "which module is missing" is answerable from one frame, "which of
#: these frames is mine" needs more than one. Counting frames reads that directly, and it
#: reads it from the artifact rather than from the wording.
#:
#: The implementation above this comment is the one that runs. A second copy was written by
#: an edit that added the ``frames?`` branch and did not remove the original, so the later
#: definition silently won and the branch was dead -- "which of these traceback frames is
#: mine?" kept being refused. Duplicated definitions that shadow each other are the same
#: failure as the two-block function body in 12.37: the code reads as if the fix is present.


def _context_signals(context: dict[str, Any] | None) -> list[tuple[str, int, str]]:
    if not context:
        return []
    out: list[tuple[str, int, str]] = []
    for key, kind, weight, note in CONTEXT_SIGNALS:
        if key not in context:
            continue
        value = context.get(key)
        # A key present with a falsy value is a different statement from one absent: the
        # caller set ``port=None`` because it looked and found nothing, which is evidence
        # *against* port-occupied, not for it. Counted, and the note says so.
        if value in (None, "", [], {}, 0):
            out.append((kind, -weight, f"{key} is present but empty: looked and found nothing"))
            continue
        if kind is None:
            # ``target`` decides by extension rather than by a fixed kind.
            routed = _target_kind(str(value))
            if routed is not None:
                out.append(routed)
            continue
        out.append((kind, weight, note))
    return out


def signals_for_kind(kind: str) -> list[str]:
    """The notes for the signals that point at ``kind``. Used by the report and the tests.

    CONTEXT_SIGNALS entries are ``(key, kind, weight, note)`` -- four fields -- so unpacking
    three of them raised ValueError, and a test that iterates the whole vocabulary hit it
    immediately. The vocabulary walk is what made it visible; nothing in the routing path
    calls this function, which is why it survived the eleven-task run.
    """
    out = [s.note or s.pattern.pattern for s in SIGNALS if s.kind == kind]
    out.extend(note for _key, sig_kind, _w, note in CONTEXT_SIGNALS if sig_kind == kind)
    return out


def route(text: str, *, context: dict[str, Any] | None = None, label: str = "",
          margin: int = DEFAULT_MARGIN) -> OpsRoute:
    """Decide which playbook a symptom needs.

    ``context`` is the structured half of the symptom and is weighted above ``text``; see
    :data:`CONTEXT_SIGNALS` for why eight of the eleven preset tasks route on it alone.

    ``label`` is the task's declared category, when the caller has one. It is never used to
    decide -- only to record whether the decision agreed with it, so a mislabelled task
    shows up as a disagreement rather than being silently papered over.
    """
    result = OpsRoute()
    if not (text or "").strip() and not context:
        result.refused = "no symptom text and no context; nothing to route on"
        return result

    scores: dict[str, int] = {}

    def add(kind: str, weight: int, note: str | None) -> None:
        scores[kind] = scores.get(kind, 0) + weight
        if note:
            result.fired.append((kind, weight, note))

    for kind, weight, note in _context_signals(context):
        add(kind, weight, note)

    body = text or ""
    target_kind: str | None = None
    for sig in SIGNALS + KIND_SIGNALS:
        if sig.pattern.search(body):
            add(sig.kind, sig.weight, sig.note)
            if sig.target_kind:
                target_kind = sig.target_kind
                result.target_kind = sig.target_kind

    # A traceback that asks to be located is a different question from one that asks what is
    # missing, and the import-error signal fires on both. Scored, not ordered, so the
    # stronger evidence wins on its own merit -- see IMPORT_LOCATOR_TEXT.
    # "这个 ImportError 是我自己的代码引起的吗？" ties 1:1 with missing-dependency's
    # ImportError signal, and the margin rule then refuses it. That refusal is defensible but
    # it is the wrong answer: the sentence names *whose* code is at fault, which is the
    # locator's question and not a request to name a module. The "my own code" signal
    # therefore has to outweigh a bare exception name, so its weight is above the
    # missing-dependency import patterns rather than beside them.
    if _looks_like_a_locator(body):
        add("import-error-locator", 6,
            "the traceback asks to place the failure in user code, not to name a module")
        # And it takes the score away from the kind it was competing with, because a
        # traceback being localised is not a traceback being diagnosed.
        if scores.get("missing-dependency"):
            scores["missing-dependency"] -= 3

    # Negative evidence is only evidence if something positive outweighs it, so a kind whose
    # score has gone negative is dropped rather than routed to.
    live = {k: v for k, v in scores.items() if v > 0}
    if not live:
        negative = {k: v for k, v in scores.items() if v < 0}
        result.refused = (
            f"the symptom argues against every kind it could name "
            f"({', '.join(f'{k} {v}' for k, v in negative.items())}); escalating rather "
            f"than guessing"
        )
        return result

    ranked = sorted(live.items(), key=lambda kv: (-kv[1], kv[0]))
    top_kind, top_score = ranked[0]
    total = sum(live.values())
    result.kind = top_kind
    result.confidence = top_score / total if total else 0.0
    result.margin = (top_score - ranked[1][1]) if len(ranked) > 1 else top_score
    result.runner_up = ranked[1][0] if len(ranked) > 1 else None

    if label:
        result.agrees_with_label = (label == top_kind)

    # A narrow win over a strong runner-up is a guess. Refusing here is what keeps a wrong
    # playbook from running a real command, and it is the same direction as
    # ``select`` returning ``None`` for a kind nobody knows.
    if margin > 0 and len(ranked) > 1 and result.margin < margin:
        result.refused = (
            f"{top_kind!r} leads {ranked[1][0]!r} by only {result.margin} point(s) "
            f"(scores {dict(ranked)}); too close to call, so escalating"
        )
        return result

    if label and not result.agrees_with_label:
        # Not a refusal. The decision stands -- the context and the text are the input and
        # the label is the thing being checked -- but the disagreement is carried, because a
        # gate that silently overrode a label would be measuring nothing.
        result.fired.append(("__disagrees_with_label__", 0,
                             f"symptom reads as {top_kind!r}, task declares {label!r}"))
    return result
