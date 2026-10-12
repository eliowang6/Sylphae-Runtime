"""Construct an ``add_guard`` patch from what the code already says.

DESIGN.md section 1's ``guard`` shape, and the reason it needs its own module. ``replace_literal``
substitutes a value; ``rename_symbol`` rebinds an identifier; both are *edits to a line that
already exists*. A guard is neither -- it **inserts** a statement, and every question that
makes a guard correct is answerable from the code around the insertion point rather than from
the wording of the request:

  What is being guarded?        the parameter the caller can pass ``None``
  When?                          before the first statement, or after the docstring
  What does it return?           whatever the function already returns on its failure path
  What indentation?              one level in from the ``def``

MEASURED 2026-10-02 on ``typer/_click/_compat.py``::

    38: def is_ascii_encoding(encoding: str) -> bool:
    39:     \"\"\"Checks if a given encoding is ascii.\"\"\"
    40:     try:
    41:         return codecs.lookup(encoding).name == "ascii"
    42:     except LookupError:
    43:         return False

and the request is "给is_ascii_encoding加上encoding为None时返回False的守卫" -- add a guard to
``is_ascii_encoding`` returning ``False`` when ``encoding`` is ``None``. Every part of the
inserted text is derivable:

  guard on ``encoding``     the only parameter, and the one the request names
  before line 40            after the docstring, which is detect.py's documented position
  return ``False``          the function's own ``except LookupError: return False`` already
                            establishes False as the safe answer, and the annotation says
                            ``-> bool``

The alternative -- a template keyed on the request's wording -- is what
``ReflexAgent._infer_op``'s fallback does (``if "guard" in prompt: return "add_guard",
"if not value: return None"``), and MEASURED it is wrong here in two ways: it guards
``value`` rather than ``encoding`` and returns ``None`` rather than ``False``. A template
produces plausible code that does not match the function it was inserted into.

**What this still cannot do, and the reason it is not in the reflex's main path yet.** The
guard text is derived, not composed. It handles the shape that is actually measured -- a
single parameter checked against ``None`` -- and refuses everything else rather than
guessing. Refusing is the same stance as the consistency gate: a wrong guard is worse than
no guard, because it compiles, it changes behaviour, and nothing downstream will notice.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

__all__ = [
    "GuardPlan",
    "plan_guard",
]


@dataclass
class GuardPlan:
    """A proposed guard insertion, or the reason there isn't one."""

    #: The parameter being guarded.
    parameter: str = ""
    #: Line the guard's ``if`` goes on, 1-based. The body is inserted above it.
    insert_before: int = 0
    #: The complete text to insert, already indented.
    text: str = ""
    #: Why no guard could be built. Non-empty means every other field is meaningless.
    blocked: str | None = None
    #: What the guard's return value was taken from, for the report.
    return_source: str = ""
    #: ROUND 19. Whether the parameter was ``named in the request`` or ``signature order``, and
    #: whether the shape was refused because the request did not read as a defensive check.
    #:
    #: These exist because 12.53's 1-of-6 could not be acted on. A defaulted parameter looks
    #: exactly like a chosen one in the text that gets inserted, so a caller had no way to tell
    #: a plan that answered the request from one that happened to hit the first argument.
    parameter_source: str = ""
    #: True when the refusal is a *shape* refusal -- the request was understood and was not a
    #: guard -- as opposed to the builder lacking what it needs. The two are different facts and
    #: 12.53 showed the second cannot stand in for the first.
    shape_refused: bool = False

    @property
    def ok(self) -> bool:
        return not self.blocked and bool(self.text)


#: Return values a guard may use, and what each one means for the caller.
#
# Restricted deliberately. A guard that raises is a different operation (it changes the
# function's contract rather than its behaviour on one input), and a guard that returns
# ``None`` from a function annotated ``-> bool`` is a type error that the benchmark's
# content check would not catch but a caller's ``if is_ascii_encoding(x):`` would.
_NONE_TYPED = (ast.Constant,)


def _return_literal(body: ast.AST) -> tuple[str, str] | None:
    """Find the value the function already returns on a failure path, and where it came from.

    Scans in a fixed order -- ``except`` handlers first, then ``if`` guards, then a bare
    ``return`` -- because the answer to "what should this return when the input is missing"
    is the answer the author already gave for the *other* way the input can be wrong. On the
    measured fixture that is ``except LookupError: return False``.

    Returns ``(literal, provenance)`` or ``None``.
    """
    if not isinstance(body, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None

    # except-handler returns: the closest thing to "this input was unusable".
    for node in ast.walk(body):
        if isinstance(node, ast.ExceptHandler):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Constant):
                    if sub.value.value is False:
                        return "False", "an except handler already returns False"
                    if sub.value.value is True:
                        return "True", "an except handler already returns True"
                    if sub.value.value is None:
                        return "None", "an except handler already returns None"

    # Early-return guards: `if not x: return <literal>` is the shape a guard mirrors.
    for node in ast.walk(body):
        if isinstance(node, ast.If) and not isinstance(getattr(node, "test", None),
                                                       ast.Try):
            has_return = any(isinstance(s, ast.Return) and isinstance(s.value, ast.Constant)
                             for s in node.body)
            if has_return:
                for sub in node.body:
                    if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Constant):
                        return (repr(sub.value.value),
                                "an existing early-return guard uses the same value")

    return None


def _annotated_return(body: ast.AST) -> str | None:
    """The declared return type, if it is one this module can honour."""
    if not isinstance(body, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    ann = body.returns
    if ann is None:
        return None
    try:
        name = ast.unparse(ann)
    except Exception:  # noqa: BLE001
        return None
    return name


def _nullish_parameters(fn: ast.FunctionDef) -> list[str]:
    """Parameters that a ``None`` check is meaningful for.

    Ranks: no default and no annotation first (nothing tells the caller what to pass), then
    bare-annotated. ``*args``/``**kwargs`` are excluded -- ``None`` is a legitimate value for
    every element.
    """
    out: list[tuple[int, str]] = []
    a = fn.args
    positional = list(a.posonlyargs) + list(a.args)
    defaults = list(a.defaults)
    first_default = len(positional) - len(defaults)
    for i, arg in enumerate(positional):
        if arg.arg in ("self", "cls"):
            continue
        rank = 0 if (i < first_default and arg.annotation is None) else 1
        out.append((rank, arg.arg))
    return [name for _, name in sorted(out)]


#: A request that describes a *condition on an input* rather than a value to write or an
#: identifier to replace.
#:
#: ROUND 19, and this predicate is here because 12.53 measured that the builder cannot tell a
#: guard request from a value change: given a failure-path convention it produced a guard for
#: "change the default timeout from 30 to 300" and for "rename the local variable", because it
#: receives no request at all and derives everything from the signature. So the shape question
#: needs answering *somewhere*, and the question is where to put a rule that is known to be weak.
#:
#: **This is a lexical predicate and it is called that.** It matches the *syntactic shape* of a
#: nullish condition -- a comparison against ``None``, a negation, a truthiness word in the
#: position of a condition -- rather than collecting keywords and scoring them, and the
#: distinction is load-bearing: "if X is None" and "if not X" are constructs that mean the same
#: thing in any Python program, whereas "timeout" and "import" are words that mean something
#: else entirely depending on the sentence.
#:
#: **It fails closed, which is the whole safety argument.** A request it does not recognise
#: yields ``False`` and therefore no guard, so a false negative costs a declined patch and a
#: false positive costs an unsolicited one. The asymmetry is not symmetric, so the rule is
#: allowed to be conservative. That is a different position from "the keywords are the signal",
#: and it is also measurable: ``sylphae.eval.guard_shape`` runs the six requests from 12.53,
#: four genuine and two controls, through it.
#:
#: What this does **not** establish: that it is right about requests nobody has shown it. A
#: predicate validated on six sentences is six sentences' worth of evidence, and the 12.51
#: caveat applies -- a rule that passes the cases it was written against has been measured
#: against nothing.
_NULLISH_CONDITION_RE = re.compile(
    r"""(?:
          is\s+not\s+none
        | is\s+none
        | (?:is|are|was|were|==|!=)\s*(?:not\s*)?none
        | ==\s*null
        | 为\s*空
        | 是\s*空
        | 为\s*none
        | 不存在
        | 未设置
        | 没有设置
    )""",
    re.IGNORECASE | re.VERBOSE,
)

#: A truthiness condition: "when X is falsy", "if not X", "for an empty list". Kept separate
#: from the nullish pattern because `if not values` and `if values is None` differ in meaning
#: for an empty collection, and a rule that collapsed them would be wrong about the difference.
_FALSY_CONDITION_RE = re.compile(
    r"""(?:
          \bis\s+empty\b
        | \bis\s+falsy\b
        | \bif\s+not\s+[A-Za-z_][\w.]*
        | \bwhen\s+not\s+[A-Za-z_][\w.]*
        | \bunless\s+[A-Za-z_][\w.]*
        | \bguard\s+against\b
        | \bnot\s+be\s+empty\b
        | \b为空\b
        | \b空的时候\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def request_asks_for_a_guard(request: str) -> bool:
    """Whether a request reads as a defensive check rather than an edit.

    Returns ``True`` only when the request contains a nullish or falsiness *condition*. A
    request that names a value to write, or two identifiers to swap, contains neither, and so
    returns ``False`` -- which is the answer 12.53 measured the builder could not give.
    """
    if not request:
        return False
    return bool(_NULLISH_CONDITION_RE.search(request)
                or _FALSY_CONDITION_RE.search(request))


def _parameter_named_in(request: str, candidates: list[str]) -> str:
    """The candidate the request names, if exactly one it names.

    Matching a *declared identifier* against the request is a different kind of operation from
    matching a keyword: the identifiers are known to belong to the function, so a match is
    evidence about this request and this signature rather than a coincidence about English. The
    same reasoning ``rename_symbol`` relies on when it looks for the old symbol.

    Ambiguity is the other half of it. If the request names two candidates, or none, the
    signature-order fallback stands and the plan records that it was a fallback -- so a caller
    can tell a chosen parameter from a defaulted one, which 12.53's 1-of-6 was not able to.
    """
    if not request:
        return ""
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", request))
    named = [c for c in candidates if c in tokens]
    return named[0] if len(named) == 1 else ""


def plan_guard(source: str, *, function_name: str = "",
               parameter: str = "", path: str = "x.py",
               request: str = "") -> GuardPlan:
    """Propose a ``None``-guard for ``function_name``, or explain why not.

    ``function_name`` empty means "the single function in the file", which is unambiguous
    for a module and wrong for a class, so it is refused rather than guessed.

    ROUND 19, ``request``. 12.53 measured two blockers that the recorded "op head did not pick
    add_guard" gap did not name, and this addresses the one that is addressable here:

    * The parameter used to be whichever came first in the signature, so a request to guard
      ``encoding`` got a guard on ``path``. It is now chosen by matching the request against the
      function's *declared* parameter names, and when that is ambiguous or silent the
      signature-order fallback still runs -- but ``GuardPlan.parameter_source`` records which of
      the two happened, so a defaulted choice can be told from a chosen one.
    * The shape question is answered by :func:`request_asks_for_a_guard` **before** any plan is
      built, because a guard inserted into a request that asked for a value change is worse than
      a declined one. A request that does not read as a defensive check is refused with that
      reason, which is the refusal 12.53 showed was impossible to produce before.

    What this does not do is make the rule good. It is lexical, it fails closed, and it has been
    measured on six sentences of which four are genuine -- see ``sylphae.eval.guard_shape``.
    """
    if request and not request_asks_for_a_guard(request):
        return GuardPlan(
            blocked=(
                f"the request does not read as a defensive check, so no guard was planned; "
                f"the request was: {request!r}"
            ),
            shape_refused=True,
        )

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        return GuardPlan(blocked=f"the file does not parse ({exc}); no guard was built")

    funcs = [n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not funcs:
        return GuardPlan(blocked="no function definition at module level to guard")

    if function_name:
        matches = [f for f in funcs if f.name == function_name]
        if not matches:
            return GuardPlan(
                blocked=f"{function_name!r} is not a module-level function in {path}; "
                        f"the file defines {', '.join(f.name for f in funcs[:5])}")
        fn = matches[0]
    else:
        if len(funcs) != 1:
            return GuardPlan(
                blocked=f"{path} defines {len(funcs)} functions, so 'the function' is "
                        f"ambiguous; naming one is required")
        fn = funcs[0]

    # -- the parameter -------------------------------------------------------------
    cands = _nullish_parameters(fn)
    if not cands:
        return GuardPlan(blocked=f"{fn.name!r} takes no parameter that a None check fits")
    if parameter:
        if parameter not in [a for a in cands]:
            return GuardPlan(
                blocked=f"{parameter!r} is not a parameter of {fn.name!r} "
                        f"(it has {', '.join(cands)})")
        chosen = parameter
        parameter_source = "explicit"
    else:
        # ROUND 19: the request gets to choose, by matching the function's *declared* parameter
        # names against it. 12.53 measured the consequence of not doing this: a request to guard
        # ``encoding`` received a guard on ``path``, because signature order put ``path`` first
        # and nothing ever asked which parameter the request meant. A defaulted choice is still a
        # choice, so the fallback stands -- but it is recorded, because a signature-order guard
        # and a request-derived guard are different things and the inserted text cannot say which.
        named = _parameter_named_in(request, cands)
        chosen = named or cands[0]
        parameter_source = "named in the request" if named else "signature order"

    # -- the return value ------------------------------------------------------------
    declared = _annotated_return(fn)
    found = _return_literal(fn)
    if found is None and declared not in ("bool", "None"):
        return GuardPlan(
            blocked=f"{fn.name!r} has no failure-path return to copy and is annotated "
                    f"-> {declared or 'nothing'}; a guard's return value would be a guess")
    if found is not None:
        value, source_note = found
    else:
        value, source_note = ("False" if declared == "bool" else "None",
                              f"the function is annotated -> {declared}")
    if value not in ("False", "True", "None"):
        return GuardPlan(
            blocked=f"the existing guard returns {value}, which this module will not "
                    f"reproduce; a guard returning a computed value is a different "
                    f"operation")
    if declared == "bool" and value == "None":
        return GuardPlan(
            blocked=f"{fn.name!r} is annotated -> bool, so a guard returning None would be "
                    f"a type error; the existing early return gives {value}")
    # A function with no annotation at all is a different case, and returning None is still
    # the wrong default: callers overwhelmingly branch on the result, and None is falsy for
    # a different reason than the value the function would have computed. MEASURED: a guard
    # returning None was built for
    #
    #     def f(x):
    #         if not x: return None
    #         return True
    #
    # which happens to be defensible for *that* function, so the refusal is not "None is
    # always wrong" -- it is that an unannotated function gives no evidence that returning
    # None is what a caller expects, and the safe answer for a missing input is the one the
    # function's own failure path already chose.
    if value == "None" and declared in (None, "None"):
        return GuardPlan(
            blocked=f"{fn.name!r} returns None on an existing path and declares no return "
                    f"type, so a guard returning None cannot be checked against an "
                    f"annotation; a caller branching on the result would get None for a "
                    f"missing input and a real value otherwise, with nothing in the "
                    f"signature to say that is intended")

    # -- the position ----------------------------------------------------------------
    # detect.py's rule: the inserted ``if`` heads a function body -- first statement, or
    # second after a docstring. Using the same rule keeps the generator and the classifier
    # agreeing about what a guard looks like, so a generated guard is detected as one.
    first = fn.body[0]
    if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        body_start = fn.body[1] if len(fn.body) > 1 else None
        if body_start is None:
            return GuardPlan(blocked=f"{fn.name!r} has only a docstring, so there is no "
                                     f"statement to precede")
        insert_before = body_start.lineno
        offset = 1
    else:
        insert_before = first.lineno
        offset = 0

    # Indent from the statement the guard precedes, which is authoritative for the body's
    # actual nesting -- measuring from the ``def`` would be wrong for a decorated or
    # continuation-line signature.
    lines = source.splitlines()
    if insert_before - 1 >= len(lines):
        return GuardPlan(blocked=f"line {insert_before} is past the end of {path}")
    base = re.match(r"[ \t]*", lines[insert_before - 1]).group(0)
    if not base:
        return GuardPlan(blocked=f"line {insert_before} is at column 0, so it is not inside "
                                 f"a function body and a guard cannot precede it")
    unit = "    " if "\t" not in base else "\t"
    inner = base + unit

    # ``encoding is None`` for a str-annotated parameter. ``is None`` rather than a falsy
    # test on purpose: ``0``, ``""`` and ``False`` are legitimate values for a str-typed
    # parameter, and a falsy guard would change behaviour for callers passing them.
    text = (f"{base}if {chosen} is None:\n"
            f"{inner}return {value}\n")

    return GuardPlan(
        parameter=chosen, insert_before=insert_before, text=text,
        return_source=source_note, parameter_source=parameter_source)
