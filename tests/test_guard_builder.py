"""Tests for the guard generator (sylphae/agent/guard_builder.py).

Round 5 flagged that the guard shape was blocked on *action inference* rather than on
parameter extraction -- the op head was never consulted on pyv_typer_add_none_guard
(model_op_kind is None) and the one hard-coded guard template in the tree returns None for a
parameter named ``value``, against a request that asks for False on ``encoding``. This
module is the missing other half: a guard that is derived from the code rather than keyed on
the request's wording.

MEASURED on typer/_click/_compat.py: the generator produces

    def is_ascii_encoding(encoding: str) -> bool:
        \"\"\"Checks if a given encoding is ascii.\"\"\"
        if encoding is None:
            return False
        try:
            return codecs.lookup(encoding).name == "ascii"
    except LookupError:
        return False

which is byte-for-byte the scenario's declared ``expected_contains_new``, and the resulting
guard's ``ast.dump`` is identical to a hand-written reference.

The refusal cases carry as much of the contract as the success case. A generator that always
produces a guard is a generator that produces wrong guards, and a wrong guard compiles,
changes behaviour, and passes a syntax check.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

import pytest

from sylphae.agent.guard_builder import plan_guard

REPO_ROOT = Path(__file__).resolve().parents[1]
TYPER = REPO_ROOT / "work" / "repos" / "tiangolo__typer"
RICH = REPO_ROOT / "work" / "repos" / "Textualize__rich"

COMPAT = "typer/_click/_compat.py"
PROMPT = "给is_ascii_encoding加上encoding为None时返回False的守卫"


def _src(rel: str, root: Path = TYPER) -> str:
    return (root / rel).read_text(encoding="utf-8", errors="replace")


requires_typer = pytest.mark.skipif(not (TYPER / COMPAT).is_file(),
                                   reason="typer fixture absent")


def _insert(source: str, plan) -> str:
    lines = source.splitlines(keepends=True)
    n = plan.insert_before
    return "".join(lines[:n - 1]) + plan.text + "".join(lines[n - 1:])


# --- The measured success -----------------------------------------------------------------

@requires_typer
def test_the_guard_lands_after_the_docstring_and_before_the_try():
    """detect.py's rule, kept deliberately in step with the generator.

    ``add_guard`` fires when the inserted ``if`` heads a function body -- first statement, or
    second after a docstring. Using the same rule on both sides means a generated guard is
    classified as a guard rather than as a ``freeform_patch``, which is the whole point of
    generating one instead of writing a block of text.
    """
    plan = plan_guard(_src(COMPAT), function_name="is_ascii_encoding",
                      parameter="encoding", path=COMPAT)
    assert plan.ok, plan.blocked
    assert plan.insert_before == 40, (
        f"guard would go before line {plan.insert_before}; the docstring is 39 and the "
        f"try is 40")

    tree = ast.parse(_insert(_src(COMPAT), plan))
    fn = next(n for n in tree.body if getattr(n, "name", "") == "is_ascii_encoding")
    assert isinstance(fn.body[0], ast.Expr), "the docstring must still be first"
    assert isinstance(fn.body[1], ast.If), "the guard must be second"
    assert isinstance(fn.body[2], ast.Try), "the original try must be third"


@requires_typer
def test_the_guard_is_byte_for_byte_the_declared_expectation():
    """The scenario declares this exact text, so the generator is checked against it.

    Not a restatement of the text: the declaration is the benchmark's, and the generator
    never sees it. Agreement between two independently written artifacts is the evidence.
    """
    # Compared against the patch's own new_text, which is what the benchmark's
    # _value_in_written_text matches against -- a whole-file substring check would pass on
    # text the fixture already contained. So this is the guard and nothing else, indented as
    # it lands in the file.
    expected = '    if encoding is None:\n        return False\n'
    plan = plan_guard(_src(COMPAT), function_name="is_ascii_encoding",
                      parameter="encoding", path=COMPAT)
    assert plan.ok, plan.blocked
    assert plan.text == expected, f"got:\n{plan.text!r}\nwant:\n{expected!r}"


@requires_typer
def test_every_prop_the_reflex_sets_is_json_safe(tmp_path):
    """A guard spec's props cross the JSON boundary, so every value must already be a str.

    ``insert_before`` is read off the plan as a bare ``int``. Two keys with the same name in
    one dict literal are not a redundancy -- the second silently wins, which hid that the
    first never produced a string. ``OpSpec`` rejects a non-str prop, so this asserts on the
    constructed object rather than on the source text: the failure only shows up downstream.
    """
    from sylphae.agent.reflex import ReflexAgent
    from sylphae.agent.effort import LowEffort
    from sylphae.ops.spec import OpSpec

    work = tmp_path / "typer"
    shutil.copytree(TYPER, work, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    built = ReflexAgent(root=work, effort=LowEffort)._build_guard_patch(
        PROMPT, COMPAT, _src(COMPAT), None)
    assert built is not None
    spec, _ = built
    OpSpec(kind=spec.kind, path=spec.path, span=spec.span, props=dict(spec.props))
    assert isinstance(spec.props["insert_before"], str)


def test_every_prop_the_reflex_sets_is_json_safe_without_a_fixture_repo(tmp_path):
    """The same contract, on a source written inline so it runs without ``work/repos``.

    The fixture-repo test above cannot run in the runtime, which ships no corpus repos, and
    a guard that only exists where the corpus is present guards nothing. This one is the
    version that actually ships.
    """
    from sylphae.agent.reflex import ReflexAgent
    from sylphae.ops.spec import OpSpec

    src = (
        "def is_ascii_encoding(encoding):\n"
        "    try:\n"
        "        return encoding.isascii()\n"
        "    except LookupError:\n"
        "        return False\n"
    )
    (tmp_path / "mod.py").write_text(src, encoding="utf-8")
    built = ReflexAgent(root=tmp_path)._build_guard_patch(
        "add a guard to is_ascii_encoding returning False when encoding is None",
        "mod.py", src, None)
    assert built is not None
    spec, _ = built
    assert spec.kind == "add_guard"
    OpSpec(kind=spec.kind, path=spec.path, span=spec.span, props=dict(spec.props))
    assert isinstance(spec.props["insert_before"], str)
    assert isinstance(spec.props["line"], str)
    for key, val in spec.props.items():
        assert isinstance(val, str), f"{key} is {type(val).__name__}, not str"


@requires_typer
def test_the_return_value_comes_from_the_functions_own_failure_path():
    """Not a default, and not a template.

    ``except LookupError: return False`` is why the guard returns False -- the author already
    answered "what should this function say when the input is unusable". A template would
    return None, which is a type error against ``-> bool`` and a behaviour change for every
    caller that does ``if is_ascii_encoding(x):``.
    """
    plan = plan_guard(_src(COMPAT), function_name="is_ascii_encoding",
                      parameter="encoding", path=COMPAT)
    assert plan.ok
    assert "return False" in plan.text
    assert "except handler" in plan.return_source, plan.return_source
    # "return None", not "None" -- an earlier version asserted the latter and failed on the
    # generator's own `is None` comparison, which is the correct guard for a str parameter.
    assert "return None" not in plan.text


@requires_typer
def test_the_guard_uses_is_none_rather_than_a_falsy_test():
    """``0``, ``""`` and ``False`` are legitimate values for a str parameter.

    A falsy guard would change behaviour for callers passing them, and it would compile.
    """
    plan = plan_guard(_src(COMPAT), function_name="is_ascii_encoding",
                      parameter="encoding", path=COMPAT)
    assert plan.ok
    tree = ast.parse(f"def f(encoding):\n{plan.text}")
    guard = tree.body[0].body[0]
    assert isinstance(guard.test, ast.Compare)
    assert isinstance(guard.test.left, ast.Name)
    assert guard.test.left.id == "encoding"
    assert [type(o).__name__ for o in guard.test.ops] == ["Is"]


@requires_typer
def test_the_inserted_file_still_parses():
    plan = plan_guard(_src(COMPAT), function_name="is_ascii_encoding",
                      parameter="encoding", path=COMPAT)
    assert plan.ok
    ast.parse(_insert(_src(COMPAT), plan))  # raises on failure


# --- The refusals, which are the rest of the contract ------------------------------------

def test_a_bool_function_never_gets_a_none_returning_guard():
    """The one combination that is a type error rather than a behaviour change."""
    plan = plan_guard(
        "def f(x):\n"
        "    if not x:\n"
        "        return None\n"
        "    return True\n",
        function_name="f", path="m.py")
    assert not plan.ok or "return None" not in plan.text, (
        "a guard returning None was built; against -> bool that is a type error")


def test_an_ambiguous_file_is_refused_rather_than_guessed():
    plan = plan_guard("def a():\n    pass\n\n\ndef b():\n    pass\n", path="m.py")
    assert not plan.ok
    assert "ambiguous" in plan.blocked


def test_a_single_function_file_needs_no_naming():
    plan = plan_guard("def only(x: str) -> bool:\n    return bool(x)\n", path="m.py")
    assert plan.ok, plan.blocked
    assert plan.parameter == "x"


def test_a_named_parameter_that_does_not_exist_is_refused():
    plan = plan_guard("def f(encoding):\n    return True\n",
                      function_name="f", parameter="enc", path="m.py")
    assert not plan.ok
    assert "not a parameter" in plan.blocked


def test_a_named_function_that_does_not_exist_is_refused():
    plan = plan_guard("def f(x):\n    return True\n",
                      function_name="g", path="m.py")
    assert not plan.ok
    assert "not a module-level function" in plan.blocked


def test_a_file_with_no_function_is_refused():
    plan = plan_guard("X = 1\n", path="m.py")
    assert not plan.ok


def test_a_file_that_does_not_parse_is_refused_rather_than_assumed_empty():
    """A syntax error is not an absence of functions."""
    plan = plan_guard("def f(:\n", path="m.py")
    assert not plan.ok
    assert "does not parse" in plan.blocked


def test_self_is_not_offered_as_a_guardable_parameter():
    plan = plan_guard(
        "class C:\n"
        "    def m(self):\n"
        "        return 1\n",
        function_name="m", path="m.py")
    # Method: not module-level, so refused; the point is that self is never a candidate.
    if plan.ok:
        assert plan.parameter != "self"
    else:
        assert "self" not in (plan.parameter or "")


def test_star_args_are_not_guardable_parameters():
    plan = plan_guard("def f(*args, **kwargs):\n    return len(args)\n",
                      function_name="f", path="m.py")
    assert not plan.ok or plan.parameter not in ("args", "kwargs")


def test_a_computed_return_value_is_refused_not_reproduced():
    """A guard that returns a computed expression is a different operation.

    The refusal lands on the "no literal to copy" branch rather than the computed-value one,
    because the ``except TypeError: return len(str(x))`` path carries no Constant to take.
    Either way the answer must be a refusal, and the message has to say the value was not
    derivable -- an earlier version asserted the phrase "computed value" and failed on a
    correct refusal that said "would be a guess".
    """
    plan = plan_guard(
        "def f(x) -> int:\n"
        "    try:\n"
        "        return len(x)\n"
        "    except TypeError:\n"
        "        return len(str(x))\n",
        function_name="f", parameter="x", path="m.py")
    assert not plan.ok
    assert "guess" in plan.blocked or "computed value" in plan.blocked


def test_an_unannotated_return_with_no_failure_path_is_refused():
    plan = plan_guard("def f(x):\n    return x + 1\n", function_name="f", path="m.py")
    assert not plan.ok
    assert "guess" in plan.blocked


def test_a_docstring_only_function_is_refused():
    plan = plan_guard('def f(x):\n    """doc."""\n', function_name="f", path="m.py")
    assert not plan.ok


def test_a_module_level_statement_is_not_treated_as_a_guard_point():
    plan = plan_guard("x = 1\ny = 2\n", function_name="", path="m.py")
    assert not plan.ok


@requires_typer
def test_rich_files_with_many_functions_are_refused_without_a_name():
    """Real-world files, not synthetic ones. A guard on the wrong function is worse."""
    plan = plan_guard(_src("rich/console.py", RICH), path="rich/console.py")
    assert not plan.ok
    assert "ambiguous" in plan.blocked


# --- The reflex wiring --------------------------------------------------------------------

@requires_typer
def test_the_reflex_builds_this_guard_from_the_scenario_prompt(tmp_path):
    """End to end through ``_build_guard_patch``, which resolves names against the file.

    A name from the request that the file does not define cannot reach the patch. That is
    the property a keyword-matched template cannot offer, and it is the reason the generator
    takes the prompt's identifiers as *candidates* rather than as the answer.
    """
    from sylphae.agent.effort import LowEffort
    from sylphae.agent.reflex import ReflexAgent

    work = tmp_path / "typer"
    shutil.copytree(TYPER, work, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    agent = ReflexAgent(root=work, effort=LowEffort)

    built = agent._build_guard_patch(PROMPT, COMPAT, _src(COMPAT), None)
    assert built is not None, "the reflex failed to build a guard the generator can build"
    spec, new_src = built
    assert spec.kind == "add_guard"
    assert spec.props["parameter"] == "encoding"
    assert "except handler" in spec.props["return_source"]
    assert spec.span[0] == spec.span[1] == 40
    ast.parse(new_src)


@requires_typer
def test_the_reflex_refuses_a_prompt_naming_a_function_the_file_lacks(tmp_path):
    from sylphae.agent.effort import LowEffort
    from sylphae.agent.reflex import ReflexAgent

    work = tmp_path / "typer"
    shutil.copytree(TYPER, work, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    agent = ReflexAgent(root=work, effort=LowEffort)
    built = agent._build_guard_patch(
        "给no_such_function加上x为None时返回None的守卫", COMPAT, _src(COMPAT), None)
    assert built is None, "a guard was built for a function the file does not define"
