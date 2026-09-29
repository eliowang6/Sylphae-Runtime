"""Negative-sample regression tests for natural-language action routing.

Every other routing suite in ``tests/`` asserts what the system *does* on input it
understands.  Nothing asserted what it must *refuse*, and that gap is how a bug
report about a floating period reached the process launcher and started a detached
service.

The cases below are frozen from that failure — routing an Eidolon diary-theme
punctuation fix (2026-09-22).  They are phrased the way a user phrased them, not the
way a parser would prefer, because the parser is what is under test.

Two invariants are covered here:

1. **No winner from no evidence.**  A softmax names an action even when every
   similarity is zero, and ``max()`` over an all-zero score dict returns whichever
   action happens to be first.  ``predict_action`` must report
   ``ActionVerb.AUTO`` instead of guessing.
2. **Absence of evidence is not evidence of absence.**  A repository that was never
   probed (no ``root``) is not a repository that was probed and found to be
   unstartable.  Collapsing the two vetoes ``start_service`` for every text-only
   caller.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sylphae.coprocessor.grounding import ground_natural_language_sync
from sylphae.coprocessor.protocol import ActionVerb
from sylphae.coprocessor.service_detect import (
    detect_service_entrypoint,
    detect_service_flavor,
)
from sylphae.model.semantic_grounder import (
    MIN_ACTION_SIMILARITY,
    get_semantic_grounder,
)
from sylphae.repo.polyglot import iter_project_files


def _actions(prompt: str, root: Path | None = None) -> list[str]:
    return [step.action for step in ground_natural_language_sync(prompt, root=root)]


# --------------------------------------------------------------------------- #
# Invariant 1: a request that matches no action prototype gets no action.
# --------------------------------------------------------------------------- #

#: Requests that name a symptom, a question, or an unrelated task.  None of these
#: express an action this system performs, and the first one is the real failure:
#: it was routed to ``start_service`` in a repository with no service entrypoint.
UNMATCHED_REQUESTS = [
    "输入框里的句号上下居中飞起来了，请修复",
    "这个句号位置不对",
    "按钮浮起来了",
    "文字上下居中飘起来了",
    "这个字体看起来有点奇怪",
    "字体有点奇怪",
    "run the tests",
    "跑一下测试",
    "测试没通过",
    "随便看看",
]


@pytest.mark.parametrize("prompt", UNMATCHED_REQUESTS)
def test_unmatched_request_never_routes_to_start_service(prompt: str):
    """A symptom report must never reach the process launcher."""
    assert ActionVerb.START_SERVICE.value not in _actions(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "run the tests",
        "跑一下测试",
        "啊",
        "随便看看",
    ],
)
def test_all_zero_similarity_does_not_produce_a_guess(prompt: str):
    """With nothing matching, the grounder must report uncertainty.

    ``run the tests`` scores 0.0 against every prototype; before this guard it was
    nonetheless reported as ``create_table`` purely because that prototype is first
    in the dict.
    """
    prediction = get_semantic_grounder().predict_action(prompt)
    assert prediction.action == ActionVerb.AUTO.value, (
        f"{prompt!r} was forced into {prediction.action!r} with no supporting evidence"
    )


def test_auto_is_reserved_for_requests_below_the_similarity_floor():
    """The floor itself must stay in the measured window."""
    grounder = get_semantic_grounder()

    def top_similarity(text: str) -> float:
        vector = grounder._vectorize(text)
        return max(
            grounder._cosine_sim(vector, proto)
            for proto in grounder._action_vectors.values()
        )

    # Measured floor window: unmatched requests top out near 0.07, real intents start
    # near 0.10.  If the constant drifts out of that window the gate stops working in
    # one direction or the other, so pin both sides.
    assert top_similarity("输入框里的句号上下居中飞起来了，请修复") < MIN_ACTION_SIMILARITY
    assert top_similarity("把导航背景调淡一点") > MIN_ACTION_SIMILARITY
    assert top_similarity("darken navbar primary color") > MIN_ACTION_SIMILARITY


def test_start_service_prototype_avoids_symptom_vocabulary():
    """Guard the prototype text itself, not just today's outcomes.

    ``起来`` used to sit in the ``start_service`` prototype.  It heads genuine
    service requests ("把服务跑起来") but also heads visual bug reports ("句号飞起来
    了"), and the shared token alone was enough to route the latter to the process
    launcher.  Restoring it would silently re-open that path.
    """
    prototype = get_semantic_grounder().ACTION_PROTOTYPES[ActionVerb.START_SERVICE.value]
    for token in ("起来", "运行测试", "跑测试", "测试服务"):
        assert token not in prototype, (
            f"{token!r} in the start_service prototype matches symptom reports and "
            f"test-run requests; it must not be there"
        )


# --------------------------------------------------------------------------- #
# Invariant 2: "not probed" is distinct from "probed and absent".
# --------------------------------------------------------------------------- #

def test_missing_repository_does_not_veto_service(tmp_path: Path):
    """Without a repository, ``start_service`` must still be reachable.

    Reported by ``test_service_runner.test_grounding_service_commands`` and
    ``test_ai_semantic_grounding.test_compound_natural_language_grounding_sync``:
    both call the grounding layer with no ``root``, and an earlier version of the
    entrypoint veto treated the resulting "unprobed" state as "no entrypoint",
    which disabled the action entirely for text-only callers.
    """
    steps = ground_natural_language_sync("启动后端测试服务器")
    assert len(steps) == 1
    assert steps[0].action == ActionVerb.START_SERVICE.value
    assert steps[0].target_hint == "backend"


def test_absent_entrypoint_vetoes_service(tmp_path: Path):
    """A probed repository with no runnable entrypoint must veto ``start_service``."""
    repo = tmp_path / "library"
    repo.mkdir()
    (repo / "lib.py").write_text("def helper():\n    return 1\n", encoding="utf-8")

    files = iter_project_files(repo)
    assert detect_service_entrypoint(repo, files) is False
    assert ActionVerb.START_SERVICE.value not in _actions("启动后端服务", root=repo)


def test_unprobed_repository_reports_none_not_false():
    """The tri-state must survive to the caller."""
    assert detect_service_entrypoint(None, ["a.py"]) is None
    assert detect_service_entrypoint(Path("."), []) is None
    assert detect_service_flavor(None, ["a.py"]) is None


@pytest.mark.parametrize(
    ("filename", "content", "expected_entrypoint", "expected_flavor"),
    [
        ("vite.config.ts", "export default {}\n", True, "frontend"),
        ("next.config.js", "module.exports = {}\n", True, "frontend"),
        ("server.py", "print('s')\n", True, None),
        ("wsgi.py", "application = None\n", True, "backend"),
        # A bare main.py that never binds a server is a script, not a service.
        ("main.py", "print('just a script')\n", False, None),
        # The same filename *with* a server binding is a service.
        ("main.py", "from fastapi import FastAPI\napp = FastAPI()\n", True, "backend"),
    ],
)
def test_entrypoint_detection_from_repository_evidence(
    tmp_path: Path,
    filename: str,
    content: str,
    expected_entrypoint: bool,
    expected_flavor: str | None,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / filename).write_text(content, encoding="utf-8")
    files = iter_project_files(repo)

    assert detect_service_entrypoint(repo, files) is expected_entrypoint
    assert detect_service_flavor(repo, files) == expected_flavor


def test_package_json_scripts_decide_launchability(tmp_path: Path):
    """``package.json`` alone is not evidence — its scripts are."""
    repo = tmp_path / "web"
    repo.mkdir()
    (repo / "index.js").write_text("export const a = 1;\n", encoding="utf-8")
    (repo / "package.json").write_text(
        '{"name": "web", "dependencies": {"react": "^19.0.0"}}', encoding="utf-8"
    )
    assert detect_service_entrypoint(repo, iter_project_files(repo)) is False

    (repo / "package.json").write_text(
        '{"name": "web", "scripts": {"dev": "vite"}}', encoding="utf-8"
    )
    assert detect_service_entrypoint(repo, iter_project_files(repo)) is True


# --------------------------------------------------------------------------- #
# Genuine requests must keep working — the gates above must not be over-eager.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "prompt",
    [
        "启动一下后端服务",
        "启动后端",
        "启动服务",
        "启动前端",
        "把后端服务启动",
        "launch the dev server",
    ],
)
def test_genuine_service_requests_still_route_to_service(prompt: str):
    """Unambiguous start requests must not be pushed into AUTO by the noise gate."""
    assert ActionVerb.START_SERVICE.value in _actions(prompt)


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("把导航栏的高度缩短一些", ActionVerb.STYLE.value),
        ("把超时时间修改为60", ActionVerb.STYLE.value),
        ("darken navbar primary color", ActionVerb.STYLE.value),
        ("新增一个用户表", ActionVerb.CREATE.value),
        ("给用户表加一个 phone 字段", ActionVerb.ADD_FIELD.value),
        ("add a login route", ActionVerb.ROUTE.value),
    ],
)
def test_real_intents_are_not_degraded_to_auto(prompt: str, expected: str):
    """The noise gate must not swallow requests that do carry an intent."""
    prediction = get_semantic_grounder().predict_action(prompt)
    assert prediction.action == expected
