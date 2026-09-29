"""The trained model must actually be reachable from the agent.

The bug these lock down: ``ReflexAgent.__init__`` accepted a ``tokenizer`` argument and
never stored it, so ``_infer_policy`` raised ``AttributeError`` on ``self.tokenizer`` for
every caller that passed a model.  Nothing caught it because **no test in the suite ever
constructed an agent with a model** -- all thirteen call sites pass ``root=`` and
``effort=`` only.  So a green suite said nothing about the neural path, and every
"real-world" number was produced by the heuristic policy while a trained checkpoint sat
unused on disk.

These tests build a deliberately tiny real ``SylphaeModel`` (one layer, ``d_model=64``) so the
neural path is exercised for real rather than stubbed -- a stub world is a second
implementation, and the whole point here is that the second implementation disagreed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sylphae.agent.effort import LowEffort
from sylphae.agent.explorer import CandidateResult
from sylphae.agent.reflex import ReflexAgent
from sylphae.model.config import ModelConfig
from sylphae.model.policy import SylphaeModel
from sylphae.model.tokenizer import CodeTokenizer
from sylphae.ops.vocab import OP_KINDS

TOKENIZER_PATH = Path("sylphae/model/artifacts/tokenizer.json")

#: Forced op index used by the determinism test.  ``change_return`` is chosen because the
#: heuristic policy never emits it for the prompt below, so the assertion cannot pass by
#: accident if the neural branch is skipped.
FORCED_OP_INDEX = OP_KINDS.index("change_return")

#: A prompt the heuristic policy answers with ``replace_literal`` -- a different op.
PROMPT = "把超时时间修改为60"


def _tiny_model(n_ops: int = len(OP_KINDS)) -> SylphaeModel:
    return SylphaeModel(ModelConfig(
        d_model=64, n_layers=1, d_state=16, n_heads=4, chunk_size=16,
        d_ff=64, max_len=128, vocab_size=32768,
        n_ops=n_ops, n_candidates=8, max_params=16, ctx_tokens=32,
    ))


def _tokenizer() -> CodeTokenizer:
    if not TOKENIZER_PATH.exists():
        pytest.skip(f"tokenizer artifact not present at {TOKENIZER_PATH}")
    return CodeTokenizer.load(str(TOKENIZER_PATH))


def _candidate(path: str = "app.py") -> CandidateResult:
    return CandidateResult(
        path=path, score=0.9, contributing_tools={"path_lexical"},
        snippet="TIMEOUT = 30", line=1, symbols=[], from_fallback=False,
    )


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "app.py").write_text("TIMEOUT = 30\n", encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------- #
# Invariant 1: the tokenizer is stored, not dropped.
# --------------------------------------------------------------------------- #


def test_tokenizer_is_stored_on_the_agent(tmp_path: Path) -> None:
    tok = _tokenizer()
    agent = ReflexAgent(root=_repo(tmp_path), tokenizer=tok, effort=LowEffort)

    assert agent.tokenizer is tok, (
        "ReflexAgent.__init__ accepted `tokenizer` and dropped it. _infer_policy gates the "
        "neural path on self.tokenizer, so dropping it made the model path unreachable."
    )


def test_agent_without_a_model_still_has_a_tokenizer_attribute(tmp_path: Path) -> None:
    agent = ReflexAgent(root=_repo(tmp_path), effort=LowEffort)

    assert agent.tokenizer is None
    assert agent.last_model_error is None


# --------------------------------------------------------------------------- #
# Invariant 2: a model without a tokenizer fails at construction, not mid-inference.
# --------------------------------------------------------------------------- #


def test_model_without_tokenizer_is_rejected_at_construction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model but no tokenizer"):
        ReflexAgent(root=_repo(tmp_path), model=_tiny_model(), effort=LowEffort)


# --------------------------------------------------------------------------- #
# Invariant 3: with a model present, the model decides -- not the heuristic.
# --------------------------------------------------------------------------- #


def test_the_neural_policy_decides_the_op_when_a_model_is_passed(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    model = _tiny_model()
    agent = ReflexAgent(root=root, model=model, tokenizer=_tokenizer(), effort=LowEffort)

    # Zero the op head's weights and set a one-hot bias: the argmax is then exactly
    # FORCED_OP_INDEX regardless of the hidden state, so the returned op kind states
    # which branch produced it.
    with torch.no_grad():
        model.op_head.weight.zero_()
        model.op_head.bias.zero_()
        model.op_head.bias[FORCED_OP_INDEX] = 1.0

    op_kind, _param, _conf, _mem = agent._infer_policy(PROMPT, _candidate(), [_candidate()])

    heuristic_kind, _p, _c, _m = agent._heuristic_policy(PROMPT, _candidate())
    assert heuristic_kind != OP_KINDS[FORCED_OP_INDEX], (
        "the test is vacuous: the heuristic already emits the forced op for this prompt"
    )
    assert op_kind == OP_KINDS[FORCED_OP_INDEX], (
        f"expected the model's forced op {OP_KINDS[FORCED_OP_INDEX]!r}, got {op_kind!r} "
        f"(the heuristic would have said {heuristic_kind!r}). The neural branch did not run. "
        f"last_model_error={agent.last_model_error!r}"
    )
    assert agent.last_model_error is None, (
        f"the neural branch raised and was silently swallowed: {agent.last_model_error}"
    )


# --------------------------------------------------------------------------- #
# Invariant 4: end to end, a model-backed agent runs and does not fall back quietly.
# --------------------------------------------------------------------------- #


def test_agent_runs_end_to_end_with_a_model(tmp_path: Path) -> None:
    import asyncio

    agent = ReflexAgent(
        root=_repo(tmp_path), model=_tiny_model(), tokenizer=_tokenizer(), effort=LowEffort,
    )
    result = asyncio.run(agent.run(prompt=PROMPT, dry_run=True, verify_in_background=False))

    assert result is not None
    assert agent.last_model_error is None, (
        "the model path raised and fell back to the heuristic policy without saying so: "
        f"{agent.last_model_error}"
    )


# --------------------------------------------------------------------------- #
# Invariant 5: the agent says which policy produced the patch.
# --------------------------------------------------------------------------- #
#
# `run()` falls back to the heuristic when the model's answer cannot be built into a
# patch.  That rescue is correct behaviour, but reporting it as a model result is not:
# on the 14-scenario benchmark the model produced a patch on 6 scenarios and every one
# of them was wrong, while the 4 that passed under `heuristic_fallback` were the
# heuristic's work.  "The model answered" and "the model was consulted and something
# else answered" have to be distinguishable from outside.


def test_policy_source_is_recorded_as_model_when_the_model_builds_the_patch(tmp_path: Path) -> None:
    import asyncio

    agent = ReflexAgent(
        root=_repo(tmp_path), model=_tiny_model(), tokenizer=_tokenizer(), effort=LowEffort,
    )
    # `TIMEOUT = 30` -> 60 is buildable, so the model's answer is the one that ships.
    agent._infer_policy = lambda *a, **k: ("replace_literal", "60", 0.9, None)

    asyncio.run(agent.run(prompt=PROMPT, dry_run=True, verify_in_background=False))

    assert agent.last_policy_source == "model", (
        f"expected the model's own patch to be attributed to it, got "
        f"{agent.last_policy_source!r}"
    )


def test_policy_source_marks_a_heuristic_rescue_as_such(tmp_path: Path) -> None:
    import asyncio

    agent = ReflexAgent(
        root=_repo(tmp_path), model=_tiny_model(), tokenizer=_tokenizer(), effort=LowEffort,
    )
    # An op the model names but that cannot be built here, so the heuristic rescues it.
    agent._infer_policy = lambda *a, **k: ("change_return", "", 0.5, None)

    asyncio.run(agent.run(prompt=PROMPT, dry_run=True, verify_in_background=False))

    assert agent.last_policy_source == "heuristic_fallback", (
        "the patch was produced by the heuristic after the model's answer could not be "
        f"built, so it must not be attributed to the model; got "
        f"{agent.last_policy_source!r}"
    )


def test_a_patch_the_model_did_not_determine_goes_to_the_heuristic(tmp_path: Path) -> None:
    """Blocker 2, the precedence rule (ROADMAP 12.4).

    ``policy_source`` describes the route, not the decision.  Some builders never read
    ``op_kind`` or ``param_val`` -- ``decompose_qualitative_intents`` and
    ``_build_named_value_patch`` read the *prompt* -- so the model's answer can be
    buildable and still not be the thing that produced the patch.  The ablation is what
    separates those two cases: feeding the ablation constant produced the identical
    patch, so ``model_output_used`` is False, and on all six qualitative benchmark
    scenarios that meant a prompt-built patch was attributed to the model and gated on
    the model's 0.01-0.14 confidence while the same patches shipped at 0.656-0.88 as
    ``heuristic_fallback`` -- 0/6 against 4/6.

    The rule under test: a patch the model did not determine is not the model's, so the
    heuristic gets a turn and its own patch and confidence are what ship.
    """
    import asyncio

    from sylphae.ops.spec import OpSpec

    H_CONF = 0.72  # over the LowEffort 0.60 gate, so the heuristic alone ships it
    agent = ReflexAgent(
        root=_repo(tmp_path), model=_tiny_model(), tokenizer=_tokenizer(), effort=LowEffort,
    )
    # The model answers confidently with a buildable op, so the pre-fix branch would
    # have set policy_source="model" and shipped at 0.9 with no heuristic turn.
    agent._infer_policy = lambda *a, **k: ("freeform_patch", "", 0.9, None)
    agent._heuristic_policy = lambda *a, **k: ("replace_literal", "60", H_CONF, None)

    # A prompt-driven builder: the same patch for every (op_kind, param_val), including
    # the ablation constant, so the ablation cannot distinguish the model's answer from
    # the constant.  A fresh equal spec per call keeps the comparison structural rather
    # than by identity.
    def _prompt_only_builder(*_args, **_kwargs):
        return OpSpec(kind="no_op", path="app.py", span=(1, 1), props={}), "TIMEOUT = 60\n"

    agent._build_patch = _prompt_only_builder

    result = asyncio.run(agent.run(prompt=PROMPT, dry_run=True, verify_in_background=False))

    assert result.model_output_used is False, (
        "the ablation constant reproduced the model's patch, so the model's output did "
        "not determine it -- model_output_used must be False"
    )
    assert agent.last_policy_source == "heuristic_fallback", (
        "a patch the model did not determine is not the model's; the heuristic gets the "
        f"turn; got {agent.last_policy_source!r}"
    )
    # The shipped patch and its gate are the heuristic's; the model's 0.9 cannot carry it.
    assert result.confidence == pytest.approx(H_CONF), (
        f"the gate must read the confidence of the policy whose output determined the "
        f"patch; the model's 0.9 was about an answer that did not build this patch, got "
        f"{result.confidence!r}"
    )
    assert result.op_kind == "replace_literal", (
        f"the shipped op is the heuristic's, not the model's freeform_patch; got "
        f"{result.op_kind!r}"
    )
    assert result.model_op_kind == "freeform_patch", (
        f"the model's own answer is still recorded separately; got "
        f"{result.model_op_kind!r}"
    )
    assert not result.escalated, "H_CONF is over the gate, so the patch must ship"


def test_policy_source_defaults_to_none(tmp_path: Path) -> None:
    agent = ReflexAgent(root=_repo(tmp_path), effort=LowEffort)

    assert agent.last_policy_source == "none"


# --------------------------------------------------------------------------- #
# Invariant 6: a dead model answer must not vote on the heuristic's patch.
# --------------------------------------------------------------------------- #
#
# The escalation gate decides whether a patch reaches disk.  For a
# ``heuristic_fallback`` patch the model's answer has already proved unusable -- no
# spec could be built from it -- so its confidence carries no information about the
# patch that the heuristic went on to build.  ``conf = max(conf, h_conf * 0.9)` let
# an inflated model head (the shipped iter23 head was anti-correlated at −0.58)
# override the heuristic's own confidence, and a calibrated head inverted the same
# defect: refit conf 0.10–0.53 withheld four line_accurate=True patches that the
# shipped head's 0.69–0.76 had shipped.  Same defect, both directions -- the model's
# confidence is not about the heuristic's patch either way.  See ROADMAP 12.3c.


def test_a_dead_model_answer_must_not_vote_on_the_heuristics_patch(
    tmp_path: Path,
) -> None:
    """The gate reads the confidence of whichever policy built the patch.

    Parametrised over the model's confidence in an answer that builds nothing: at
    0.99 the shipped bug shipped the heuristic's patch; at 0.01 a calibrated head
    withheld it.  Either way the number is not about the heuristic's patch, so the
    escalation state must be identical.
    """
    import asyncio

    H_CONF = 0.5  # under the LowEffort 0.60 gate, so the heuristic alone escalates

    for dead_model_conf in (0.99, 0.01):
        agent = ReflexAgent(
            root=_repo(tmp_path), model=_tiny_model(), tokenizer=_tokenizer(),
            effort=LowEffort,
        )
        # `change_return` cannot be built for this prompt, so the model's answer is
        # provably dead and the heuristic rescues it at H_CONF.
        agent._infer_policy = lambda *a, **k: ("change_return", "", dead_model_conf, None)
        agent._heuristic_policy = lambda *a, **k: ("replace_literal", "60", H_CONF, None)

        result = asyncio.run(
            agent.run(prompt=PROMPT, dry_run=True, verify_in_background=False)
        )

        assert agent.last_policy_source == "heuristic_fallback", (
            f"the setup requires the heuristic to rescue an unbuildable model answer "
            f"(model conf {dead_model_conf})"
        )
        # The gate reads the heuristic only: 0.5 is under the 0.60 threshold, and the
        # model's dead answer -- 0.99 in one loop, 0.01 in the other -- cannot move it.
        assert result.confidence == pytest.approx(H_CONF), (
            f"the gate must read the confidence of the policy that built the patch, not "
            f"max it with the model's dead answer; got {result.confidence!r}"
        )
        assert result.escalated, (
            f"the model answered at {dead_model_conf} and that answer built nothing, so "
            f"its confidence must not carry the heuristic's patch over the gate"
        )
        assert f"{dead_model_conf:.2f}" not in result.escalation_reason, (
            f"the escalation reason must quote the confidence of the policy that built "
            f"the patch, not the model's dead answer; got {result.escalation_reason!r}"
        )


def test_policy_source_is_heuristic_when_no_model_is_configured(tmp_path: Path) -> None:
    """The attribution must not credit the model for the heuristic's work.

    `_infer_policy` falls back to the heuristic *internally* when no model is
    configured, so its return value cannot say which policy answered.  Reading the
    return value and calling any buildable patch "model" is how a heuristic-only run
    gets reported as a model result -- the exact misattribution this field exists to
    prevent, and a bug the first version of this code had.
    """
    import asyncio

    agent = ReflexAgent(root=_repo(tmp_path), effort=LowEffort)
    result = asyncio.run(agent.run(prompt=PROMPT, dry_run=True, verify_in_background=False))

    assert result.immediate_patch_ok is True, "the heuristic should build a patch here"
    assert agent.last_policy_source == "heuristic", (
        f"no model was configured, so the patch is the heuristic's; got "
        f"{agent.last_policy_source!r}"
    )


# --------------------------------------------------------------------------- #
# Invariant 6: a swallowed model failure leaves a trace.
# --------------------------------------------------------------------------- #


def test_a_broken_model_path_is_recorded_not_silently_swallowed(tmp_path: Path) -> None:
    model = _tiny_model()
    agent = ReflexAgent(
        root=_repo(tmp_path), model=model, tokenizer=_tokenizer(), effort=LowEffort,
    )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("injected failure inside the neural path")

    # collate is the first thing the neural branch calls; breaking it stands in for any
    # internal failure. The contract is: fall back, but say so.
    import sylphae.agent.reflex as reflex_mod
    original = reflex_mod.collate
    reflex_mod.collate = _boom
    try:
        op_kind, _p, _c, _m = agent._infer_policy(PROMPT, _candidate(), [_candidate()])
    finally:
        reflex_mod.collate = original

    assert agent.last_model_error is not None, (
        "the neural path failed and the agent fell back to the heuristic policy with no "
        "record. 'The model ran and failed' and 'the model was never configured' must not "
        "look identical from the outside."
    )
    assert "injected failure" in agent.last_model_error
    # and it still returned a usable answer rather than propagating
    assert op_kind in OP_KINDS
