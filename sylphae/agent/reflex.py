"""ReflexAgent: The core System-1 development agent.

Coordinates:
Natural Language Prompt
    ↓
Parallel Explorer (concurrent ripgrep / AST / git / lexical search)
    ↓
Mamba-2 SSD Policy Core (+ persistent latent memory)
    ↓
Structured Patch Execution (T=0 optimistic return)
    ↓
Background Async Verification (syntax compile, affected pytest)
    ↓
Latent Memory Update (MemState decay + write-gate)
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from sylphae.agent.effort import EffortConfig, LowEffort
from sylphae.agent.explorer import CandidateResult, ParallelExplorer, extract_keywords
from sylphae.agent.named_values import (
    color_property_names,
    find_assigned_color,
    find_valued_property,
)
from sylphae.agent.qualitative import (
    adjust_tailwind_class,
    darken_color,
    decompose_qualitative_intents,
    lighten_color,
    scale_font_size,
)
from sylphae.agent.spatial import SPATIAL_KEYWORDS, resolve_spatial_anchors
from sylphae.exec.patch import apply_to_tree
from sylphae.exec.verify import VerifyReport, affected_tests, verify
from sylphae.model.collate import collate
from sylphae.model.policy import SylphaeModel, MemState
from sylphae.model.tokenizer import CodeTokenizer
from sylphae.ops.polyglot import (
    add_css_properties,
    replace_css_property,
)
from sylphae.ops.scope import entity_end_line, entity_start_line, same_entity, symbols_in
from sylphae.ops.spec import OpSpec
from sylphae.ops.vocab import OP_KINDS
from sylphae.repo.index import REPO_INDEX_FILE_NAME, RepoIndex, iter_python_files
from sylphae.vision.defect_mapper import map_vision_defect_to_patch
from sylphae.vision.inspector import VisionReport, inspect_ui_image


#: Confidence ceiling applied when the target came from the explorer's no-match
#: fallback.  Deliberately far below every effort policy's ``min_confidence`` (0.60 low,
#: 0.80 high): the request can still describe a real edit, but nothing has established
#: *which file* it applies to, and an escalated result is the honest report of that.
#: It is a ceiling rather than a fixed value so a weaker pattern match still reads as
#: weaker than a stronger one.
_FALLBACK_CONFIDENCE: float = 0.25

#: The constant the model's ``(op_kind, param_val)`` is replaced with to test whether the
#: model contributed anything to the patch.  ``freeform_patch`` with an empty parameter
#: is the choice that reaches the *fewest* rule branches -- the numeric substitution at
#: the end of ``_build_patch`` requires a non-empty ``param_val`` -- so a spec that is
#: unchanged under this substitution was not built from the model's output.
_ABLATION_OP: str = "freeform_patch"
_ABLATION_PARAM: str = ""


def _same_spec(a: OpSpec | None, b: OpSpec | None) -> bool:
    """Whether two specs would perform the same edit."""
    if a is None or b is None:
        return a is None and b is None
    return (a.kind, a.path, tuple(a.span), dict(a.props)) == (
        b.kind, b.path, tuple(b.span), dict(b.props)
    )

#: A CSS declaration whose value is a colour.  ``border`` is included so ``border: 1px
#: solid red`` is reachable; a request that names a colour without naming a region will
#: take the first match in the anchor's block.
_CSS_COLOR_DECL = re.compile(
    r"(\b(?:[a-z-]*color|background|border)\b\s*:\s*)([^;]+)(;?)", re.I
)


@dataclass
class ReflexResult:
    prompt: str
    target_path: str
    op_kind: str
    confidence: float
    immediate_patch_ok: bool
    new_source: str | None
    spec: OpSpec | None
    candidates: list[CandidateResult]
    escalated: bool
    escalation_reason: str | None = None
    #: True when the exploration budget expired before the tools answered, so
    #: ``candidates`` is a prefix of an answer rather than an answer.  Without this,
    #: a run that gave up waiting is indistinguishable from a run that searched and
    #: found nothing -- and the two deserve different responses from a caller.
    exploration_truncated: bool = False
    #: Tools still running when the budget expired.
    pending_tools: list[str] = field(default_factory=list)
    #: Whether the model's ``(op_kind, param_val)`` actually reached the patch.
    #:
    #: ``None`` when no model was involved.  ``False`` means a patch was produced on the
    #: model's route but the model's own output did not determine it -- which is what
    #: happened on all 14 real-world scenarios: MEASURED, replacing the model's output
    #: with a constant changed the produced spec on 0 of 14.  ``policy_source ==
    #: "model"`` says *which branch ran*, not *who decided*.
    model_output_used: bool | None = None
    #: What the model actually emitted, before anything else got a turn.
    #:
    #: ``model_output_used`` says *whether* the model's answer reached the patch; it does
    #: not say what the answer was, so "the model's output did not decide the patch on 0
    #: of 14" had no route to a fix: it could not distinguish "the model chose an op that
    #: blocks every builder" from "the model chose the right op and an empty value".  Both
    #: show up as the same ``False``.  These are ``None`` when no model was consulted or
    #: the call raised -- which is a different state from emitting an empty string.
    model_op_kind: str | None = None
    model_param_val: str | None = None
    model_confidence: float | None = None
    verification_report: VerifyReport | None = None
    vision_report: VisionReport | None = None
    specs: list[OpSpec] = field(default_factory=list)
    #: Phase timings in milliseconds; ``None`` means the phase did not run.
    #:
    #: These exist because end-to-end latency is not reflex latency, and the two were
    #: being reported as one number.  MEASURED on ``work/repos/tiangolo__typer`` (812
    #: files): ``run()`` took 12.0 s, of which exploration was 0.4 s, policy inference
    #: plus patch construction plus the disk write together were ~1 ms, and *running the
    #: affected test suite* was 9.8 s.  So "Mean Response Latency: 2617 ms" described a
    #: pytest invocation with a reflex attached, not the agent.
    #:
    #: ``verify_ms`` is ``None`` when verification was not requested, when no patch was
    #: written, or when it was handed to a background task -- in the last case the work
    #: is still happening, so reporting a duration would be reporting a partial one.
    explore_ms: float | None = None
    patch_ms: float | None = None
    verify_ms: float | None = None

    @property
    def reflex_ms(self) -> float | None:
        """Time spent deciding and editing, excluding verification.

        This is the number the sub-150 ms design goal is about.  ``None`` when the
        phases were not recorded at all, so an unmeasured run cannot be read as a fast
        one.
        """
        if self.explore_ms is None or self.patch_ms is None:
            return None
        return self.explore_ms + self.patch_ms

    def to_dict(self) -> dict[str, Any]:
        """Serialise this result for ``--json`` consumers.

        ``new_source`` is reduced to its length on purpose: it can be an entire file,
        and a caller that wants the edit has ``spec`` (span + props) plus
        ``target_path``.  Reporting the length keeps the field from being invisible,
        and ``spec is None`` remains the signal that no edit was produced at all.
        """
        return {
            "prompt": self.prompt,
            "target_path": self.target_path,
            "op_kind": self.op_kind,
            "confidence": round(self.confidence, 4),
            "immediate_patch_ok": self.immediate_patch_ok,
            "new_source_length": (
                len(self.new_source) if self.new_source is not None else None
            ),
            "spec": self.spec.to_dict() if self.spec is not None else None,
            "specs": [s.to_dict() for s in self.specs],
            "candidates": [_jsonable(c) for c in self.candidates],
            "escalated": self.escalated,
            "escalation_reason": self.escalation_reason,
            "exploration_truncated": self.exploration_truncated,
            "pending_tools": list(self.pending_tools),
            "verification": _jsonable(self.verification_report),
            "vision": _jsonable(self.vision_report),
            "explore_ms": self.explore_ms,
            "patch_ms": self.patch_ms,
            "verify_ms": self.verify_ms,
            "reflex_ms": self.reflex_ms,
            "model_output_used": self.model_output_used,
        }


def _jsonable(value: Any) -> Any:
    """Recursively convert ``value`` into something ``json.dumps`` accepts.

    ``dataclasses.asdict`` is not enough on its own: ``CandidateResult`` carries
    ``contributing_tools`` as a ``set``, which raises ``TypeError: Object of type set
    is not JSON serializable`` at dump time — after the work is done, turning a
    successful run into a traceback and an empty stdout.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (set, frozenset)):
        # Sorted so the output is stable across runs and callers can diff it.
        return sorted(value, key=str)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    from dataclasses import asdict, is_dataclass

    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    # Last resort so one unrecognised field cannot make the whole result
    # unserialisable; the text form is still more useful to a caller than a crash.
    return str(value)


class ReflexAgent:
    """A resident, millisecond-latency System-1 developer agent."""

    def __init__(
        self,
        root: Path,
        model: SylphaeModel | None = None,
        tokenizer: CodeTokenizer | None = None,
        repo_index: RepoIndex | None = None,
        effort: EffortConfig = LowEffort,
        device: str = "cpu",
    ):
        self.root = Path(root).resolve()
        self.model = model
        # ``_infer_policy`` gates the neural path on both of these, so a model without
        # a tokenizer is a *construction* mistake and is rejected here rather than
        # surfacing as an ``AttributeError`` mid-inference.  This assignment was
        # missing: the parameter was accepted and dropped, which made the model path
        # unreachable -- every caller that passed a model crashed on `self.tokenizer`
        # before the model ran, and because no test ever passed a model, nothing said
        # so.  A run that silently degrades to the heuristic policy while reporting
        # success is the exact failure this guard exists to prevent.
        if model is not None and tokenizer is None:
            raise ValueError(
                "ReflexAgent was given a model but no tokenizer. The policy head cannot "
                "decode a parameter without one, and falling back to the heuristic policy "
                "silently would attribute the model's absence to the request. Pass "
                "tokenizer=<CodeTokenizer>, or omit model= to select the heuristic policy "
                "deliberately."
            )
        self.tokenizer = tokenizer
        #: Why the neural path was abandoned on the most recent inference, or ``None``.
        #: The policy branch catches broadly so one bad batch cannot take down the
        #: agent, but a swallowed exception that leaves no trace is indistinguishable
        #: from a model that was never configured. This records it for callers.
        self.last_model_error: str | None = None
        #: Which policy produced the most recent patch -- ``"model"``,
        #: ``"heuristic_fallback"`` (the model produced nothing buildable),
        #: ``"vision"``, or ``"none"``.  Set by :meth:`run`.
        self.last_policy_source: str = "none"
        #: Whether the model's own output determined the most recent patch, measured by
        #: ablation rather than inferred from the route.  ``None`` when no model was
        #: involved.  See :attr:`ReflexResult.model_output_used`.
        self.last_model_output_used: bool | None = None
        #: The model's raw answer on the most recent inference: what it chose before any
        #: other builder got a turn.  ``None`` means the model was not consulted or the
        #: call raised; ``""`` means it was consulted and emitted nothing.  Recorded
        #: because ``last_model_output_used`` alone cannot say *why* the answer did not
        #: reach the patch.
        self.last_model_op_kind: str | None = None
        self.last_model_param_val: str | None = None
        self.last_model_confidence: float | None = None
        #: True when the most recent exploration ran out of budget before its tools
        #: answered.  Read after :meth:`run` returns.
        self.last_exploration_truncated: bool = False
        #: Tools still running when that budget expired.
        self.last_pending_tools: list[str] = []
        if repo_index is not None:
            self.repo_index = repo_index
        else:
            cache_file = self.root / REPO_INDEX_FILE_NAME
            idx = None
            if cache_file.exists():
                try:
                    idx = RepoIndex.load(cache_file)
                except Exception:
                    idx = None
            if idx is None:
                idx = RepoIndex.build(self.root)
                try:
                    idx.save(cache_file)
                except Exception:
                    pass
            self.repo_index = idx
        self.effort = effort
        self.device = torch.device(device)
        self.explorer = ParallelExplorer(self.root, self.repo_index, effort=self.effort)

        # Persistent latent memory across requests (animal-brain experience bias)
        self.latent_memory: MemState | None = None
        if self.model is not None:
            self.model.eval()
            self.model.to(self.device)

    async def run(
        self,
        prompt: str,
        image_path: Path | str | None = None,
        dry_run: bool = True,
        on_intermediate_candidate: Callable[[list[CandidateResult]], None] | None = None,
        verify_in_background: bool = True,
    ) -> ReflexResult:
        """Run the reflex loop: parallel search -> policy inference -> optimistic patch -> async verify."""

        # Phase timings, so the caller can tell the reflex from the verification.  A
        # single end-to-end number cannot: MEASURED on ``work/repos/tiangolo__typer``
        # it was 12.0 s, of which 9.8 s was the affected test suite.
        run_t0 = time.perf_counter()
        # Reset per call: a previous request's answer must not be read as this one's.
        self.last_model_output_used = None
        self.last_model_op_kind = None
        self.last_model_param_val = None
        self.last_model_confidence = None

        # 0. Multimodal vision inspection
        vision_rep = None
        extra_kws: list[str] = []
        if image_path:
            vision_rep = inspect_ui_image(image_path, prompt_hint=prompt)
            extra_kws.extend(vision_rep.extracted_keywords)

        # 1. Parallel exploration (asynchronous stream collection with early locking)
        candidates = await self.explorer.explore(
            prompt=prompt,
            additional_keywords=extra_kws,
            early_lock_threshold=0.90,
            on_candidate_update=on_intermediate_candidate,
        )
        explore_ms = (time.perf_counter() - run_t0) * 1000.0

        # Recorded before anything reads `candidates`: the truncation flag is what
        # tells "searched and found nothing" apart from "stopped waiting".
        self.last_exploration_truncated = bool(
            getattr(self.explorer, "last_exploration_truncated", False)
        )
        self.last_pending_tools = list(
            getattr(self.explorer, "last_pending_tools", [])
        )

        if not candidates:
            return ReflexResult(
                prompt=prompt,
                target_path="",
                op_kind="no_op",
                confidence=0.0,
                immediate_patch_ok=False,
                new_source=None,
                spec=None,
                candidates=[],
                escalated=True,
                escalation_reason=self._escalation_reason_for_truncation(
                    "No candidate files found for request"
                ),
                exploration_truncated=self.last_exploration_truncated,
                pending_tools=list(self.last_pending_tools),
                vision_report=vision_rep,
                # No patch phase was reached, so it contributes 0 ms rather than
                # ``None``: the run did complete a reflex, it just declined to edit.
                explore_ms=explore_ms,
                patch_ms=0.0,
            )

        target_cand = candidates[0]
        target_path = target_cand.path

        # Check multimodal defect patch first if image defect exists
        spec = None
        new_source = None
        conf = 0.85
        op_kind = "replace_literal"
        #: Which policy produced the patch that is about to be returned.  Recorded so a
        #: caller can tell a model answer from a heuristic rescue; see
        #: ``last_policy_source``.
        policy_source = "none"

        if vision_rep and vision_rep.has_defect:
            full_path = self.root / target_path
            if full_path.exists():
                with open(full_path, "r", encoding="utf-8", newline="", errors="replace") as f:
                    old_src = f.read()
                new_source, spec = map_vision_defect_to_patch(vision_rep, target_path, old_src)
                if spec:
                    op_kind = spec.kind
                    conf = vision_rep.confidence
                    policy_source = "vision"

        if spec is None:
            # Reset before inference: a stale error from a previous request must not be
            # read as this request's outcome.
            self.last_model_error = None
            # 2. Policy inference
            op_kind, param_val, conf, next_mem = self._infer_policy(prompt, target_cand, candidates)

            # `_infer_policy` falls back to the heuristic *internally* when no model is
            # configured, so its return value alone cannot say which policy answered.
            # The first version of this attribution read the return value and labelled
            # every buildable patch "model" -- including heuristic-only runs, which is
            # precisely the misattribution this field exists to prevent. The decision is
            # therefore made from observable state instead.
            neural_available = self.model is not None and self.tokenizer is not None
            neural_answered = neural_available and self.last_model_error is None
            if neural_answered:
                # Recorded before `_build_patch`, so it is the model's own answer and not
                # what the builders turned it into.  Guarded on `neural_answered` rather
                # than on `last_model_error` alone: with no model configured the error
                # field is never set either, and an unguarded read would file the
                # heuristic's answer under the model.
                self.last_model_op_kind = op_kind
                self.last_model_param_val = param_val
                self.last_model_confidence = conf

            # 3. Structured Patch Construction
            spec, new_source = self._build_patch(prompt, target_path, op_kind, param_val, cand=target_cand)
            if spec is not None:
                policy_source = "model" if neural_answered else "heuristic"
                if neural_answered:
                    # Ablation: rebuild the same patch with the model's output replaced by
                    # a constant.  If the spec is unchanged, the model's op/param did not
                    # reach the patch -- and `policy_source == "model"` is then a claim
                    # about the *route*, not about the content.
                    #
                    # MEASURED on the 14 real-world scenarios with ``runs/iter7``: the
                    # model's op/param changed the produced spec on **0 of 14**.  Six of
                    # them are built by ``decompose_qualitative_intents`` /
                    # ``_build_named_value_patch``, which read the *prompt* and never look
                    # at ``op_kind`` or ``param_val`` at all -- there is no gate for the
                    # model to open, so no op head can reach those six and the ablation
                    # cannot detect a contribution there however good the head gets.
                    # The other eight are blocked *by* the model's op choice -- feeding
                    # ``replace_literal`` with any non-empty parameter builds a correct
                    # patch on the correct line (``DefaultReadTimeout = ...``).
                    #
                    # So the ceiling of ``model_output_used`` on this benchmark is 6 of
                    # 14, and it was measured rather than argued: an oracle op head
                    # injected into the real benchmark (``scratch/bench_oracle_op.py``)
                    # reached exactly ``model output decided it: 6 of 14`` with
                    # ``patch applied: 14/14``.  ``tests/test_model_reach.py`` freezes the
                    # ceiling by asserting the six are invariant to every
                    # ``(op_kind, param_val)`` pair, with a numeric scenario as falsifier.
                    ablated, _ = self._build_patch(
                        prompt, target_path, _ABLATION_OP, _ABLATION_PARAM, cand=target_cand
                    )
                    model_output_used = not _same_spec(ablated, spec)
                    self.last_model_output_used = model_output_used

                    # Blocker 2, the precedence rule (ROADMAP 12.4): a patch is the
                    # model's only if the model's own output determined it.  When
                    # ``model_output_used`` is False the patch came from a builder that
                    # read the *prompt* -- ``decompose_qualitative_intents`` and
                    # ``_build_named_value_patch`` never look at ``op_kind`` or
                    # ``param_val`` -- so ``policy_source == "model"`` describes the
                    # route, not the decision.  Two consequences follow, and both are
                    # the same defect as 12.3c seen from the other side:
                    #
                    # * the model's confidence is not about this patch either way.  It
                    #   was 0.01-0.14 on all six qualitative scenarios, which withheld
                    #   patches the heuristic had been shipping at 0.656-0.88 -- 0/6
                    #   passed as "model" against 4/6 as "heuristic_fallback".
                    # * a buildable patch suppresses the safety net entirely.  The
                    #   fallback below only fires on ``spec is None``, so a patch built
                    #   without the model leaves the heuristic no turn at all.
                    #
                    # The rule: when the model's output did not determine the patch, the
                    # heuristic gets a turn, and its own patch and confidence are what
                    # ship.  This is also what makes the first working parameter head
                    # safe rather than merely loud -- once ``param_target_string`` asks
                    # for the value, a head that emits a real value will build a
                    # buildable patch, and without this rule it ships unopposed.
                    if not model_output_used:
                        h_op, h_param, h_conf, _ = self._heuristic_policy(prompt, target_cand)
                        h_spec, h_src = self._build_patch(
                            prompt, target_path, h_op, h_param, cand=target_cand
                        )
                        if h_spec is not None:
                            spec, new_source = h_spec, h_src
                            op_kind, param_val = h_op, h_param
                            conf = h_conf
                            policy_source = "heuristic_fallback"
            else:
                # The policy named an operation but no patch could be built from it.  The
                # heuristic gets a turn, and it is recorded separately, because "the model
                # answered" and "the model failed and something else answered" are
                # different results -- reporting the second as the first is how a
                # fallback silently becomes a capability claim.
                h_op, h_param, h_conf, _ = self._heuristic_policy(prompt, target_cand)
                spec, new_source = self._build_patch(prompt, target_path, h_op, h_param, cand=target_cand)
                if spec is not None:
                    op_kind, param_val = h_op, h_param
                    # The escalation gate must read the confidence of whichever policy
                    # built this patch.  ``conf`` here is the *model's* confidence in an
                    # answer that just proved unusable -- it produced no spec at all --
                    # so it carries no information about the heuristic's patch, and
                    # maxing it in let an inflated model head ship (or, with a calibrated
                    # head, withhold) the heuristic's work for reasons the model never
                    # earned.  MEASURED: the four heuristic_fallback scenarios that moved
                    # between the shipped and refit conf heads were all line_accurate=True
                    # under the shipped head, and the refit head's lower conf is the only
                    # thing that changed -- see ROADMAP 12.3c.
                    #
                    # The ``* 0.9`` discount that accompanied the ``max`` is removed for
                    # the same reason: it was a tax on the heuristic's confidence to keep
                    # it below the model's, and once the two are no longer competing it is
                    # an arbitrary number that happened to land exactly on the gate
                    # boundary (0.656 * 0.9 = 0.59 against a 0.60 threshold withheld four
                    # correct patches).  The gate now reads the heuristic's own number.
                    conf = h_conf
                    policy_source = "heuristic_fallback"
        else:
            next_mem = None

        # A candidate from the no-match fallback was not located by evidence: it is the
        # file that sorts first, picked because nothing matched at all.  ``conf`` above
        # measures how well the *request* fits a known pattern, which says nothing about
        # whether the file is the right one -- so on a fallback the chain is unsupported
        # however clean the wording looked.  MEASURED: "把导航栏高度缩短一些" run against
        # this repository returned ``sylphae/data/reflex_synth.py`` at confidence 0.88, a
        # file with no relation to the request, because the wording matched the
        # "缩短" pattern and nobody asked whether the target had been found.
        if target_cand.from_fallback:
            conf = min(conf, _FALLBACK_CONFIDENCE)

        # Check confidence escalation
        escalated = conf < self.effort.min_confidence
        escalation_reason = f"Confidence {conf:.2f} < threshold {self.effort.min_confidence:.2f}" if escalated else None

        apply_ok = False
        if spec is not None and new_source is not None:
            # Escalation has to be a state with consequences, not a field in the report.
            # It means "this needs a human", and a human cannot un-edit a file.  Writing
            # the patch anyway made the flag advisory: the run reported doubt and edited
            # the user's code in the same breath.  The patch is still built and shown --
            # it is the disk write that waits for confirmation.
            if escalated and not dry_run:
                escalation_reason = (
                    f"{escalation_reason}. Patch withheld from disk: "
                    f"{spec.kind} on {target_path!r} was not applied"
                )
            else:
                # Execute patch optimistically
                apply_res = apply_to_tree(self.root, target_path, spec, dry_run=dry_run)
                apply_ok = apply_res.ok
                if not apply_ok:
                    escalated = True
                    escalation_reason = f"Patch apply failed: {apply_res.error}"
        else:
            # The policy named an operation but no patch could be built from it.
            # ``conf`` above measures how well the *request* matched a known pattern,
            # not the odds that the edit is right, so it cannot stand in for "a usable
            # patch exists".  Leaving it untouched here is how a run that produced no
            # diff at all still reported 0.65 confidence and a clean exit.
            escalated = True
            escalation_reason = (
                f"No patch could be constructed for the inferred action "
                f"{op_kind!r} on {target_path!r}"
            )

        # Which policy actually produced `spec`: "vision", "model", "heuristic_fallback",
        # or "none".  Read after `run()` returns.
        self.last_policy_source = policy_source

        if escalated:
            escalation_reason = self._escalation_reason_for_truncation(
                escalation_reason or "escalated"
            )

        # Everything between the end of exploration and the start of verification:
        # policy inference, patch construction, and the optimistic disk write.
        patch_ms = (time.perf_counter() - run_t0) * 1000.0 - explore_ms

        res = ReflexResult(
            prompt=prompt,
            target_path=target_path,
            op_kind=op_kind,
            confidence=conf,
            immediate_patch_ok=apply_ok,
            new_source=new_source,
            spec=spec,
            candidates=candidates,
            escalated=escalated,
            escalation_reason=escalation_reason,
            exploration_truncated=self.last_exploration_truncated,
            pending_tools=list(self.last_pending_tools),
            vision_report=vision_rep,
            explore_ms=explore_ms,
            patch_ms=patch_ms,
            model_output_used=self.last_model_output_used,
            model_op_kind=self.last_model_op_kind,
            model_param_val=self.last_model_param_val,
            model_confidence=self.last_model_confidence,
        )

        # 4. Background Async Verification & Memory Update
        if apply_ok and not dry_run:
            if verify_in_background:
                # Left as ``None``: the work is still in flight, so any duration here
                # would be a partial one, and a partial verification cost reported as
                # the verification cost is the same class of error as a missing
                # measurement reported as a pass.
                asyncio.create_task(self._async_verify_and_update(target_path, next_mem))
            else:
                verify_t0 = time.perf_counter()
                rep = await self._async_verify_and_update(target_path, next_mem)
                res.verify_ms = (time.perf_counter() - verify_t0) * 1000.0
                res.verification_report = rep

        return res

    def _escalation_reason_for_truncation(self, reason: str) -> str:
        """Name the exploration budget when it, and not the policy, is the cause.

        MEASURED: on ``work/repos/tiangolo__typer`` (812 files) the reason reported for
        a failed run was ``Confidence 0.25 < threshold 0.60``.  That is true and
        useless: it describes the symptom.  The cause was that the 0.5 s exploration
        budget expired with ripgrep and path_lexical still running, so nothing was ever
        located, the no-match fallback supplied a file, and the confidence was capped
        by that fallback.  A reason that points at the wrong subsystem sends the next
        reader to tune a threshold that was never the problem.
        """
        if not self.last_exploration_truncated:
            return reason
        pending = ", ".join(self.last_pending_tools) or "unknown"
        return (
            f"Exploration budget ({self.effort.timeout_seconds}s) expired with tool(s) "
            f"still running ({pending}); no target was located. {reason}"
        )

    def _infer_policy(
        self,
        prompt: str,
        target_cand: CandidateResult,
        candidates: list[CandidateResult],
    ) -> tuple[str, str, float, MemState | None]:
        """Infer target, op kind, parameter, confidence and memory update."""
        if self.model is not None and self.tokenizer is not None:
            try:
                # Read candidate text & source
                target_file = self.root / target_cand.path
                old_src = target_file.read_text(encoding="utf-8", errors="replace") if target_file.exists() else ""

                cand_line = target_cand.line or 1
                dummy_ex = {
                    "request": prompt,
                    "target_path": target_cand.path,
                    "old_source": old_src,
                    "new_source": old_src,
                    "repo_files": [c.path for c in candidates],
                    "op": {"kind": "replace_literal", "span": [cand_line, cand_line], "props": {}},
                }

                def cand_provider(ex, k):
                    return candidates[:k]

                batch = collate([dummy_ex], self.tokenizer, cand_provider, self.model.cfg).to(self.device)

                with torch.no_grad():
                    out, next_mem = self.model(batch, mem=self.latent_memory)

                op_id = int(out.op_logits[0].argmax(dim=-1).item())
                op_kind = OP_KINDS[op_id] if 0 <= op_id < len(OP_KINDS) else "replace_literal"

                # Generate parameter (clean byte decode up to eos)
                param_ids = out.param_logits[0].argmax(dim=-1).tolist()
                clean_ids = []
                for t in param_ids:
                    if int(t) == self.tokenizer.eos_id:
                        break
                    if int(t) >= 7:
                        clean_ids.append(int(t))
                param_val = self.tokenizer.decode(clean_ids).strip()
                conf = float(out.confidence[0].item())

                return op_kind, param_val, conf, next_mem
            except Exception as exc:
                # Broad by design: a malformed batch must not take down a resident
                # agent. But it is recorded, because "the model ran and failed" and
                # "the model was never configured" produce identical output otherwise.
                self.last_model_error = f"{type(exc).__name__}: {exc}"
                pass

        # Heuristic / rule-based fallback when model is not active
        return self._heuristic_policy(prompt, target_cand)

    def _heuristic_policy(
        self,
        prompt: str,
        cand: CandidateResult,
    ) -> tuple[str, str, float, None]:
        """Heuristic analysis for common developer prompts (size, timeout, rename, etc.).

        The confidence returned here describes how well the *request* matches a known
        pattern.  It is deliberately not a probability that the resulting edit is
        correct: the pattern can match perfectly while no patch is constructible at
        all, because the located line may not even hold the value being changed.
        Callers therefore treat "no spec was built" as escalation regardless of this
        number — see the ``spec is None`` branch in :meth:`run`.
        """
        nums = re.findall(r"\d+(?:\.\d+)?", prompt)

        # "把导航栏的高度缩短一些" / "缩短一半" / "height 32"
        if any(w in prompt for w in ("高度", "height", "缩短", "尺寸", "变小", "半")):
            if nums and not ("一半" in prompt or "0.5" in prompt):
                return "replace_literal", nums[-1], 0.90, None
            if "一半" in prompt or "0.5" in prompt:
                return "replace_literal", "0.5x", 0.92, None
            if "缩短" in prompt or "变小" in prompt:
                return "replace_literal", "0.5x", 0.88, None
            return "replace_literal", "32", 0.85, None

        if any(w in prompt for w in ("超时", "timeout", "秒")):
            val = nums[-1] if nums else "60"
            return "replace_literal", val, 0.90, None

        if "重命名" in prompt or "rename" in prompt:
            words = prompt.split()
            val = words[-1] if words else "new_name"
            return "rename_symbol", val, 0.88, None

        if "guard" in prompt or "判空" in prompt or "检查" in prompt:
            return "add_guard", "if not value: return None", 0.85, None

        return "replace_literal", (nums[-1] if nums else ""), 0.65, None

    def _anchor_context(
        self,
        lines: list[str],
        cand: CandidateResult | None,
    ) -> tuple[int | None, set[str]]:
        """The line the explorer points at, plus the identifiers that line carries.

        ``None`` means there is no positional evidence at all, which is a different
        state from "line 1".  Callers seeing ``None`` must not impose a placement
        constraint, because there is nothing to constrain the patch against.
        """
        anchor = (
            cand.line
            if (cand and cand.line and 1 <= cand.line <= len(lines))
            else None
        )
        if anchor is None:
            return None, set()
        return anchor, symbols_in(f"{cand.snippet or ''} {lines[anchor - 1]}")

    @staticmethod
    def _placement_allowed(
        lines: list[str],
        anchor: int | None,
        anchor_symbols: set[str],
        line_no: int,
        text: str,
    ) -> bool:
        """Whether a patch may land on ``line_no``.

        Keyword matching reaches every line in the file, so a miss on the anchor line
        used to widen the search to whichever unrelated line matched next.  That is how
        a request aimed at one entity silently edited another: the file still parsed, so
        nothing objected.

        A patch outside the anchor's syntactic entity now needs its own evidence -- it
        must name a symbol the anchor names too.  Syntactic agreement or symbolic
        agreement, either will do; neither, and the line is skipped so the caller can
        escalate.  Refusing is cheaper than editing the wrong thing.  See
        :mod:`sylphae.ops.scope`.
        """
        if anchor is None or same_entity(lines, anchor, line_no):
            return True
        if anchor_symbols and not (anchor_symbols & symbols_in(text)):
            return False
        return True

    @staticmethod
    def _block_indent(lines: list[str], start: int, end: int) -> str:
        """The indentation a line added inside a brace block should use."""
        for i in range(start + 1, end):
            text = lines[i - 1]
            if text.strip():
                m = re.match(r"^[ \t]+", text)
                return m.group(0) if m else "  "
        return "  "

    @staticmethod
    def _anchor_from_prompt(
        lines: list[str],
        prompt: str,
        anchor: int | None,
        anchor_symbols: set[str],
    ) -> tuple[int | None, set[str]]:
        """Prefer the line the request names, when the request names one.

        A candidate from a path match carries no line, and then the request's own
        identifiers are the only positional evidence there is.  Resolving them to a line
        makes the block they name the constraint, which matters because a file holding
        several rules offers several places to put an edit: "给 badge 加一个圆角" names
        the badge rule, and without this the first rule in the file -- the navbar --
        received it.

        Only ASCII identifiers are considered, so a Chinese request contributes the
        selector names it mentions and nothing else.
        """
        wanted = symbols_in(prompt)
        if not wanted:
            return anchor, anchor_symbols
        for i in range(1, len(lines) + 1):
            named = wanted & symbols_in(lines[i - 1])
            if named:
                return i, named
        return anchor, anchor_symbols

    def _build_named_value_patch(
        self,
        prompt: str,
        target_path: str,
        lines: list[str],
        lines_raw: list[str],
        old_src: str,
        cand: CandidateResult | None,
    ) -> tuple[OpSpec | None, str | None]:
        """Build a patch for a value the request names outright.

        Two shapes, and both are lookups rather than inferences -- which is exactly why
        the qualitative branch could not serve either: a colour named after an
        assignment cue ("改成深蓝色"), and a property given with its value ("圆角 8px").
        MEASURED before this existed: "把 navbar 背景改成深蓝色" located ``theme.css``
        correctly, named no number for the numeric branch to substitute, and produced no
        patch at all.  File and action were both right; the request still could not be
        carried out.
        """
        anchor, anchor_symbols = self._anchor_context(lines, cand)
        anchor, anchor_symbols = self._anchor_from_prompt(
            lines, prompt, anchor, anchor_symbols
        )

        def emit(index: int, new_line: str) -> tuple[OpSpec, str]:
            raw_line = lines_raw[index - 1]
            ending = "\r\n" if raw_line.endswith("\r\n") else ("\n" if raw_line.endswith("\n") else "")
            spec = OpSpec(
                kind="replace_literal",
                path=target_path,
                span=(index, index),
                props={
                    "old_text": raw_line,
                    "new_text": new_line + ending,
                    "old": lines[index - 1].strip(),
                    "new": new_line.strip(),
                },
            )
            new_lines = list(lines)
            new_lines[index - 1] = new_line
            return spec, "\n".join(new_lines) + ("\n" if old_src.endswith("\n") else "")

        color = find_assigned_color(prompt)
        if color is not None:
            wanted = color_property_names(prompt)
            for i in range(1, len(lines) + 1):
                line = lines[i - 1]
                if not self._placement_allowed(lines, anchor, anchor_symbols, i, line):
                    continue
                m = _CSS_COLOR_DECL.search(line)
                if not m:
                    continue
                if wanted:
                    prop_name = m.group(1).split(":")[0].strip().lower()
                    if prop_name not in wanted:
                        continue
                return emit(i, line[: m.start(2)] + color + line[m.end(2) :])
            return None, None

        valued = find_valued_property(prompt)
        if valued is None:
            return None, None
        prop, value = valued

        # Already declared inside the anchor's block: this is a change to it.
        for i in range(1, len(lines) + 1):
            line = lines[i - 1]
            if not self._placement_allowed(lines, anchor, anchor_symbols, i, line):
                continue
            m = re.search(rf"(\b{re.escape(prop)}\s*:\s*)([^;]+)(;?)", line, re.I)
            if m:
                return emit(i, line[: m.start(2)] + value + line[m.end(2) :])

        # Not declared: add it inside the anchor's block, which needs the block's end.
        if anchor is None:
            return None, None
        end = entity_end_line(lines, anchor)
        closing = lines[end - 1]
        if "}" not in closing:
            return None, None
        indent = self._block_indent(lines, entity_start_line(lines, anchor), end)
        return emit(end, f"{indent}{prop}: {value};\n" + closing)

    def _build_patch(
        self,
        prompt: str,
        target_path: str,
        op_kind: str,
        param_val: str,
        cand: CandidateResult | None = None,
    ) -> tuple[OpSpec | None, str | None]:
        """Construct structured OpSpec and compute prospective source."""
        full_path = self.root / target_path
        if not full_path.exists():
            return None, None
        try:
            with open(full_path, "r", encoding="utf-8", errors="surrogateescape", newline="") as handle:
                old_src = handle.read()
        except OSError:
            return None, None

        lines = old_src.splitlines()
        lines_raw = old_src.splitlines(keepends=True)

        # Handle height / style / size scaling
        if "0.5x" in param_val or ("缩短" in prompt and ("高度" in prompt or "height" in prompt)):
            # This branch used to scan the whole file and take the first line that
            # looked like a height, with no reference to the anchor at all -- so it was
            # the one remaining path where "where the user is pointing" and "which line
            # gets edited" could still disagree.  The anchor is consulted here like
            # everywhere else.
            anchor, anchor_symbols = self._anchor_context(lines, cand)
            order = ([anchor] if anchor is not None else []) + [
                i for i in range(1, len(lines) + 1) if i != anchor
            ]
            height_re = re.compile(
                r"(\b(?:[A-Za-z_]*height[A-Za-z_]*|h)\b\s*[:=]\s*)(\d+)", re.I
            )
            for i in order:
                line = lines[i - 1]
                m = height_re.search(line)
                if not m:
                    continue
                if not self._placement_allowed(lines, anchor, anchor_symbols, i, line):
                    continue
                orig_val = int(m.group(2))
                new_num = max(1, orig_val // 2)
                new_line = line[: m.start(2)] + str(new_num) + line[m.end(2) :]
                raw_line = lines_raw[i - 1]
                ending = "\r\n" if raw_line.endswith("\r\n") else ("\n" if raw_line.endswith("\n") else "")
                spec = OpSpec(
                    kind="replace_literal",
                    path=target_path,
                    span=(i, i),
                    props={
                        "old_text": raw_line,
                        "new_text": new_line + ending,
                        "old": str(orig_val),
                        "new": str(new_num),
                    },
                )
                new_lines = list(lines)
                new_lines[i - 1] = new_line
                return spec, "\n".join(new_lines) + ("\n" if old_src.endswith("\n") else "")

        # Handle Qualitative CSS & TSX modifications (e.g. "把右上角的按钮颜色加深，字体调大一些")
        qual_intents = decompose_qualitative_intents(prompt)
        ext = target_path.lower()
        is_css = ext.endswith((".css", ".scss", ".less"))
        is_tsx = ext.endswith((".tsx", ".jsx", ".vue", ".html"))

        if qual_intents and (is_css or is_tsx):
            spatial = resolve_spatial_anchors(prompt)
            current_src = old_src
            combined_spec = None

            # Extract best ASCII context hint from search keywords
            hint = None
            for kw in spatial.search_keywords:
                if kw.isascii():
                    hint = kw
                    break
            if not hint:
                hint = "button" if "button" in prompt.lower() or "btn" in prompt.lower() else "btn"

            if is_css:
                for q in qual_intents:
                    if q.category == "color":
                        for prop in ("background-color", "background", "color", "border-color"):
                            m = re.search(rf"\b{prop}\s*:\s*([^;]+);", current_src, re.I)
                            if m:
                                old_c = m.group(1).strip()
                                new_c = darken_color(old_c) if q.action == "darken" else lighten_color(old_c)
                                current_src, combined_spec = replace_css_property(current_src, prop, new_c, context_hint=hint)
                                break
                    elif q.category == "font_size":
                        m = re.search(r"\bfont-size\s*:\s*([^;]+);", current_src, re.I)
                        if m:
                            old_sz = m.group(1).strip()
                            new_sz = scale_font_size(old_sz, 1.2 if q.action == "enlarge" else 0.85)
                            current_src, combined_spec = replace_css_property(current_src, "font-size", new_sz, context_hint=hint)
                    elif q.category == "dimension":
                        for prop in ("height", "min-height", "width", "max-width"):
                            m = re.search(rf"\b{prop}\s*:\s*([^;]+);", current_src, re.I)
                            if m:
                                old_dim = m.group(1).strip()
                                from sylphae.agent.qualitative import scale_dimension
                                new_dim = scale_dimension(old_dim, 1.2 if q.action == "enlarge" else 0.8)
                                current_src, combined_spec = replace_css_property(current_src, prop, new_dim, context_hint=hint)
                                break
                    elif q.category == "spacing":
                        for prop in ("padding", "margin", "gap"):
                            m = re.search(rf"\b{prop}\s*:\s*([^;]+);", current_src, re.I)
                            if m:
                                old_sp = m.group(1).strip()
                                from sylphae.agent.qualitative import scale_dimension
                                new_sp = scale_dimension(old_sp, 1.25 if q.action == "spacious" else 0.75)
                                current_src, combined_spec = replace_css_property(current_src, prop, new_sp, context_hint=hint)
                                break
                    elif q.category == "border_radius":
                        m = re.search(r"\bborder-radius\s*:\s*([^;]+);", current_src, re.I)
                        if m:
                            old_rad = m.group(1).strip()
                            from sylphae.agent.qualitative import scale_dimension
                            new_rad = scale_dimension(old_rad, 1.5 if q.action == "round" else 0.5)
                            current_src, combined_spec = replace_css_property(current_src, "border-radius", new_rad, context_hint=hint)
                    elif q.category == "text_overflow":
                        current_src, combined_spec = add_css_properties(current_src, hint, {
                            "overflow": "hidden",
                            "text-overflow": "ellipsis",
                            "white-space": "nowrap",
                        })

                if combined_spec:
                    combined_spec.path = target_path
                    return combined_spec, current_src

            elif is_tsx:
                best_line_idx = None
                best_score = -1
                best_transformed_line = None

                comp_keywords: list[str] = []
                if spatial.component_type:
                    comp_keywords.extend(SPATIAL_KEYWORDS.get(spatial.component_type, [spatial.component_type]))
                for kw in spatial.search_keywords:
                    if kw.isascii():
                        comp_keywords.append(kw)
                if not comp_keywords:
                    comp_keywords = ["button", "btn"]

                for line_idx, line in enumerate(lines, 1):
                    line_lower = line.lower()
                    score = 0
                    context_window = " ".join(lines[max(0, line_idx - 6) : line_idx]).lower()
                    if any(kw.lower() in line_lower for kw in comp_keywords) or any(f"<{kw.lower()}" in context_window for kw in comp_keywords):
                        score += 10
                    if any(pos_kw.lower() in line_lower for pos_kw in ("top-right", "header", "nav", "action", "search", "card")):
                        score += 3

                    new_line = line
                    modified_categories = 0
                    for q in qual_intents:
                        if q.category == "color":
                            m_col = re.search(r"\b((?:bg|text|border)-[a-z]+-[0-9]+(?:/[0-9]+)?)\b", new_line)
                            if m_col:
                                old_t = m_col.group(1)
                                new_t = adjust_tailwind_class(old_t, q.action)
                                new_line = new_line.replace(old_t, new_t, 1)
                                modified_categories += 1
                        elif q.category == "font_size":
                            m_sz = re.search(r"\b(text-(?:xs|sm|base|lg|xl|2xl|3xl|4xl))\b", new_line)
                            if m_sz:
                                old_s = m_sz.group(1)
                                new_s = adjust_tailwind_class(old_s, q.action)
                                new_line = new_line.replace(old_s, new_s, 1)
                                modified_categories += 1
                        elif q.category in ("spacing", "dimension"):
                            m_sp = re.search(r"\b((?:p|px|py|gap|m|mx|my|h|w)-[0-9\.]+)\b", new_line)
                            if m_sp:
                                old_sp = m_sp.group(1)
                                new_sp = adjust_tailwind_class(old_sp, q.action)
                                new_line = new_line.replace(old_sp, new_sp, 1)
                                modified_categories += 1
                        elif q.category == "border_radius":
                            m_rad = re.search(r"\b(rounded(?:-[a-z0-9]+)?)\b", new_line)
                            if m_rad:
                                old_r = m_rad.group(1)
                                new_r = adjust_tailwind_class(old_r, q.action)
                                new_line = new_line.replace(old_r, new_r, 1)
                                modified_categories += 1
                        elif q.category == "text_overflow":
                            if "truncate" not in new_line:
                                m_cls = re.search(r'((?:class(?:Name)?)=["\'])([^"\']+)(["\'])', new_line)
                                if m_cls:
                                    new_line = new_line[:m_cls.start(2)] + m_cls.group(2) + " truncate" + new_line[m_cls.end(2):]
                                    modified_categories += 1
                                else:
                                    m_cn = re.search(r'((?:cn|clsx)\s*\([^)]*["\'])([^"\']+)(["\'])', new_line)
                                    if m_cn:
                                        new_line = new_line[:m_cn.start(2)] + m_cn.group(2) + " truncate" + new_line[m_cn.end(2):]
                                        modified_categories += 1

                    if modified_categories > 0:
                        score += modified_categories * 5
                        if score > best_score:
                            best_score = score
                            best_line_idx = line_idx
                            best_transformed_line = new_line

                if best_line_idx is not None and best_transformed_line is not None:
                    line_idx = best_line_idx
                    new_line = best_transformed_line
                    raw_line = lines_raw[line_idx - 1]
                    ending = "\r\n" if raw_line.endswith("\r\n") else ("\n" if raw_line.endswith("\n") else "")
                    spec = OpSpec(
                        kind="replace_literal",
                        path=target_path,
                        span=(line_idx, line_idx),
                        props={
                            "old_text": raw_line,
                            "new_text": new_line + (ending if not new_line.endswith(("\r", "\n")) else ""),
                            "old": lines[line_idx - 1].strip(),
                            "new": new_line.strip(),
                        },
                    )
                    new_lines = list(lines)
                    new_lines[line_idx - 1] = new_line.rstrip("\r\n")
                    return spec, "\n".join(new_lines) + ("\n" if old_src.endswith("\n") else "")

                # Vue SFC fallback: if no Tailwind classes matched in template, check <style> block
                if ext.endswith((".vue", ".html")) and "<style" in old_src:
                    m_style = re.search(r"(<style[^>]*>)(.*?)(</style>)", old_src, re.DOTALL | re.I)
                    if m_style:
                        css_content = m_style.group(2)
                        cur_css = css_content
                        css_spec = None
                        for q in qual_intents:
                            if q.category == "color":
                                for prop in ("background-color", "background", "color", "border-color"):
                                    m = re.search(rf"\b{prop}\s*:\s*([^;]+);", cur_css, re.I)
                                    if m:
                                        old_c = m.group(1).strip()
                                        new_c = darken_color(old_c) if q.action == "darken" else lighten_color(old_c)
                                        cur_css, css_spec = replace_css_property(cur_css, prop, new_c, context_hint=hint)
                                        break
                            elif q.category == "font_size":
                                m = re.search(r"\bfont-size\s*:\s*([^;]+);", cur_css, re.I)
                                if m:
                                    old_sz = m.group(1).strip()
                                    new_sz = scale_font_size(old_sz, 1.2 if q.action == "enlarge" else 0.85)
                                    cur_css, css_spec = replace_css_property(cur_css, "font-size", new_sz, context_hint=hint)
                            elif q.category == "border_radius":
                                m = re.search(r"\bborder-radius\s*:\s*([^;]+);", cur_css, re.I)
                                if m:
                                    old_rad = m.group(1).strip()
                                    from sylphae.agent.qualitative import scale_dimension
                                    new_rad = scale_dimension(old_rad, 1.5 if q.action == "round" else 0.5)
                                    cur_css, css_spec = replace_css_property(cur_css, "border-radius", new_rad, context_hint=hint)
                            elif q.category in ("spacing", "dimension"):
                                for prop in ("padding", "margin", "gap", "height", "width"):
                                    m = re.search(rf"\b{prop}\s*:\s*([^;]+);", cur_css, re.I)
                                    if m:
                                        old_v = m.group(1).strip()
                                        from sylphae.agent.qualitative import scale_dimension
                                        new_v = scale_dimension(old_v, 1.25 if q.action in ("spacious", "enlarge") else 0.75)
                                        cur_css, css_spec = replace_css_property(cur_css, prop, new_v, context_hint=hint)
                                        break

                        if css_spec and cur_css != css_content:
                            new_vue_src = old_src[:m_style.start(2)] + cur_css + old_src[m_style.end(2):]
                            style_start_line = old_src[:m_style.start(2)].count("\n") + 1
                            css_rel_line = css_spec.span[0]
                            total_line = style_start_line + css_rel_line - 1
                            raw_line = lines_raw[total_line - 1] if 1 <= total_line <= len(lines_raw) else ""
                            ending = "\r\n" if raw_line.endswith("\r\n") else ("\n" if raw_line.endswith("\n") else "")
                            spec = OpSpec(
                                kind="replace_literal",
                                path=target_path,
                                span=(total_line, total_line),
                                props={
                                    "old_text": raw_line,
                                    "new_text": css_spec.props.get("new_text", "") + (ending if not css_spec.props.get("new_text", "").endswith(("\r", "\n")) else ""),
                                    "old": css_spec.props.get("old", ""),
                                    "new": css_spec.props.get("new", ""),
                                },
                            )
                            return spec, new_vue_src

        # Named values before numeric substitution: "给 badge 加一个圆角 8px" carries a
        # number, but the number is part of a property the request also names, and the
        # numeric branch would read it as a value to swap on whatever line it reached.
        if is_css or is_tsx:
            nv_spec, nv_src = self._build_named_value_patch(
                prompt, target_path, lines, lines_raw, old_src, cand
            )
            if nv_spec is not None:
                return nv_spec, nv_src

        # Handle numeric / string literal replacement
        if (op_kind == "replace_literal" or not op_kind) and param_val and param_val != "0.5x":
            anchor_line, anchor_symbols = self._anchor_context(lines, cand)

            # Prioritize candidate match line if available
            check_lines = []
            if anchor_line is not None:
                check_lines.append((anchor_line, lines[anchor_line - 1]))
            for i, line in enumerate(lines, 1):
                if anchor_line != i:
                    check_lines.append((i, line))

            kw_set = [k.lower() for k in extract_keywords(prompt)]
            for i, line in check_lines:
                line_lower = line.lower()
                is_target_line = (anchor_line == i) or any(k in line_lower for k in kw_set)
                if is_target_line:
                    if not self._placement_allowed(lines, anchor_line, anchor_symbols, i, line):
                        continue
                    m = re.search(r"(=\s*|:\s*|\bvalue\s*=\s*)([0-9]+(?:\.[0-9]+)?|\"[^\"]*\"|'[^']*')", line)
                    if m:
                        prefix = m.group(1)
                        old_val = m.group(2).strip("\"'")
                        new_literal = param_val if (param_val.startswith('"') or param_val.replace('.', '', 1).isdigit()) else f'"{param_val}"' if ('"' in m.group(2) or "'" in m.group(2)) else param_val
                        new_line = line[:m.start(2)] + new_literal + line[m.end(2):]
                        raw_line = lines_raw[i - 1]
                        ending = "\r\n" if raw_line.endswith("\r\n") else ("\n" if raw_line.endswith("\n") else "")
                        spec = OpSpec(
                            kind="replace_literal",
                            path=target_path,
                            span=(i, i),
                            props={
                                "old_text": raw_line,
                                "new_text": new_line + ending,
                                "old": old_val,
                                "new": new_literal.strip("\"'"),
                            },
                        )
                        new_lines = list(lines)
                        new_lines[i - 1] = new_line
                        return spec, "\n".join(new_lines) + ("\n" if old_src.endswith("\n") else "")

        # Handle rename_symbol
        if op_kind == "rename_symbol" and param_val:
            words = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", prompt) if len(w) >= 3]
            old_sym = words[0] if len(words) >= 2 else (cand.symbols[0] if cand and cand.symbols else None)
            new_sym = param_val
            if old_sym and old_sym in old_src:
                count = len(re.findall(r"\b" + re.escape(old_sym) + r"\b", old_src))
                spec = OpSpec(
                    kind="rename_symbol",
                    path=target_path,
                    span=(1, len(lines)),
                    props={"old": old_sym, "new": new_sym, "count": str(count)},
                )
                from sylphae.ops.apply import apply_op
                try:
                    new_src = apply_op(old_src, spec)
                    return spec, new_src
                except Exception:
                    pass

        return None, None

    async def _async_verify_and_update(
        self,
        changed_path: str,
        next_mem: MemState | None,
    ) -> VerifyReport:
        """Run verification in background and update persistent latent memory."""
        all_tests = [rel for rel in iter_python_files(self.root) if "test" in rel.lower()]
        target_tests = affected_tests(self.root, [changed_path], all_tests)

        def _run_verify() -> VerifyReport:
            return verify(self.root, [changed_path], target_tests, timeout_s=15)

        report = await asyncio.to_thread(_run_verify)

        # Optimistic verification check
        if report.compiles:
            # Experience persists: update recurrent state
            if next_mem is not None:
                self.latent_memory = next_mem.detach()
            try:
                cache_file = self.root / REPO_INDEX_FILE_NAME
                if cache_file.exists():
                    self.repo_index = RepoIndex.build(self.root)
                    self.repo_index.save(cache_file)
            except Exception:
                pass

        return report
