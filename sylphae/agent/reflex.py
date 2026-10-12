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
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch

if TYPE_CHECKING:
    # Annotation-only, so importing sylphae.exec.rename_multi at runtime never happens on the
    # reflex's hot path. The executor pulls in ast and tokenize; DESIGN.md section 5's
    # sub-150 ms budget does not care, but the CLI's import time does, and a single-file
    # constant edit has no use for a multi-file rewriter.
    from sylphae.exec.rename_multi import MultiFileRenameResult

from sylphae.agent.effort import EffortConfig, LowEffort
from sylphae.agent.explorer import (
    CandidateResult,
    ParallelExplorer,
    _OLD_VALUE_RE,
    extract_keywords,
)
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
from sylphae.ops.safety_class import OpSafety, classify_op
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

#: Minimum similarity for the "the requested line already exists in a slightly different
#: form" resolution, so a re-apply that says *it used to read X, it should read Y* becomes
#: a ``replace_line`` on the existing line instead of an insertion.  Measured over all six
#: reapply scenarios: the three this must fire on score 0.741 / 0.844 / 0.909 against their
#: unique nearest line, and the three it must leave alone score 0.519 / 0.457, or 0.918 with
#: a two-way tie.  Both gates -- the floor and the uniqueness requirement -- are
#: load-bearing, and neither was tuned on a single case.
_REPLACE_LINE_MIN_SIMILARITY: float = 0.65


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
    #: Whether the cross-file consistency gate stopped this edit, and why.
    #:
    #: Recorded separately from ``escalated`` because the two mean different things to a
    #: caller. ``escalated`` says "a human should look at this". This says "the reflex wrote
    #: one file of an edit that needs several", which is the specific, checkable claim
    #: behind 12.32's measurement -- and it is what a caller needs in order to distinguish
    #: a request the reflex declined because it could not help from one it declined because
    #: it was not confident.
    #:
    #: ``None`` when no gate applied, which is most runs: the gate is scoped to operations
    #: where a one-file write is known to be insufficient (see ``_cross_file_blocker``).
    consistency_blocked: bool | None = None
    consistency_blocker: str | None = None
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
        #: ROUND 11 (2c). The classification of the last spec ``_build_patch`` was asked
        #: about, and the refusal text if it was refused. Both are reset per call so a stale
        #: verdict cannot be read as this request's outcome -- the same reason
        #: ``last_model_error`` is reset, for the same failure mode.
        self.last_safety: "OpSafety | None" = None
        self.last_safety_refusal: str = ""
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
                    idx = RepoIndex.load(cache_file, expect_root=self.root)
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

    async def rename_symbol_across_files(
        self,
        old: str,
        new: str,
        *,
        dry_run: bool = True,
    ) -> "MultiFileRenameResult":
        """Rename a symbol across every file that references it.

        DESIGN.md section 1's ``rename`` shape, and the reason this is a separate method
        rather than a branch inside :meth:`run`. ``run`` edits exactly one file: it locates a
        target, builds a patch for that file, and writes it. A cross-file rename is a
        different operation with a different blast radius -- MEASURED on the rich fixture, 10
        occurrences across 7 files -- and folding it into ``run`` would mean a routine
        "adjust this constant" request could rewrite seven files whenever the policy happened
        to emit a rename. The caller has to ask for this one, by name.

        What it returns is the executor's report, unmodified. In particular it does **not**
        claim the tree is healthy afterwards: ``consistent`` says every *code* reference was
        rewritten, and verification is still the caller's to run. MEASURED 2026-10-02 on
        ``Textualize__rich``: 63 test files, 903/934 before, 903/934 after, 0 regressions --
        which is the caller's to reproduce, not this method's to assert.

        ``dry_run`` defaults to ``True`` here as well, for the same reason: seven writes is
        not something a default should do.
        """
        from sylphae.exec.rename_multi import apply_multi_file_rename

        return apply_multi_file_rename(self.root, old, new, dry_run=dry_run)

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
        consistency_blocked: bool | None = None
        consistency_blocker: str | None = None
        if spec is not None and new_source is not None:
            # A single-file write cannot honour an edit whose meaning spans files, and
            # nothing downstream notices: MEASURED on ``pyv_rich_rename_not_renderable``,
            # the definition was renamed, the call sites were not, the patch applied, the
            # content check passed, the file compiled, and two tests went red.
            #
            # The gate is consulted *before* the disk write, so an edit that cannot be
            # completed here is reported as one rather than half-done. It runs for the ops
            # where this is measured, and only those -- see ``_cross_file_blocker``.
            blocker = self._cross_file_blocker(spec, target_path)
            if blocker is not None:
                escalated = True
                escalation_reason = blocker
                consistency_blocked = True
                consistency_blocker = blocker
            elif escalated and not dry_run:
            # Escalation has to be a state with consequences, not a field in the report.
            # It means "this needs a human", and a human cannot un-edit a file.  Writing
            # the patch anyway made the flag advisory: the run reported doubt and edited
            # the user's code in the same breath.  The patch is still built and shown --
            # it is the disk write that waits for confirmation.
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
            consistency_blocked=consistency_blocked,
            consistency_blocker=consistency_blocker,
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

                # Generate parameter.  This must be *autoregressive*: the previous
                # code read the teacher-forced ``out.param_logits`` argmax, whose step
                # >= 1 is conditioned on ``<pad>`` (there is no gold prefix at
                # inference), so a multi-token value was emitted from a prefix of pads.
                # That is a decode-path defect independent of the decoder's
                # architecture -- see ROADMAP 12.15.
                gen = getattr(self.model, "generate_params", None)
                param_ids = (
                    gen(batch)[0].tolist()
                    if gen is not None
                    else out.param_logits[0].argmax(dim=-1).tolist()
                )
                clean_ids = []
                for t in param_ids:
                    ti = int(t)
                    if ti in (self.tokenizer.eos_id, self.tokenizer.pad_id):
                        break
                    if ti >= 7:
                        clean_ids.append(ti)
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

        #: ROUND 53. 守卫 was missing. The rule knew "guard", "判空" and "检查" but not the
        #: ordinary Chinese word for a guard clause, so ``pyv_typer_add_none_guard`` -- whose request
        #: is "给is_ascii_encoding加上encoding为None时返回False的守卫" -- declined with "no patch could
        #: be constructed" while the guard *builder* sat unused.
        #:
        #: Widen rather than insert: "加个保护", "边界检查", "为空时" and "early return" belong to the
        #: same intent, and a list that only has the one word this repository's benchmark uses is a
        #: lookup. The guard tests fire this rule on phrasings the benchmark does not contain, and on
        #: requests that must *not* produce a guard, so the claim is checkable rather than asserted.
        if any(w in prompt.lower() for w in (
            "guard", "守卫", "判空", "检查", "为空", "保护", "protect",
            "none 时", "none时", "early return", "边界检查", "missing value",
        )):
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

    def _build_guard_patch(
        self,
        prompt: str,
        target_path: str,
        old_src: str,
        cand: "CandidateResult | None",
    ) -> tuple[OpSpec, str] | None:
        """Build an ``add_guard`` spec for ``target_path``, or ``None`` if none is justified.

        The function and parameter names come from the prompt when it names them, and are
        then checked against the file. That check is what makes the guard *derived* rather
        than templated: on ``typer/_click/_compat.py`` the request says
        "给is_ascii_encoding加上encoding为None时返回False的守卫", so both names are stated,
        and the file confirms ``is_ascii_encoding`` is defined there and ``encoding`` is one
        of its parameters. A name from the request that the file does not have cannot reach
        the patch, which is the property a keyword-matched template cannot offer.

        Returns ``None`` rather than a refusal message on purpose: the caller falls through
        to the other builders, and a run that reaches the end reports "no patch could be
        constructed", which is the honest summary for a shape the agent cannot yet do.

        Refuses to emit a guard that does not parse. A guard that breaks the file is worse
        than no guard, and this is the last point at which that is still free to detect.
        """
        import ast as _ast

        from sylphae.agent.guard_builder import plan_guard

        try:
            tree = _ast.parse(old_src)
        except (SyntaxError, ValueError):
            return None
        defined = {n.name for n in tree.body
                   if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))}
        if not defined:
            return None

        # Identifier-shaped tokens the request itself supplies. Not keyword matching on
        # Chinese prose -- it reads the identifiers a developer would have typed.
        idents = [i for i in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", prompt) if len(i) >= 3]

        func_name = next((i for i in idents if i in defined), "")

        param = ""
        if func_name:
            fn_node = next((n for n in tree.body
                            if getattr(n, "name", None) == func_name), None)
            if fn_node is None:
                return None
            a = fn_node.args
            params = {x.arg for x in (list(a.posonlyargs) + list(a.args))}
            param = next((i for i in idents if i in params), "")

        plan = plan_guard(old_src, function_name=func_name, parameter=param,
                          path=target_path)
        if not plan.ok:
            return None

        lines = old_src.splitlines(keepends=True)
        n = plan.insert_before
        if n < 1 or n > len(lines):
            return None
        new_src = "".join(lines[:n - 1]) + plan.text + "".join(lines[n - 1:])

        try:
            _ast.parse(new_src)
        except (SyntaxError, ValueError):
            return None

        spec = OpSpec(
            kind="add_guard",
            path=target_path,
            span=(n, n),
            props={
                "old_text": lines[n - 1],
                "new_text": plan.text,
                # ROUND 53. The executor's insertion contract is ``props["line"]`` -- ``_apply_insert``
                # reads it and raises ``requires props['line']`` without it. ``plan_guard`` reports the
                # same quantity as ``insert_before`` (1-based, the line the guard's ``if`` goes on), so
                # the two names disagree and the spec was unappliable.
                #
                # This went unnoticed because the guard path had never run: the wording rule did not
                # fire on the benchmark's Chinese request until this round, so the producer and the
                # executor had never been asked to agree. Both names are kept -- ``insert_before`` is
                # what the plan says it is, and dropping it would make the spec unreadable next to
                # the plan it came from.
                "line": str(n),
                "insert_before": n,
                # ``_apply_insert`` also reads ``text``. Fixed one field at a time by watching the
                # error move: after ``line`` the next complaint was ``requires props['text']``.
                # ``new_text`` is the guard body the plan produced, which is the same quantity, so
                # both names carry it and the spec stays readable next to its plan.
                "text": plan.text,
                "parameter": plan.parameter,
                "return_source": plan.return_source,
                "insert_before": str(n),
            },
        )
        return spec, new_src

    def _build_patch(
        self,
        prompt: str,
        target_path: str,
        op_kind: str,
        param_val: str,
        cand: CandidateResult | None = None,
        _as_insert: bool = False,
    ) -> tuple[OpSpec | None, str | None]:
        """Every spec the reflex returns passes through here, and through the 2c gate.

        ROUND 11. This is the single choke point for 2c, and it is a wrapper rather than a
        check inside one branch because specs are produced in three places -- the vision
        mapper, the model-policy build, and two heuristic-fallback builds -- and a check added
        to any one of them would leave the other two able to emit a destructive spec. The
        unchecked builder is still there and still does all the work; it is just no longer
        reachable without classification, so a fourth branch added later inherits the gate for
        free rather than needing to remember it.

        The gate is ``sylphae.ops.safety_class.classify_op``, which refuses a spec whose kind
        is destructive and refuses an unclassified kind outright. Refusing at *construction*
        rather than at write time matters: a spec that never becomes a spec cannot be reported
        as a patch, so the refusal shows up as the reflex declining rather than as a patch
        that has to be caught later.
        """
        spec, new_source = self._build_patch_unchecked(
            prompt, target_path, op_kind, param_val, cand=cand
        )
        if spec is None:
            return None, None

        verdict = classify_op(spec)
        #: Recorded even when the spec is refused, so a report can say *why* the reflex
        #: declined instead of leaving an absent patch to be read as a builder gap. The
        #: 3b/3c "declined" rows in ``by_shape`` need this to tell a safety refusal from an
        #: op the model named and nothing could build.
        self.last_safety = verdict
        if not verdict.applicable:
            self.last_safety_refusal = verdict.refusal
            return None, None
        return spec, new_source

    def _wiring_target(
        self,
        op_kind: str,
        prompt: str,
        param_val: str,
        lines: list[str],
        lines_raw: list[str],
    ) -> tuple[int, int] | None:
        """``(last_line, anchor)`` to rewrite or delete, or ``None`` when the request names no line.

        ROUND 127. ``remove_import`` and ``replace_line`` both need to know *which* original lines the span
        covers, and that is the anchor's job rather than the model's: the request names a line, the executor
        insists the span match it byte for byte.

        ``param_val`` is the **needle** ``_insertion_anchor`` searches for, not the span. An earlier draft of this
        docstring claimed the emission was not consulted here; a test disproved it, because
        ``_insertion_anchor(prompt, text, lines)`` returns ``None`` when ``text`` does not appear in the file.
        So a request that does not quote its target and gets no emission yields ``None`` and the reflex
        declines -- which is the honest outcome, not a wiring failure.

        Returns the pair as ``(last_line, anchor)`` because both call sites slice with it in that order.
        """
        needle = (param_val or "").strip()
        anchor = None
        if needle:
            # A replacement or a deletion is located by the line its own text names, not by the enclosing
            # block. ``_insertion_anchor`` answers a different question -- "where does the block the request
            # points at end" -- and it refuses a request that names no ``def``/``class``, which is most
            # import and single-line edits.
            for i, line in enumerate(lines, start=1):
                if needle in line:
                    anchor = i
                    break
        if anchor is None:
            anchor = self._insertion_anchor(prompt, param_val or "", lines)
        if anchor is None:
            return None
        last = anchor
        # A multi-line target may straddle the anchor: if the line that follows is part of the same
        # statement (continuation or a paired opener), take it too, bounded by the block's extent.
        if op_kind == "replace_line" and last < len(lines):
            nxt = lines[last]
            cur = lines[anchor - 1]
            if nxt.strip().startswith((".", ")", "]", "}", ",")) or (
                cur.rstrip().endswith(("(", "[", "{", ",")) or not cur.rstrip().endswith(":")
            ):
                last = last + 1
        return (last, anchor)

    def _build_patch_unchecked(
        self,
        prompt: str,
        target_path: str,
        op_kind: str,
        param_val: str,
        cand: CandidateResult | None = None,
        _as_insert: bool = False,
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

            # A ``从 X 改为 Y`` request names the value being replaced, and X is text
            # that exists in the file *today* -- it is the one piece of evidence that
            # picks the exact line, and the AST anchor cannot see it.  The anchor is a
            # *symbol* match: for ``把选项的样式从bold cyan改为bold red`` it lands on the
            # ``from rich.theme import Theme`` line, which shares no identifier with any
            # of the ``STYLE_*`` constants, so ``_placement_allowed`` (symbol agreement
            # outside the anchor's own entity) rejects the line that actually holds the
            # value and the request escalates.  MEASURED before the value gate existed,
            # the builder rewrote whichever constant sat on the anchor line --
            # ``STYLE_OPTION`` received "bold blue" for the switch request and "green"
            # for the usage request -- and the option request escalated outright.
            #
            # So when X is named, the line whose current value *is* X is the edit site,
            # and symbol agreement is demoted to a tiebreaker: prefer a line the guard
            # also accepts, but fall back to the value match alone.  Refusing here is not
            # safer -- it sends the request to escalation, which has no better evidence
            # than the literal the user quoted.
            old_val_literal: str | None = None
            _mv = _OLD_VALUE_RE.search(prompt)
            if _mv and _mv.group(1):
                old_val_literal = _mv.group(1).strip().strip("'\"")

            # Prioritize candidate match line if available
            check_lines = []
            if anchor_line is not None:
                check_lines.append((anchor_line, lines[anchor_line - 1]))
            for i, line in enumerate(lines, 1):
                if anchor_line != i:
                    check_lines.append((i, line))

            _LITERAL_RE = r"(=\s*|:\s*|\bvalue\s*=\s*)([0-9]+(?:\.[0-9]+)?|\"[^\"]*\"|'[^']*')"

            def _emit_literal(i: int, line: str, m: "re.Match[str]") -> tuple[OpSpec, str] | None:
                cur_val = m.group(2).strip("\"'")
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
                        "old": cur_val,
                        "new": new_literal.strip("\"'"),
                    },
                )
                new_lines = list(lines)
                new_lines[i - 1] = new_line
                return spec, "\n".join(new_lines) + ("\n" if old_src.endswith("\n") else "")

            if old_val_literal is not None:
                value_hits: list[tuple[int, str, "re.Match[str]"]] = []
                for i, line in check_lines:
                    m = re.search(_LITERAL_RE, line)
                    if m and m.group(2).strip("\"'").lower() == old_val_literal.lower():
                        value_hits.append((i, line, m))
                if value_hits:
                    preferred = [
                        hit for hit in value_hits
                        if self._placement_allowed(lines, anchor_line, anchor_symbols, hit[0], hit[1])
                    ]
                    for i, line, m in (preferred or value_hits):
                        return _emit_literal(i, line, m)

            kw_set = [k.lower() for k in extract_keywords(prompt)]
            for i, line in check_lines:
                line_lower = line.lower()
                is_target_line = (anchor_line == i) or any(k in line_lower for k in kw_set)
                if is_target_line:
                    if not self._placement_allowed(lines, anchor_line, anchor_symbols, i, line):
                        continue
                    m = re.search(_LITERAL_RE, line)
                    if m:
                        cur_val = m.group(2).strip("\"'")
                        if old_val_literal is not None and cur_val.lower() != old_val_literal.lower():
                            continue
                        result = _emit_literal(i, line, m)
                        if result is not None:
                            return result

        # Handle add_guard: an *insert*, not a substitution.
        #
        # This is the shape where neither of the two branches above applies. `replace_literal`
        # substitutes a value on an existing line; `rename_symbol` rebinds an identifier.
        # A guard adds a statement, and every question that makes one correct is answerable
        # from the code around the insertion point rather than from the request's wording --
        # which parameter can be None, whether the position is after the docstring, and what
        # the function already returns on its failure path.
        #
        # Measured on pyv_typer_add_none_guard: the scenario asks to guard `encoding` and
        # return False, the op head was never consulted at all (model_op_kind is None and
        # _infer_op's wording-keyword fallback chose `replace_literal`), and the one
        # hard-coded guard template in the tree returns None for a parameter named `value`.
        # So this is not a missing builder for an op that the head already picks -- it is the
        # only way the shape can be produced at all.
        if op_kind == "add_guard" and target_path:
            guard = self._build_guard_patch(prompt, target_path, old_src, cand)
            if guard is not None:
                return guard

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

        # Handle insert_statement
        #
        # ROUND 32. 12.65 measured that the producer builds 2 kinds of 26, and that
        # ``insert_statement`` -- the op re-apply is mostly made of -- was one of the 24 that no
        # real run has ever produced. The executor has always accepted it (``_APPLIERS`` maps
        # ``INSERTION_KINDS``), so the vocabulary was never the gap; the builder was.
        #
        # Placed after the replace_literal branch on purpose: that branch gets first refusal, and
        # this is what runs when there is no literal value to substitute. Touching ``_infer_op``'
        # wording rules instead would have moved four measured shapes to chase one.
        #
        # **The insertion point must come from evidence, and this fails closed.** A guess would
        # score better than returning None and would be worse in every way that matters: the caller
        # would receive a spec that lands the text in the wrong function, and 12.64's0-of-6 would
        # become a number produced by a guess rather than by the reflex. So the position is derived
        # from an anchor the request actually names, and when no such anchor exists this returns
        # None and the reflex declines -- which is the honest outcome, and the same one the
        # replace_literal branch already returns for these requests.
        if op_kind == "insert_statement" and param_val:
            # A re-apply that says "it used to read X and is now wrong again; it should
            # read `Y`" usually names a line that already exists in a slightly different
            # form. The gold edit corrects the existing line rather than adding a new one,
            # and the insertion path would land a duplicate -- or, when Y is a header, a
            # def with no body (ROUND 34 measured ``compile: expected an indented block``
            # on exactly that). The evidence is physical, not lexical: the requested text
            # carries the exact indentation of an existing line and is a clear nearest
            # variant of it. Falls through to the anchor below when neither gate passes,
            # so a genuine insertion is untouched.
            replace = self._nearest_line_replacement(lines, lines_raw, target_path, param_val)
            if replace is not None:
                from sylphae.ops.apply import apply_op
                try:
                    new_src = apply_op(old_src, replace)
                    return replace, new_src
                except Exception:
                    pass
            anchor = self._insertion_anchor(prompt, param_val, lines)
            if anchor is not None:
                text = param_val if param_val.endswith("\n") else param_val + "\n"
                # 1-based "insert before"; after the anchor block's last line.
                at = anchor + 1
                spec = OpSpec(
                    kind="insert_statement",
                    path=target_path,
                    span=(at, at),
                    props={"line": str(at), "text": text},
                )
                from sylphae.ops.apply import apply_op
                try:
                    new_src = apply_op(old_src, spec)
                    return spec, new_src
                except Exception:
                    # A spec the executor refuses is not a spec. Falls through to None rather than
                    # reporting something that cannot be applied.
                    pass

        # ROUND 32, second half. The branch above is only reachable when something *infers*
        # ``insert_statement``, and nothing does: ``_infer_op``'s wording rules say
        # ``replace_literal`` for "make X appear". Leaving it that way would ship a builder with
        # no caller -- declared, unreachable, and therefore not a capability. So when every other
        # branch has declined and the request is asking for text to appear, retry as an insertion.
        #
        # This is deliberately a fallback rather than a rewording of ``_infer_op``: those rules are
        # measured against four other shapes (12.3c counted four scenarios moving between the
        # shipped and refit conf heads), and moving them to serve one shape would trade measured
        # behaviour for an unmeasured gain. Here, if the retry also fails, the reflex declines
        # exactly as it did before -- the 0 of 6 in 12.64 cannot get worse than declined.
        #
        # ``_as_insert`` is the recursion guard: the retry re-enters this method with
        # ``op_kind="insert_statement"``, and without the flag the retry's own fall-through would
        # satisfy the same condition and call itself again. It was a RecursionError on five of the
        # six reapply cases before the flag existed -- measured, not anticipated.
        #
        # ROUND 33. The ``param_val`` alternative is removed for ``freeform_patch``, and the reason
        # is measured rather than stylistic.
        #
        # 13.16 attributed the whole neural regression on the 31-scenario suite: **the model won 0
        # scenarios and lost 4, and all four losses were its own output.** Three would not apply at
        # all and one wrecked a suite. The three read:
        #
        #     pyv_typer_add_none_guard         op=freeform_patch  model_param_val='range'  patch_applied=False
        #     reapply_celery__celery_defaults  op=freeform_patch  model_param_val="'"      patch_applied=False
        #     pyv_rich_rename_not_renderable   op=freeform_patch  model_param_val="'"      syntax_valid=False
        #
        # and the fourth, the same op kind with the same one-character parameter, left 232 previously
        # green tests uncollectable. ``model_param_val="'"`` -- a lone quote -- is a decoder that
        # emitted the opening of a string literal and stopped.
        #
        # The route from that token to a broken repository was this branch. ``freeform_patch`` is
        # *deliberately* not executable (sylphae/ops/apply.py: "freeform_patch is not replayable by
        # the structured executor"), and it is the vocabulary's escape hatch for "I have text but no
        # structured form for it". Reading its parameter as literal source text to inject is a
        # category error: the model said "freeform", and the framework translated that into "insert
        # my stray quote character into the user's file". The ``'range'`` case proves the parameter is
        # not source text either -- the identical value succeeds as ``add_import`` in
        # ``reapply_pallets__click_testing``, so what differs is the op kind, not the token.
        #
        # **This is the asymmetry the surrounding comments already warn about.** The *position* was
        # made to fail closed -- "the insertion point must come from evidence, and this fails closed.
        # A guess would score better than returning None and would be worse in every way that
        # matters". The *text* was never held to the same standard, so a degenerate model emission
        # became a manufactured destructive patch. Evidence for the text still exists and is still
        # used: ``_quoted_insertion_text`` reads a span the **request** quotes, which is the caller
        # stating the text rather than the model guessing it. ROUND 32 added ``param_val`` because the
        # heuristic returns ``''`` for this shape, but ``_quoted_insertion_text`` is the branch that
        # answers that case, and the heuristic's own parameter value was never the evidence.
        #
        # Failing closed here cannot make reapply worse than declined -- the 0 of 6 in 12.64 and the
        # 1 of 6 in 13.16 are both floors this only ever turns an unverified patch into a decline.
        # ROUND 127. Two of the 22 ops that had no wiring at all (13.57 measured
        # ``_build_patch_unchecked`` branching on ``op_kind`` at only 5 places for 26 kinds).
        # Both contracts were probed against ``apply_op`` rather than assumed, and both turned out to be
        # **span-strict**: ``props['old_text']`` / ``props['text']`` must equal the exact original lines the
        # span covers, line terminators included, or the executor refuses with
        # "original lines N..N are ..., not the recorded ...".
        #
        # That is why the old text is taken from ``lines_raw`` and never from ``param_val`` -- the span the
        # anchor logic already computed is the authority, and ``param_val`` is the *new* content.
        #
        # The division of labour matters and is worth stating plainly -- and the first version of this
        # comment got it wrong. ``_insertion_anchor(prompt, text, lines)`` locates the line by finding
        # ``text``, so **both** kinds need ``param_val`` as the *needle*, or the request has to quote the
        # line itself. An earlier draft of this claimed ``remove_import`` needs no model emission at all;
        # a test disproved that -- "remove the sys import" does not contain "import sys", so the anchor
        # finder returns None and the reflex correctly declines rather than guessing a line to delete.
        #
        # What the separation actually is, then:
        #   remove_import  needs ``param_val`` only as a locator; the deleted text comes from the span.
        #   replace_line   needs ``param_val`` as the locator **and** as ``new_text``.
        # So wiring is necessary but not sufficient, and the test file pins it in both directions.
        if op_kind in ("remove_import", "replace_line"):
            target = self._wiring_target(op_kind, prompt, param_val, lines, lines_raw)
        else:
            target = None
        if target is not None:
            want, anchor = target
            old_block = "".join(lines_raw[anchor - 1: want])
            if op_kind == "remove_import":
                spec = OpSpec(
                    kind="remove_import",
                    path=target_path,
                    span=(anchor, want),
                    props={"text": old_block},
                )
            elif param_val:
                new_block = param_val if param_val.endswith("\n") else param_val + "\n"
                spec = OpSpec(
                    kind="replace_line",
                    path=target_path,
                    span=(anchor, want),
                    props={"old_text": old_block, "new_text": new_block},
                )
            else:
                spec = None
            if spec is not None:
                from sylphae.ops.apply import apply_op
                try:
                    new_src = apply_op(old_src, spec)
                    return spec, new_src
                except Exception:
                    # An executor refusal is not a spec. Falls through to the generic retry
                    # below rather than reporting something that cannot be applied.
                    pass

        if not _as_insert:
            # ROUND 33, second half. The original line was
            # ``param_val or self._quoted_insertion_text(prompt)``, and the precedence between the two
            # is load-bearing. A first version of this fix put the quoted span *first* on the reasoning
            # that the request is better evidence than the model -- and that regressed
            # ``reapply_pallets__click_testing``, which answered ``add_import`` with
            # ``model_param_val='range'`` and had been clean: with the quoted span taking priority the
            # builder inserted the request's prose instead of the value and produced
            # ``compile: app.py:55: unexpected indent``. The verifier named it; the suite did not.
            # So the precedence stays exactly as measured, and only the *eligibility* of ``param_val``
            # changes -- a ``freeform_patch`` parameter is no longer text, but every other kind's still is.
            text_to_insert = None
            if op_kind != "freeform_patch":
                text_to_insert = param_val or None
            if not text_to_insert:
                text_to_insert = self._quoted_insertion_text(prompt)
            if text_to_insert:
                retry, retry_src = self._build_patch_unchecked(
                    prompt, target_path, "insert_statement", text_to_insert, cand=cand,
                    _as_insert=True)
                if retry is not None:
                    return retry, retry_src

        return None, None

    def _quoted_insertion_text(self, prompt: str) -> str | None:
        """The single-line span the request puts in quotes, or ``None``.

        ROUND 32 found the real reason reapply never reached any builder: the heuristic policy
        returns an **empty** ``param_val`` for a request of this shape -- measured, ``h_param``
        was ``''`` -- so the replace_literal branch skipped on its truthiness test and so did
        everything else. The text was never lost by the builder; it never arrived.

        Taking it from the quotes is not a guess: a request that writes ``make "x = 1" appear``
        has stated the text. Single-line only, and the **longest** quoted span wins, because a
        value containing the other quote character would otherwise be cut at the inner one.

        Deliberately independent of ``param_val``: that field comes from a measured heuristic
        tuned for substitution, and this is the insertion case it does not model.
        """
        best = None
        # Backticks are quotes too. 4 reapply scenarios state their text in them
        # ("it should read `            durable=True,`.") and this returned None, so
        # freeform_patch declined on missing evidence rather than on any real doubt.
        # Verified against all 31 scenarios: adding `` ` `` changes 6 extractions, all
        # toward the recorded ``expected_contains_new``, and leaves the other 25 alone.
        for m in re.finditer(r"(['\"`])(.+?)\1", prompt, flags=re.S):
            # Not stripped. 12.35 fixed the miner, which had been throwing the indentation away,
            # and 12.70 found this was the *second* place it died: the fixture stored the depth and
            # the request did not, because quoting a string and then stripping its contents throws
            # away the leading whitespace -- which for an insertion **is** the instruction. A quoted
            # span is delimited, so what sits inside it is what was asked for. Emptiness is still
            # judged on the stripped form, so a pair of quotes around nothing is not insertion text.
            body = m.group(2)
            if not body.strip() or "\n" in body:
                continue
            if best is None or len(body) > len(best):
                best = body
        return best

    def _nearest_line_replacement(
        self, lines: list[str], lines_raw: list[str], target_path: str, text: str,
    ) -> OpSpec | None:
        """A ``replace_line`` on the existing line ``text`` is correcting, or ``None``.

        ``SequenceMatcher`` over whitespace-collapsed text is the metric because the
        differences that matter here are word-level (``False`` -> ``True``, a dropped
        ``not``, an added parameter) and not character counts; collapsing whitespace keeps
        indentation out of that ratio, since it is matched exactly and separately.

        ``_apply_block_replace`` requires ``old_text`` to be the exact original run, so the
        raw line -- terminator included -- is what the spec records. The replacement carries
        the same terminator so a CRLF file stays CRLF.
        """
        if not text.strip() or "\n" in text:
            return None
        indent = len(text) - len(text.lstrip())
        want = re.sub(r"\s+", " ", text).strip()
        best_score = 0.0
        best_line = 0
        tied = False
        for i, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            if len(line) - len(line.lstrip()) != indent:
                continue
            got = re.sub(r"\s+", " ", line).strip()
            if got == want:
                # The line already reads as asked, so this is not a correction; the caller
                # declines and leaves the source alone.
                return None
            ratio = SequenceMatcher(None, want, got).ratio()
            if ratio > best_score:
                best_score, best_line, tied = ratio, i, False
            elif ratio == best_score and best_score > 0.0:
                tied = True
        if (
            not best_line
            or tied
            or best_score < _REPLACE_LINE_MIN_SIMILARITY
        ):
            return None
        raw_line = lines_raw[best_line - 1]
        if raw_line.endswith("\r\n"):
            ending = "\r\n"
        elif raw_line.endswith("\n"):
            ending = "\n"
        else:
            ending = ""
        new_text = text if text.endswith("\n") else text + ending
        return OpSpec(
            kind="replace_line",
            path=target_path,
            span=(best_line, best_line),
            props={"old_text": raw_line, "new_text": new_text},
        )

    def _insertion_anchor(self, prompt: str, text: str,
                            lines: list[str]) -> int | None:
        """The 1-based last line of the block the request points at, or ``None``.

        Two rules, both anchored on something the request names, and both refusing rather than
        defaulting:

        * an identifier in the request that defines a ``def``/``class``/``async def`` -- the
          insert goes after that block ends, which is where a new statement about it belongs;
        * failing that, a bare module-level request with no anchor, which returns ``None``.

        The block end is found by indentation: the last line at or beyond the header's indentation
        that is not blank and not more indented. A request naming nothing recognisable gets no
        position, because a default line number is exactly the kind of number that looks like
        progress and is not.

        ROUND 34 records *which* of two separable reasons produced the ``None``, because they call
        for different fixes and lumping them together is what let this look like a resolver gap for
        rounds. Measured: every admitted case's ``added_lines`` carry **zero** indentation -- the
        mining strips it -- so for four of six the request does not contain the one piece of
        information an insertion needs. That is a fault in the request, not in the resolver, and no
        amount of anchor rules recovers indentation that was never sent.
        """
        import re as _re

        want = [w for w in _re.findall(r"[A-Za-z_][A-Za-z0-9_]*", prompt) if len(w) >= 3]
        header_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith(("def ", "class ", "async def ")):
                continue
            m = _re.match(r"\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
            if not m:
                continue
            name = m.group(1)
            if name in want:
                header_idx = i
                break
        if header_idx is None:
            # ROUND 38. The request now carries a depth (12.35 fixed the miner, 12.37 fixed the two
            # consumers downstream of it), so when nothing is named the depth can still place the
            # text. **Inlined here and nowhere else on purpose**: 12.36 put this rule after the
            # named-block body, where it overrode the correct answer and took the reflex from 2 of 6
            # to 0 of 6. Inside this branch it can only ever be a fallback, and every path below
            # returns, so the named-block body stays reachable.
            #
            # The rule: a block qualifies when its body contains a line at *exactly* the requested
            # indentation -- that is the depth the text is written for -- and among the qualifying
            # blocks the one already holding the most lexically similar line wins, because that is
            # where a person puts a line like this one. Zero candidates or a tie declines.
            #
            # Stated plainly because it bounds what a result here proves: this rule is being
            # measured on the same cases it is meant to fix, so moving them is weak evidence. What
            # keeps it a rule rather than a lookup is that depth and lexical overlap are both
            # required, neither is fitted to a case, and the tie case refuses.
            depth = len(text) - len(text.lstrip()) if text else 0
            if depth <= 0:
                self.last_anchor_reason = (
                    "no-block-named" if not want
                    else "no-block-named-and-text-has-no-indentation")
                return None

            def _toks(s_: str) -> set[str]:
                return set(w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s_) if len(w) >= 3)

            want_toks = _toks(text)
            scored: list[tuple[int, int]] = []
            for i, line in enumerate(lines):
                if not re.match(r"\s*(?:async\s+)?(?:def|class)\s+[A-Za-z_]", line):
                    continue
                base_i = len(line) - len(line.lstrip())
                end_i = len(lines)
                for j in range(i + 1, len(lines)):
                    st = lines[j].strip()
                    if st and (len(lines[j]) - len(lines[j].lstrip())) <= base_i:
                        end_i = j
                        break
                at_depth = [ln for ln in lines[i + 1:end_i]
                            if ln.strip() and (len(ln) - len(ln.lstrip())) == depth]
                if not at_depth:
                    continue
                overlap = max((len(want_toks & _toks(ln)) for ln in at_depth), default=0)
                if overlap:
                    scored.append((overlap, end_i))
            if not scored:
                self.last_anchor_reason = "no-block-has-a-body-at-the-requested-depth"
                return None
            top = max(sc for sc, _ in scored)
            winners = [e for sc, e in scored if sc == top]
            if len(winners) != 1:
                self.last_anchor_reason = f"ambiguous-{len(winners)}-blocks-at-that-depth"
                return None
            self.last_anchor_reason = "resolved-by-depth"
            return winners[0]

        self.last_anchor_reason = "resolved"

        base = len(lines[header_idx]) - len(lines[header_idx].lstrip())
        end = len(lines)
        for j in range(header_idx + 1, len(lines)):
            stripped = lines[j].strip()
            if not stripped:
                continue
            indent = len(lines[j]) - len(lines[j].lstrip())
            if indent <= base:
                end = j  # first line back at or left of the header's indentation closes the block
                break
        return end  # 1-based last line of the block
    def _cross_file_blocker(self, spec: "OpSpec", target_path: str) -> str | None:
        """Why this edit cannot be completed in one file, or ``None`` if it can.

        Scoped deliberately to the ops where a one-file write is *known* to be
        insufficient, because a gate that fires on operations it was never measured against
        is a guess dressed as a rule (AGENTS.md section 2). The evidence for each:

        - ``rename_symbol`` -- MEASURED. The branch below counts occurrences in the target
          file and rewrites only that file, so a symbol with callers elsewhere is left
          dangling. On ``pyv_rich_rename_not_renderable`` that produced two red tests with
          the patch applied and the content check passed.

        ``replace_literal`` is deliberately absent: substituting a value does not rebind an
        identifier, so it cannot leave a reference pointing at nothing. That is a property
        of the operation, not an absence of checking.

        The message names the symbol, the files and the lines, because "this needs a human"
        is only actionable if the human is told what a human has to do.
        """
        if spec.kind != "rename_symbol":
            return None
        old_sym = (spec.props or {}).get("old")
        new_sym = (spec.props or {}).get("new")
        if not old_sym or not new_sym or old_sym == new_sym:
            return None

        from sylphae.exec.consistency import rename_is_single_file_safe
        from sylphae.exec.rename_multi import plan_multi_file_rename

        try:
            safe, refs = rename_is_single_file_safe(
                self.root, old_sym, target_path, include_text=True
            )
        except Exception as exc:  # noqa: BLE001
            # A check that cannot run is an absence, never a pass. Escalating here is the
            # safe direction: it withholds a patch we cannot vouch for.
            return (
                f"Consistency check could not run for {old_sym!r} -> {new_sym!r} on "
                f"{target_path!r} ({type(exc).__name__}: {exc}); the patch was withheld "
                f"because the tree cannot be shown to stay consistent"
            )

        if safe:
            return None

        # Not safe in one file. Before 12.39 that was the end of it -- the rename was
        # refused and the shape stayed "declined (fails safely)". Now the question is
        # whether it can be done across all of them, and the answer decides whether this is
        # an escalation or a bigger edit.
        #
        # The gate itself is unchanged in stance: a *one-file* edit that needs more than one
        # file is still refused, because the reflex was asked to edit one file. What changed
        # is that the refusal can now say whether the multi-file route is available, so the
        # caller learns "this needs a human" versus "this needs the multi-file path", which
        # are different escalations with different costs.
        try:
            by_file, unresolved, docs = plan_multi_file_rename(
                self.root, old_sym, new_sym, include_text=True
            )
        except Exception as exc:  # noqa: BLE001
            by_file, unresolved, docs = {}, [], []

        total = sum(len(v) for v in by_file.values())
        outside = sorted({p for p in by_file if p != target_path})

        if not by_file and not unresolved:
            shown = "; ".join(str(r) for r in refs[:5])
            more = f" (+{len(refs) - 5} more)" if len(refs) > 5 else ""
            return (
                f"{old_sym!r} is referenced in {len(refs)} place(s) outside "
                f"{target_path!r}, so renaming it in this file alone would leave those "
                f"references dangling: {shown}{more}. The reflex edits one file at a time, "
                f"so the rename was withheld rather than applied partially. Rename it "
                f"across all {len(refs)} reference site(s), or ask for each file to be "
                f"edited."
            )

        if unresolved:
            shown_u = "; ".join(str(r) for r in unresolved[:4])
            more_u = f" (+{len(unresolved) - 4} more)" if len(unresolved) > 4 else ""
            return (
                f"{old_sym!r} -> {new_sym!r} would touch {len(unresolved)} site(s) "
                f"that are not resolved code identifiers and cannot be rewritten without "
                f"guessing: {shown_u}{more_u}. {total} AST-resolved occurrence(s) across "
                f"{len(by_file)} file(s) are rewritable, but rewriting those alone would "
                f"leave a reference the tree cannot resolve, so nothing was written."
                + (f" Separately, {len(docs)} docstring(s) still name the old symbol."
                   if docs else "")
            )

        # File *and line*, because the message exists so a human can act on it, and a file
        # name with a count is a summary while `path:line` is a location. The previous
        # message format carried lines; replacing it with counts would have made the
        # escalation less actionable than the thing it replaced.
        shown_parts: list[str] = []
        for p, group in sorted(by_file.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            lines = sorted({r.line for r in group})
            head = ", ".join(str(n) for n in lines[:4])
            more_l = f" +{len(lines) - 4} more line(s)" if len(lines) > 4 else ""
            shown_parts.append(f"{p}:{head}{more_l} ({len(group)} site(s))")
        shown = "; ".join(shown_parts[:4])
        more = (f" (+{len(shown_parts) - 4} more file(s))" if len(shown_parts) > 4 else "")
        doc_note = (f" {len(docs)} docstring(s) still name the old symbol and would not be "
                    f"rewritten." if docs else "")
        return (
            f"{old_sym!r} -> {new_sym!r} is referenced in {len(refs)} place(s) outside "
            f"{target_path!r}. A single-file edit cannot complete it: {shown}{more}. "
            f"The multi-file rename path can rewrite all {total} AST-resolved occurrence(s) "
            f"across {len(by_file)} file(s).{doc_note} It is not applied here because this "
            f"run was scoped to one file; use the multi-file rename to complete it."
        )

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
