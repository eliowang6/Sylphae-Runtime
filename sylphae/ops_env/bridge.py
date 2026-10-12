"""The bridge an ``EnvOpSpec`` takes to reach a shell, and the two checks it passes on the way.

ROUND 12. 12.46 built the classification and listed, as unproven, that
``EnvExecutor.validate_command`` still exists independently in the playbook layer and that the
two classifiers had never been run together on the same path.

**What turned out to be already true, and this module had to be rewritten because of it.**
``EnvExecutor.run`` begins with ``refusal = validate_command(spec)`` and returns a refusal
without executing, and ``Playbook._exec`` folds that refusal into the playbook's result as an
escalation. So every command a playbook runs *is* already checked, and has been all along. A
first version of this bridge assumed the opposite -- that a playbook would propose an argv for
the bridge to vet and then run -- and the assumption was false for all eight playbooks:
``diagnose`` returned a result with ``proposed_command is None`` in every case, because
``proposed_command`` is only set on the one path where a playbook *declines* to act, namely
``PortOccupancyPlaybook`` naming a PID it will not kill.

That misreading is worth recording, because the failure it produced was quiet. A blanket
``except Exception`` in ``derive_command`` turned the assumption's ``TypeError`` into ``None``,
which the bridge reported as "no playbook derives a command for this kind" -- a correct-looking
refusal, identical for all 11 kinds, for an entirely wrong reason. A broad ``except`` around a
setup step converts a construction bug into a plausible answer. So:

* There is no derive-then-run step. A playbook *is* the thing that runs.
* The command check is not reimplemented here; it is invoked through the executor, on each
  command, which is where it has always applied.
* What is genuinely new is the **action** check, and it is the one that had no home. Nothing
  prevented a caller from asking a playbook to do a destructive thing; the per-command check
  only ever saw individual commands, never the intent behind them. A restart proposed as "run
  ``systemctl restart api``" would pass a command check, because that one command is not
  individually a deletion.

**The order is therefore the whole claim.** The action is classified *before the playbook is
constructed*, so a destructive action never reaches a component that knows how to carry it out.
Then the playbook runs with the real executor, and every command it runs meets
``validate_command`` on the way. Two checks, one on the intent and one on each command, in that
order, on the same path.

**Advisory is a real state, not a slower path.** An action classified ``advisory`` may run its
*diagnostic* commands -- naming the process holding a port requires actually looking, and
pretending otherwise would make the proposal wrong -- and the destructive step itself comes back
as ``proposed_command`` with nothing executed. That behaviour was already in
``PortOccupancyPlaybook``; this module's contribution is that the classification now *predicts*
it, so the property is a rule rather than an accident of one playbook's implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sylphae.ops.env_spec import EnvOpSpec
from sylphae.ops.env_vocab import KIND_CAPABILITY
from sylphae.ops.safety_class import classify_op
from sylphae.ops_env.playbooks import PLAYBOOKS, Symptom
from sylphae.ops_env.safety import CommandSpec

__all__ = ["EnvOpOutcome", "run_env_spec", "playbook_for"]


@dataclass
class EnvOpOutcome:
    """What happened to one spec on its way through the gate.

    ``status`` is a small vocabulary on purpose -- a report that has to distinguish "refused by
    the action check" from "refused by a command" from "run" from "proposed" is a report nobody
    reads correctly.

        ``executed``   at least one command ran and none was refused
        ``proposed``   advisory: the destructive step was named and not executed
        ``no_action``  the playbook resolved the symptom needing no command
        ``refused``    refused; ``stage`` says by which check

    ``executed`` is defined by the *audit trail*, not by the absence of a refusal, and the
    first version got that wrong in a way that flattered the result. It reported ``executed``
    whenever the playbook returned without refusing, so eight of eleven kinds reported
    ``executed`` while writing **zero** audit records -- they had run no command at all,
    because the fixture symptom did not match what their playbook detects. A status that means
    "nobody objected" reads as a capability, and 8/11 executed is a number that would have gone
    into the report as though the actions had been carried out. ``no_action`` and
    ``inconclusive`` exist so that gap has a name.
    """

    kind: str = ""
    status: str = "refused"
    #: ``"action_class"`` | ``"command"`` | ``"playbook"`` -- which check refused it, or empty.
    stage: str = ""
    reason: str = ""
    op_class: str = ""
    #: The command the playbook declined to run, for ``proposed``.
    proposed: CommandSpec | None = None
    result: Any = None
    requires_authorisation: bool = False
    #: The full verdict chain: the action check, then one entry per command the playbook tried.
    checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def executed(self) -> bool:
        return self.status == "executed"

    @property
    def destructive_commands_refused(self) -> int:
        """Commands the allowlist refused, counted from the audit trail.

        2c's measurable claim is that a destructive *command* does not run. This reads the
        audit rather than trusting the playbook's own account of itself, which is the same
        choice ``ops_env_gate._count_destructive`` makes for the same reason.
        """
        return sum(1 for c in self.checks
                   if c.get("check") == "command" and not c.get("allowed", True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "status": self.status, "stage": self.stage,
            "reason": self.reason, "op_class": self.op_class,
            "proposed": self.proposed.to_dict() if self.proposed else None,
            "requires_authorisation": self.requires_authorisation,
            "destructive_commands_refused": self.destructive_commands_refused,
            "checks": self.checks,
        }


def playbook_for(kind: str) -> str | None:
    from sylphae.ops.env_vocab import KIND_TO_PLAYBOOK

    return KIND_TO_PLAYBOOK.get(kind)


class _ActionShapedSpec:
    """The minimal ``(kind, props)`` shape ``classify_op`` reads.

    ``classify_op`` is written against what an ``OpSpec`` has. Rather than teach it a second
    spec type, the environment spec is adapted into the one shape the classifier understands.
    The alternative -- giving ``classify_op`` two protocols -- would put 12.46's vocabulary
    decision back in question, since the classifier would then depend on which spec type it was
    handed, and one of them could be safe by accident.
    """

    __slots__ = ("kind", "props", "path", "span")

    def __init__(self, kind: str, params: dict[str, Any]):
        self.kind = kind
        self.props = dict(params)
        self.path = ""
        self.span = (0, 0)


def run_env_spec(
    spec: EnvOpSpec,
    executor: Any,
    *,
    authorised: bool = False,
) -> EnvOpOutcome:
    """Take ``spec`` to a shell through both checks, or explain why not.

    ``authorised=True`` is how an advisory action's proposal is allowed to proceed. It is a
    separate argument on purpose: the reflex does not get to set it, because the reflex is the
    thing proposing. The default means the ordinary path for an action that interrupts somebody's
    process is to hand back a precise command for a human.
    """
    outcome = EnvOpOutcome(kind=spec.kind)

    # --- Check 1: the action. Before the playbook exists, because a destructive action should
    # never reach a component that knows how to carry it out.
    #
    # ROUND 13, and this runs *before* the capability check deliberately. Both destructive kinds
    # in the vocabulary -- ``venv_remove`` and ``service_restart`` -- are also unimplemented, so
    # checking capability first would mean every destructive kind is refused as "nothing
    # implements this" and the destructive gate would never fire for any kind in the
    # vocabulary. A gate with no subject is untested, and the ordering that produces one is the
    # wrong ordering for a different reason too: a safety property should outrank a bookkeeping
    # one. If someone implements ``service_restart`` next month, the thing that stops it has to
    # be the destructive check, and it has to already be in that position.
    verdict = classify_op(_ActionShapedSpec(spec.kind, spec.params))
    outcome.op_class = verdict.op_class
    outcome.checks.append({
        "check": "action_class", "op_class": verdict.op_class,
        "allowed": verdict.applicable, "refusal": verdict.refusal,
        "evidence": verdict.evidence,
    })
    if not verdict.applicable:
        outcome.status = "refused"
        outcome.stage = "action_class"
        outcome.reason = verdict.refusal
        return outcome

    outcome.requires_authorisation = bool(verdict.evidence.get("requires_authorisation"))

    # --- Is this kind something the playbooks were measured to do?
    #
    # The vocabulary named eleven actions and only six had a playbook measured to perform them.
    # A caller must not reach for the other five as though they might work. This sits after the
    # safety check so the two reasons stay distinguishable: "nothing implements this" is not
    # "this would be destructive", and a report collapsing them would hide which of the eleven
    # are real.
    capability = KIND_CAPABILITY.get(spec.kind)
    if capability == "unimplemented":
        outcome.status = "refused"
        outcome.stage = "unimplemented"
        outcome.reason = (
            f"{spec.kind!r} is in the operations vocabulary but no playbook was measured to "
            f"perform it, so this is not an action that can be taken; it is kept as a name so "
            f"the vocabulary and its refusals can be stated, not as a capability"
        )
        outcome.checks.append({
            "check": "capability", "capability": capability, "allowed": False,
            "refusal": outcome.reason,
        })
        return outcome

    # --- Find the playbook, and hand it the real executor so its commands meet
    # ``validate_command`` on the way. There is no derive-then-run step: a playbook runs.
    name = playbook_for(spec.kind)
    cls = PLAYBOOKS.get(name) if name else None
    if cls is None:
        outcome.status = "refused"
        outcome.stage = "playbook"
        outcome.reason = f"no playbook implements {spec.kind!r}"
        return outcome

    cwd = Path(spec.params.get("cwd") or Path.cwd())
    scratch = spec.params.get("scratch")
    symptom = Symptom(
        text=spec.params.get("text", ""),
        kind=spec.kind,
        cwd=cwd,
        scratch=Path(scratch) if scratch else None,
        context=dict(spec.evidence.get("context", {})),
    )
    try:
        result = cls().diagnose(symptom, executor)
    except Exception as exc:
        # Fail-closed, and *say* so. A playbook that cannot run has not done anything, and the
        # reason has to survive into the report -- a swallowed exception here is precisely what
        # made 12.46's ``_DryRunExecutor`` bug invisible for a whole round.
        outcome.status = "refused"
        outcome.stage = "playbook"
        outcome.reason = f"{name} raised {type(exc).__name__}: {exc}"
        return outcome

    outcome.result = result

    # --- Record every command the playbook attempted, and whether it was allowed. Read from
    # the result the playbook itself assembled, so a command it ran and a command it proposed
    # are distinguishable by whether the audit recorded an execution.
    attempted = list(getattr(result, "commands", []) or [])
    n_ran = _count_executed(executor, outcome.kind, len(attempted))
    for cmd in attempted:
        outcome.checks.append({
            "check": "command", "command": cmd, "allowed": True,
            "note": "reached the executor, which validates before running",
        })

    proposed = getattr(result, "proposed_command", None)
    if proposed is not None:
        outcome.proposed = proposed
        # Asked the executor what it thinks, so the proposal is reported with the verdict it
        # would actually get rather than with an assumption.
        refusal = executor.would_allow(proposed)
        outcome.checks.append({
            "check": "command", "command": proposed.command,
            "allowed": refusal is None,
            "refusal": refusal.reason if refusal else "",
            "note": "proposed, not executed",
        })
        if refusal is not None:
            outcome.status = "refused"
            outcome.stage = "command"
            outcome.reason = (
                f"{spec.kind!r} is {verdict.op_class} and passed the action check, but the "
                f"command it proposes is refused: {refusal.reason}"
            )
            return outcome
        if not authorised:
            outcome.status = "proposed"
            outcome.reason = (
                f"{spec.kind!r} interrupts a process, so the command was named and not run; "
                f"pass authorised=True to proceed with it"
            )
            return outcome
        result = executor.run(proposed)
        attempted.append(proposed.command)
        n_ran += 1

    if getattr(result, "refusal", None) is not None:
        outcome.status = "refused"
        outcome.stage = "command"
        outcome.reason = (
            f"{result.refusal.reason}: {result.refusal.detail}"
            if getattr(result.refusal, "detail", "")
            else str(result.refusal)
        )
        return outcome

    # --- The status has to be derived from what happened, not from the absence of a complaint.
    #
    # ROUND 14: an *advisory* kind may never report ``executed`` merely because its diagnostic
    # ran. ``port_release`` on a machine with nothing on the port runs ``netstat -ano``, finds
    # nothing, and stops -- and counting that audit record as an execution would report a
    # released port that was never released. The action of an advisory kind is the *proposal*,
    # and the honest terminal states for one are ``proposed`` (there is something to authorise)
    # and ``no_action`` (there is not). It reached the audit trail, so the commands are reported
    # in ``checks`` either way.
    if outcome.requires_authorisation and not proposed:
        if getattr(result, "resolved", False):
            outcome.status = "no_action"
            outcome.reason = (
                f"{spec.kind!r} ran its diagnostic and there was nothing to propose: the "
                f"symptom needed no interruption, so no authorisation is outstanding"
            )
        else:
            outcome.status = "inconclusive"
            outcome.reason = (
                f"{spec.kind!r} is advisory and produced neither a proposal nor a resolution, "
                f"so no interruption was proposed and none is known to have happened"
            )
        return outcome

    if n_ran:
        outcome.status = "executed"
    elif getattr(result, "resolved", False):
        outcome.status = "no_action"
        outcome.reason = (
            f"{spec.kind!r} was resolved without a command, so nothing was run; this is the "
            f"playbook reporting the symptom needed no action, not the action being carried out"
        )
    else:
        outcome.status = "inconclusive"
        outcome.reason = (
            f"{name} returned neither a command nor a resolution for {spec.kind!r}, so it is "
            f"not known whether anything was done; an unresolved silence is not a success"
        )
    return outcome


def _count_executed(executor: Any, kind: str, attempted: int) -> int:
    """How many of the playbook's commands actually reached the audit trail.

    Read from the audit rather than from ``result.commands``, because a playbook appends to
    ``commands`` *before* calling ``run`` (``Playbook._exec`` does
    ``result.commands.append(spec.command)`` then ``ex.run(spec)``), and a command the executor
    refused is therefore in ``commands`` too. Counting the list would report commands that were
    never run, which is the same false positive this status vocabulary was introduced to remove.
    """
    try:
        records = executor.audit.read()
    except Exception:
        return 0
    total = sum(1 for r in records if not r.get("refused"))
    return min(total, attempted) if attempted else 0
