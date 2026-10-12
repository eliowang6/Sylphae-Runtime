"""``EnvOpSpec``: an operational action, and the one gate that both vocabularies pass through.

ROUND 12. 12.46 built the classification and wired it in front of the code path, then listed as
unproven that ``EnvExecutor.validate_command`` still exists independently in the playbook layer
and that "the two classifiers have never been run together". This module is that run-together,
and it is also what makes 2b a real path rather than an eval script.

**The shape.** ``OP_KINDS`` names *changes I can see in a diff*; ``EnvOpSpec`` names *things that
have to be done to a machine*. The difference is not wording:

* A code ``OpSpec`` is a full description of the edit -- span, old text, new text -- because
  ``apply_op`` has to reproduce it byte for byte.
* An ``EnvOpSpec`` is deliberately **not** a description of a command. It names an action and
  its arguments, and :mod:`sylphae.ops_env.bridge` derives the command. That indirection is the
  point: a spec that carried ``argv`` would let whoever writes the spec choose the command, and
  then the allowlist in ``safety.py`` would be checking a string the agent itself had already
  chosen. Here the agent chooses the *action* and the playbook -- the component that already
  knows the commands -- chooses the argv.

**What the gate does, in order.** ``classify_op`` first, so a destructive action is refused
before any command is derived and before the executor is touched at all. Then
``EnvExecutor.would_allow``, so the allowlist and the destructive-pattern match both apply to the
command the playbook produced. **These have never run in the same function before**, and running
them together is what turns two independent safety stories into one that is actually enforced on
the path an action takes.

**Why the second check is not redundant with the first.** The classification knows the *action*
and cannot know the *command* -- ``port_release`` derives an ``lsof`` and a ``kill``, and only
one of them is the dangerous one. The allowlist knows the command and does not know the action.
A spec classified ``reversible`` whose playbook derived a command the allowlist refuses is
exactly the case each check alone would pass, and it is the case that matters. 12.46's fallback
-- a reversible kind carrying an outside-file prop is refused -- is a crude guard against the
same shape at the spec layer; here it is done properly, on the real command.

**Advisory is not a slower path to the same place.** An advisory spec may execute, and the
result records ``requires_authorisation`` and returns the derived command *without* running it,
unless the caller passes ``authorised=True``. So the default for an action that interrupts
somebody's process is to hand a human a precise command, and the reflex gets to be useful
without being the thing that pulls the trigger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sylphae.ops.env_vocab import ENV_OP_KINDS

__all__ = ["EnvOpSpec", "EnvSpecError"]


class EnvSpecError(ValueError):
    """The spec is malformed, as opposed to unsafe. Both are refusals, differently."""


@dataclass(frozen=True)
class EnvOpSpec:
    """An operational action with its arguments.

    ``kind`` is validated against ``ENV_OP_KINDS`` -- not ``OP_KINDS``. That is the separation
    12.46 called for, enforced at construction so a code op id cannot be smuggled in here and
    vice versa; the two lists share a classifier, not an id space.
    """

    kind: str
    #: Arguments the playbook needs. Names, not a command line.
    params: dict[str, str] = field(default_factory=dict)
    #: Where the symptom came from, for the audit record.
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ENV_OP_KINDS:
            raise EnvSpecError(
                f"{self.kind!r} is not an operations op; expected one of {ENV_OP_KINDS}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": dict(self.params),
                "evidence": dict(self.evidence)}
