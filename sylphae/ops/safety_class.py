"""Classify an ``OpSpec`` by whether applying it can be undone, before anything applies it.

ROUND 11, and the precondition 12.45 established: 2b wants the ops tasks solved "by reflex",
2c wants destructive operations to escalate rather than execute, and 12.45 measured that
neither exists on the ``OpSpec`` path. ``sylphae/ops/apply.py`` raises ``OpApplyError`` for a
malformed spec, which is a correctness check; it has no notion of an edit being *dangerous*,
and ``grep -ri destructive sylphae/ops/`` returns nothing. Meanwhile the only place in the
repository that classifies destruction is ``ops_env/safety.py``, which matches command
argument strings.

**So adding operations ops to the vocabulary right now would be the unsafe order.** An
``OpSpec`` of kind ``service_restart`` would be built, the reflex would report it as a
confident patch, and ``apply_op`` would carry it out with nothing between it and the machine
but the consumer's own caution. 2c exists to prevent exactly that. This module is what has to
be in place first, so that the vocabulary question becomes "which ops are *admissible*" rather
than "which ops exist".

**Where the line is drawn.** Three classes, and the middle one is the whole point:

``reversible``
    The edit can be undone from what is on disk. A literal substitution, a rename, an
    inserted guard: the previous text is either recorded in the spec or recoverable by
    re-reading the file, and a wrong one is a small diff. These apply.

``advisory``
    The edit *proposes* something destructive and carries the proposal, but does not perform
    it. A ``port_release`` that names the PID and the port and escalates is advisory even
    though ``kill`` is in its description -- the difference between describing a kill and
    performing one is the difference between a tool and a loaded gun, and it is checkable
    rather than a matter of trust. These apply, and the result records that a human is
    required before the proposal runs.

``destructive``
    Applying the spec performs an action whose effect outlives the process: killing a
    process someone else owns, removing a directory tree, overwriting state that has no
    undo. These are refused. Not "warned about" -- refused, with a reason a caller can act
    on, because a warning in a field a caller might not read is the same as no warning.

**What this deliberately does not do.** It does not decide whether an operation is a good
idea. ``rm -rf /tmp/build`` and ``rm -rf /`` are the same verb and this module treats them
alike, because deciding between them needs a judgement about the path's importance that
belongs to the caller, which receives the refusal and the spec and can decide. Overreach
here would be a tool that blocks the ordinary case to be safe against the rare one, which is
the failure mode 12.35 documented for ``clean`` reading only ``syntax_valid``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "OpClass",
    "OpSafety",
    "classify_op",
    "OPS_CLASSES",
]


@dataclass(frozen=True)
class OpClass:
    """The classification of one op kind, and why."""

    #: ``reversible`` | ``advisory`` | ``destructive``
    name: str
    note: str


#: Classification of every kind in ``sylphae.model.collate.OP_KINDS``.
#:
#: MEASURED 2026-10-02, and the interesting ones are not the reversible ones. Reading the
#: vocabulary against "can this be undone from what is on disk":
#:
#:   The *code* ops are all reversible. A substitution, a rename, a reindent, a delete of a
#:   line -- each is a small diff whose previous text is either in the spec or re-readable
#:   from the file. So the classification is not "is it risky" but "is the effect confined to
#:   a file I can show the user".
#:
#:   ``delete_statement`` deserves its own line because it is the one code op whose effect is
#:   not a substitution: it removes something. It is still reversible -- the text is on disk
#:   until written, and the spec records the span -- but it is the shape a reapply-shaped
#:   mistake would take, so a caller that wants to be careful about deletions can ask.
#:
#:   ``add_import`` / ``remove_import`` are reversible like any other code edit. They appear
#:   here because a substring search for "import" in 12.45's test matched them and it was
#:   worth being explicit that they are code edits and not operations-class ops.
OPS_CLASSES: dict[str, OpClass] = {}


def _code_ops() -> tuple[str, ...]:
    # From ``ops.vocab``, the defining module, rather than via a re-export: a classification
    # that depends on somebody else's forwarding stays correct only as long as that
    # forwarding does, and the vocabulary is exactly the thing that grows over time.
    from sylphae.ops.vocab import OP_KINDS

    return tuple(OP_KINDS)


for _kind in _code_ops():
    OPS_CLASSES[_kind] = OpClass(
        "reversible",
        "a source edit confined to a file whose previous text the spec records",
    )

#: The one code op whose effect is a removal rather than a substitution. Kept separate so a
#: caller can be careful about deletions without that becoming the default posture.
OPS_CLASSES["delete_statement"] = OpClass(
    "reversible",
    "removes a statement; reversible because the span and its text are in the spec, but it "
    "is the shape a mistaken reapply would take",
)

#: The operations vocabulary. A *sibling* of ``OP_KINDS``, not an extension of it: 12.46
#: established that ``OP_KINDS`` is defined as the operations "detectable from a diff", so a
#: service restart does not belong in it -- see :mod:`sylphae.ops.env_vocab`.
#:
#: MEASURED 2026-10-02. Reading the eleven kinds against "can this be undone, and who does the
#: undoing", and the interesting split is not code-vs-operations but who-holds-the-state:
#:
#:   ``reversible``  7 -- ``venv_create`` and ``dependency_install`` are additive and undoable
#:                        by deleting what was added; ``test_selection``, ``test_retry_once``
#:                        and ``import_locate`` change nothing at all, they observe; and
#:                        ``lint_autofix`` and ``dependency_resolve_conflict`` are diff-able
#:                        edits, which is the property ``OP_KINDS`` exists to guarantee.
#:   ``advisory``    2 -- name the process or the service, propose the action.
#:   ``destructive`` 2 -- ``venv_remove`` (a tree has no undo) and ``service_restart``
#:                        (interrupts a running service, which is the user's state and not
#:                        something the agent can restore).
#:
#: The two destructive members are the whole reason this list is worth having. Everything else
#: can be allowed through, and the argument for allowing it is not that it is safe but that it
#: is *undoable* -- which is a different claim, and one the caller can check for itself.
OPS_CLASSES.update({
    "venv_create": OpClass(
        "reversible",
        "creates a virtualenv; additive, and removable by deleting the directory",
    ),
    "venv_remove": OpClass(
        "destructive",
        "removes a directory tree; there is no undo for a recursive delete",
    ),
    "dependency_install": OpClass(
        "reversible",
        "installs a package; additive and removable by uninstalling, which only holds if the "
        "version that was present is recorded first",
    ),
    "dependency_resolve_conflict": OpClass(
        "reversible",
        "resolves a dependency conflict into a requirements change plus an install; both "
        "halves are diff-able or additive",
    ),
    "port_release": OpClass(
        "advisory",
        "names the process holding the port and proposes the release; the kill is a proposal "
        "the caller must authorise",
    ),
    "service_stop": OpClass(
        "advisory",
        "names the service and proposes the stop",
    ),
    "service_restart": OpClass(
        "destructive",
        "interrupts a running service, which is somebody else's state and which the agent "
        "cannot put back",
    ),
    "test_selection": OpClass(
        "reversible",
        "selects and runs tests; observes the system and changes nothing",
    ),
    "test_retry_once": OpClass(
        "reversible",
        "re-runs a test once; observes, changes nothing",
    ),
    "lint_autofix": OpClass(
        "reversible",
        "applies a formatter or a linter's own fix; the result is a diff, which is the "
        "property OP_KINDS is built around",
    ),
    "import_locate": OpClass(
        "reversible",
        "locates an import failure; observes, changes nothing",
    ),
})


@dataclass
class OpSafety:
    """The verdict on one spec, with the reason attached.

    ``refusal`` is empty exactly when the spec may be applied. A caller that checks
    ``if not verdict.refusal`` cannot accidentally read an absent field as a pass, which is
    the 12.35 failure inverted -- there a missing check read as a pass; here a missing reason
    cannot read as a permit.
    """

    kind: str = ""
    op_class: str = "unknown"
    note: str = ""
    #: Why it may not be applied. Empty means it may.
    refusal: str = field(default_factory=str)
    #: Extra evidence for the caller, e.g. the paths a delete would touch.
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def applicable(self) -> bool:
        return not self.refusal

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "op_class": self.op_class, "note": self.note,
            "applicable": self.applicable, "refusal": self.refusal,
            "evidence": self.evidence,
        }


#: Props whose presence means a spec reaches outside the file it names.
#:
#: MEASURED 2026-10-02 by reading the prop names the builders in ``sylphae/agent/reflex.py``
#: actually set, rather than guessing: ``old_text``/``new_text`` for a line rewrite,
#: ``old``/``new`` for a substitution, ``text``/``line`` for an insert. A spec carrying any
#: of these is a file edit by construction.
_FILE_EDIT_PROPS = ("new_text", "text", "new", "insert_before", "line")

#: Props that would mean a spec is doing something other than editing a file. None are set by
#: any current builder, which is why the classification can be a closed set today; the tuple
#: exists so a future builder that sets one of them fails a test rather than sailing through.
_NON_FILE_PROPS = ("command", "argv", "path_to_delete", "target_path", "pid", "port")


def classify_op(spec: Any) -> OpSafety:
    """Whether ``spec`` may be applied, and under what class.

    Unknown kinds are **not** permitted. A kind that is in neither ``OPS_CLASSES`` nor
    reachable from the op vocabulary is either a typo or an op added without deciding what
    class it belongs to, and both should stop the write rather than pass through -- the
    alternative is a vocabulary that grows by accretion until every new member is
    unrestricted by default, which is how a safety mechanism stops being one.
    """
    kind = str(getattr(spec, "kind", "") or "")
    props = dict(getattr(spec, "props", None) or {})

    declared = OPS_CLASSES.get(kind)
    if declared is None:
        return OpSafety(
            kind=kind, op_class="unknown",
            refusal=(
                f"{kind!r} is not in the classified op vocabulary, so it is not known "
                f"whether applying it is reversible; refusing rather than assuming"
            ),
            evidence={"known_kinds": len(OPS_CLASSES)},
        )

    verdict = OpSafety(kind=kind, op_class=declared.name, note=declared.note)

    # A props key that names a filesystem target, a process or a command makes the spec
    # reach outside its file whatever the kind says. Recorded as evidence and *then* the class
    # decides -- there is no early return here, because the first version of this function had
    # one and it silently skipped the destructive refusal for exactly the specs most likely to
    # be dangerous: a ``service_restart`` carrying ``command`` returned from this block with
    # ``refusal=""``. A safety check that a branch can skip is not a safety check, and the
    # branch that skips it is always the one handling the unusual case.
    outside = sorted(k for k in props if k in _NON_FILE_PROPS)
    if outside:
        verdict.evidence["outside_file_props"] = outside
        if declared.name == "reversible":
            # The classification's promise was "confined to a file". A prop that names a
            # target outside the file breaks that promise, and the honest response is to
            # refuse rather than to reason about a prop the classification has never seen.
            verdict.op_class = "destructive"
            verdict.refusal = (
                f"{kind!r} is classified reversible but carries {outside}, which names a "
                f"target outside the file being edited; the classification does not cover "
                f"what that prop does, so it is refused"
            )
            return verdict
        # Advisory stays advisory: ``port_release`` genuinely has to name a PID and a port,
        # and refusing it would be overreach that blocks the correct proposal. The
        # authorisation requirement is still attached below.

    if declared.name == "destructive":
        verdict.refusal = (
            f"{kind!r} is destructive: {declared.note}. Refused rather than warned, because "
            f"a warning in a field the caller may not read is the same as no warning"
        )
    elif declared.name == "advisory":
        # Advisory may apply, and the record says a human is required before the proposal
        # runs. Carried in evidence so a report cannot show "applied" without it.
        verdict.evidence["requires_authorisation"] = True
        verdict.evidence["advisory_note"] = declared.note
    return verdict


def destructive_kinds() -> tuple[str, ...]:
    """The kinds that may never be applied. Exposed for the report and for tests."""
    return tuple(sorted(k for k, c in OPS_CLASSES.items() if c.name == "destructive"))


def advisory_kinds() -> tuple[str, ...]:
    return tuple(sorted(k for k, c in OPS_CLASSES.items() if c.name == "advisory"))
