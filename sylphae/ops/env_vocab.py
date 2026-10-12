"""A sibling vocabulary for operational actions, kept separate from ``OP_KINDS``.

ROUND 12, and the reason this file exists rather than four more entries in
``sylphae/ops/vocab.py``. That module defines its own contract:

    only operations that are *detectable from a diff* and *deterministically executable*
    are in it

and adds that a list index *is* the op id, which the corpus and the model's op head both index
into. A service restart is neither diff-detectable nor a file edit, so appending it would not
extend that vocabulary -- it would break its stated meaning. It would also grow the model's
label space from 26 to 31, moving the op head's distribution, which means ``op_accuracy``
(0.8892) and ``op_macro_f1`` (0.2049) would both have to be re-measured before either could be
quoted again. 12.46 found this by trying to construct the spec and reading the error.

**So this is a sibling, and the separation is load-bearing rather than tidiness.** The op head's
26 labels keep meaning "a change I can see in a diff". The eight labels here mean "something
that has to be done to a machine". Neither list has to stretch to cover the other, and
neither one's ids are disturbed.

**What is deliberately absent.** There is no "restart the box", "drop the database", "rotate
credentials" or "edit the firewall". This is not caution for its own sake: every one of those
would have to be *executed* to be useful, and an action that can only be exercised by performing
it is an action whose safety is a matter of someone's restraint rather than of code. The
vocabulary is sized to what the executor can do under a gate, and the gate is the thing that
makes the difference between an agent and a hazard.

**The three-way classification is not in this file.** It lives in
:mod:`sylphae.ops.safety_class`, and this module names the kinds that classification exists in
service of. Keeping the classes there rather than here means the code vocabulary and the
operations vocabulary are judged by the same code, so "reversible" means the same thing in both
lists -- which is the property that lets one gate sit in front of both.
"""

from __future__ import annotations

__all__ = [
    "ENV_OP_KINDS",
    "ENV_OPS_REQUIRING_AUTHORISATION",
    "ENV_OPS_NEVER_EXECUTED",
    "KIND_TO_PLAYBOOK",
]

#: Frozen, ordered. Index in this list is *this list's* id and nothing else's -- no corpus, no
#: model head, and no relationship to ``OP_KINDS`` indices. Ids start at 0 here precisely so
#: nobody later assumes they continue the op numbering.
ENV_OP_KINDS: tuple[str, ...] = (
    "venv_create",
    "venv_remove",
    "dependency_install",
    "dependency_resolve_conflict",
    "port_release",
    "service_stop",
    "service_restart",
    "test_selection",
    "test_retry_once",
    "lint_autofix",
    "import_locate",
)

#: Kinds this vocabulary will not contain, recorded so their absence is a decision rather than
#: an oversight. Each would be ``destructive`` and each would have to be executed to be worth
#: having.
ENV_OPS_NEVER_EXECUTED: tuple[str, ...] = (
    "host_reboot",
    "database_drop",
    "credential_rotate",
    "firewall_edit",
    "filesystem_format",
)

#: Kinds whose effect requires a human to authorise before the proposal runs. The classification
#: itself lives in ``safety_class``; this is the operational reading of the same set, kept here
#: so a caller asking "what needs authorisation in the operations vocabulary" does not have to
#: import the code vocabulary to find out.
ENV_OPS_REQUIRING_AUTHORISATION: tuple[str, ...] = ("port_release", "service_stop")

#: Which playbook knows how to carry out each kind. This is the one place the two vocabularies
#: are allowed to meet: the operations vocabulary names the *action*, the playbook knows the
#: *commands*. Deriving commands from an action here rather than in the vocabulary is what lets
#: a kind stay declarative while the executor stays the only component that touches a shell.
KIND_TO_PLAYBOOK: dict[str, str] = {
    "venv_create": "venv",
    "venv_remove": "venv",
    "dependency_install": "missing-dependency",
    "dependency_resolve_conflict": "pip-conflict",
    "port_release": "port-occupied",
    "service_stop": "port-occupied",
    "service_restart": "port-occupied",
    "test_selection": "affected-tests",
    "test_retry_once": "flake-retry",
    "lint_autofix": "lint-fix",
    "import_locate": "import-error-locator",
}

#: What each kind was *measured* to do, as of 2026-10-02.
#:
#: MEASURED by ``scratch/probe_material_map.py``, which runs all eight playbooks against
#: artefacts shaped the way each reads and records the commands that reach the audit trail. The
#: eleven kinds above were named in 12.47 *before* that measurement existed, and the measurement
#: did not support all of them. Four values, and the distinction between the last two is the one
#: that matters:
#:
#:   ``act``           a playbook runs commands that carry out this kind
#:   ``propose``       a playbook derives a command and hands it over unexecuted
#:   ``diagnose``      a playbook concludes without acting -- and that is it working, not failing
#:   ``unimplemented`` nothing performs this. The kind is named but has no behaviour behind it
#:
#: ``diagnose`` and ``unimplemented`` are different in kind, not degree. ``missing-dependency``
#: declining to install a distribution it cannot import is a correct decision -- installing would
#: be guessing -- so ``dependency_install`` is a diagnosis. ``service_stop`` maps to the same
#: registry entry as ``port_release``, but ``port-occupied`` does not stop services, so it is
#: unimplemented and the mapping is simply wrong.
#:
#: The first version of this census measured capability **per playbook** and inherited it for
#: every kind mapped to that playbook, which reported 9 of 11 as able to act. That was false:
#: ``port-occupied`` runs ``netstat -ano``, so ``service_stop``, ``service_restart`` and
#: ``venv_remove`` all inherited a capability from a playbook that does none of their work. **A
#: capability measured at one level cannot be inherited at another** -- the same mistake
#: ``by_shape`` records in 12.35 for a benchmark that counted a scenario's outcome at a
#: different granularity than it counted its evidence. 6 of 11 is the number that is true.
#:
#: The three ``unimplemented`` kinds are kept rather than deleted, for one reason:
#: ``service_restart`` is the subject the destructive gate refuses, and a gate with no subject is
#: untested. They are marked so that nothing can read them as performable.
KIND_CAPABILITY: dict[str, str] = {
    "venv_create": "act",
    "venv_remove": "unimplemented",
    "dependency_install": "diagnose",
    "dependency_resolve_conflict": "act",
    "port_release": "propose",
    "service_stop": "unimplemented",
    "service_restart": "unimplemented",
    "test_selection": "propose",
    "test_retry_once": "act",
    "lint_autofix": "act",
    "import_locate": "diagnose",
}

#: The command each kind was measured to produce, as evidence rather than as documentation.
#: Empty for kinds that produced none, which is how ``unimplemented`` was arrived at.
KIND_MEASURED_COMMAND: dict[str, str] = {
    "venv_create": "python -m venv <path>",
    "dependency_resolve_conflict": "pip install --dry-run -r requirements.txt",
    "port_release": "netstat -ano  (then a proposed kill)",
    "test_selection": "pytest --co -q <file>   [proposed, not run]",
    "test_retry_once": "pytest -q --tb=short <file>  (x2)",
    "lint_autofix": "ruff format --no-cache <file>; ruff check --no-cache <file>",
}

__all__ += ["KIND_CAPABILITY", "KIND_MEASURED_COMMAND", "performable_kinds"]


def performable_kinds() -> tuple[str, ...]:
    """The kinds a playbook was measured to act on or propose. Everything else is not a claim."""
    return tuple(sorted(k for k in ENV_OP_KINDS
                        if KIND_CAPABILITY.get(k) in ("act", "propose")))

