"""``sylphae.ops`` — the edit-operation vocabulary.

Detect a structured operation from a real ``(before, after)`` pair, and replay it
deterministically::

    from sylphae.ops import detect_op, detect_sequence, apply_op, apply_sequence

    spec = detect_op(old, new, "src/flask/app.py")
    if spec is not None and spec.kind != "freeform_patch":
        assert apply_op(old, spec) == new      # guaranteed by detect_op

    seq = detect_sequence(old, new, "src/flask/app.py")
    if seq and seq[0].kind != "freeform_patch":
        assert apply_sequence(old, seq) == new  # guaranteed by detect_sequence

The invariant that matters: for any ``(old, new)`` where :func:`detect_op` returns
a spec whose kind is not ``freeform_patch``, ``apply_op(old, spec) == new``
exactly, character for character.  The same holds for :func:`detect_sequence` and
:func:`apply_sequence`, with the sequence applied back-to-front (see
:func:`sylphae.ops.detect.detect_sequence`).
"""

from .apply import apply_op, apply_sequence
from .detect import detect_op, detect_sequence
from .spec import OpApplyError, OpSpec
from .vocab import (
    DELETION_KINDS,
    EXECUTABLE_KINDS,
    FREE_FORM_KIND,
    GENERATIVE_OPS,
    ID_TO_OP,
    INSERTION_KINDS,
    N_OPS,
    NEVER_DETECTED,
    OP_KINDS,
    OP_TO_ID,
    PARAMETERIZED_OPS,
    REPLACEMENT_KINDS,
    WRAP_KINDS,
    kind_id,
)

__all__ = [
    "OP_KINDS",
    "OP_TO_ID",
    "ID_TO_OP",
    "N_OPS",
    "FREE_FORM_KIND",
    "NEVER_DETECTED",
    "EXECUTABLE_KINDS",
    "INSERTION_KINDS",
    "DELETION_KINDS",
    "REPLACEMENT_KINDS",
    "WRAP_KINDS",
    "PARAMETERIZED_OPS",
    "GENERATIVE_OPS",
    "kind_id",
    "OpSpec",
    "OpApplyError",
    "detect_op",
    "detect_sequence",
    "apply_op",
    "apply_sequence",
]
