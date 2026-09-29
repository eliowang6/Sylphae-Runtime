"""Turning a model's param-head emission into an executable ``OpSpec``.

This is the read-side mirror of :mod:`sylphae.ops.params`.  That module answers
"what single value should the head emit for this row?"; this one answers "given
what it emitted, what edit does that mean?" -- the same question asked from the
other end, so the two must never disagree about which prop key carries the
copyable value.

They share one definition rather than two mirrored copies.  The mirror used to
be duplicated inside the evaluation harness, because importing it from the model
side would have pulled ``torch`` onto the eval path; :mod:`sylphae.ops.params` is
stdlib-only, so the duplication is no longer buying anything and now costs the
only thing that matters here -- a decoder that silently disagrees with the
labels it is scored against.

**This is an interpretation of the frozen contract, not a definition from it.**
``INTERFACES.md`` fixes ``OpSpec`` and the param head's *shape*, but never the
wire format of the param tokens.  Demanding one format is wrong: a small
decoder cannot reliably emit well-formed JSON props (measured -- held-out
payloads came back undecodable), yet it *can* emit a single short value.  So the
decoder is tolerant of both:

* if ``param_text`` parses as a JSON object of string values, it is used as the
  props directly;
* otherwise ``param_text`` is taken as the new *value* and placed onto the gold
  op's props at the key :func:`sylphae.ops.params.param_target_key` chose.  When
  that key is a value key, the replacement chunk is then **derived from the
  emission** by substituting it for the recorded old value inside ``old_text``;
  when it is a chunk key the emission *is* the payload.

Deriving the chunk is not optional.  If the gold ``new_text`` were left in place,
``apply_op`` would ignore the emission entirely and the edit would be scored
100% correct by construction -- a metric that flatters the model by measuring
the label.  When the substitution is undefined the decoder refuses (``None``)
rather than silently scoring the gold chunk.

Two consequences, stated rather than hidden:

* the ``span`` comes from the **gold** op (there is no span head), so this
  measures target + op-kind + value, not span prediction;
* a predicted ``op_kind`` that disagrees with the gold will not round-trip and
  ``apply_op`` raises -- correctly counted as "did not apply."
"""

from __future__ import annotations

import json
from typing import Any, Callable

from ..ops.params import (
    _VALUE_PAIRS,
    param_target_key,
    substitute_value,
)

__all__ = ["default_decode_spec", "DecodeError"]

#: Reverse map from a value key to the old-value key it is derived against.
_VALUE_KEY_TO_OLD: dict[str, str] = {new: old for new, old in _VALUE_PAIRS}


class DecodeError(RuntimeError):
    """Raised when an emission cannot be turned into an ``OpSpec`` at all."""


def default_decode_spec(example: dict, op_kind: str, param_text: str, path: str) -> Any | None:
    """Turn an emitted ``(op kind, decoded params, target path)`` into an ``OpSpec``.

    Returns ``None`` when the emission is a *value* that cannot be turned back
    into a chunk -- the one case where refusing beats guessing, because the only
    alternative is to score the gold text as if the model had produced it.

    Pass this (or any compatible callable) as ``decode_fn`` to override the
    param codec.
    """
    op = example.get("op")
    if not isinstance(op, dict) or "span" not in op:
        raise ValueError(
            "example has no op.span; the span is needed to construct the emitted OpSpec "
            "(INTERFACES.md section 2 makes op an OpSpec.to_dict())"
        )
    span = op["span"]
    gold_props = op.get("props")
    props: dict[str, str] = {}
    if isinstance(gold_props, dict):
        props = {str(k): str(v) for k, v in gold_props.items()}

    text = (param_text or "").strip()
    parsed = None
    if text:
        try:
            candidate = json.loads(text)
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, dict) and all(
                isinstance(k, str) and isinstance(v, str) for k, v in candidate.items()):
            parsed = candidate
    if parsed is not None:
        props = parsed                      # the head emitted full JSON props
    else:
        chosen = param_target_key(props)
        if chosen is None:
            if not text:
                props = {}
        else:
            props[chosen] = param_text or ""  # the head emitted just the new value
            old_key = _VALUE_KEY_TO_OLD.get(chosen)
            if old_key is not None and "old_text" in props:
                # The op's applier reads ``new_text``, not the value key, so the chunk
                # has to come from the emission.  Refusing is the only honest answer
                # when it cannot.
                rebuilt = substitute_value(props, chosen, old_key, param_text or "")
                if rebuilt is None:
                    return None
                props["new_text"] = rebuilt

    OpSpec, _, _ = _resolve_ops()
    return OpSpec(kind=op_kind, path=path, span=(int(span[0]), int(span[1])), props=props)


def _resolve_ops() -> tuple[Any, Callable, type]:
    """``(OpSpec, apply_op, OpApplyError)``, imported lazily.

    Lazy so that :mod:`sylphae.model.decode` stays importable -- and therefore
    unit-testable -- before ``sylphae.ops`` has been built, which is the same
    contract the rest of the codebase keeps (INTERFACES.md section 1).
    """
    try:
        from ..ops.apply import apply_op  # noqa: PLC0415 - deliberate lazy import
        from ..ops.spec import OpApplyError, OpSpec  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - depends on a sibling workstream
        raise DecodeError(
            "sylphae.ops.apply/sylphae.ops.spec are not importable, so an emission cannot be "
            "decoded into an OpSpec. Build sylphae/ops/ first (INTERFACES.md section 1)."
        ) from exc
    return OpSpec, apply_op, OpApplyError
