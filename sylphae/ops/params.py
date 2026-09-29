"""Torch-free definitions of the parameter / confidence labels.

These predicates answer one question -- *"what single value must the parameter
head emit for this row, and can that value be read off the request?"* -- and
three separate consumers need the same answer:

* :mod:`sylphae.model.collate`, which turns it into ``L_param`` supervision,
* :mod:`sylphae.data.synth_validate`, which rejects rows that contradict it,
* :func:`sylphae.model.decode.default_decode_spec`, which is its read-side mirror.

So the definitions live here, in ``sylphae.ops`` and in pure stdlib, and all three
call them.  A validator and a training run cannot drift apart, because there is
only one function to drift from.

Why ``sylphae.ops`` and not either of the other two homes:

* They describe what a *parameterised op* means (INTERFACES.md section 1) --
  which prop key carries the copyable value, and whether the recorded chunk
  really is a substitution of it.  That is ``ops`` vocabulary, not model state.
* The model side imports ``torch`` at module scope and the data side must not,
  so the shared code cannot live in either.  ``sylphae.ops`` imports neither, which
  is what makes the sharing real instead of aspirational: the corpus validator
  keeps loading without paying for a torch import (AGENTS.md 2.2), and the
  runtime does not have to know the data package exists.

Nothing here may import torch, numpy, or anything from ``sylphae.model`` or
``sylphae.data``.
"""

from __future__ import annotations

__all__ = [
    "substitute_value",
    "param_target_key",
    "param_target_rejection_reason",
    "param_target_string",
    "param_answerable",
]


#: INTERFACES.md section 2 defines a *parameterised* op as one the model emits "by
#: choosing a short value or name (a literal, an identifier, a keyword-argument name,
#: a dict key)".  For those ops the head must be asked for the value and the chunk
#: derived.  Asking for the chunk instead contradicts the contract *and* the consumer:
#: ``sylphae.agent.reflex._build_patch`` reads a bare literal from ``param_val``, so a
#: whole-line target is unusable by the agent no matter how well the head learns it.
#: Most-specific first.
_VALUE_PAIRS: tuple[tuple[str, str], ...] = (
    ("new_value", "old_value"),
    ("new_default", "old_default"),
    ("new_name", "old_name"),
    ("new", "old"),
)

#: Fallback priority when no value pair is present -- the insert / remove family,
#: where the payload *is* the new text and nothing derives it.
#:
#: ``text`` comes first because it is the key those appliers actually read
#: (``sylphae.ops.apply._apply_insert`` / ``_apply_delete``); ``new_text`` is only the
#: payload of the *block replace* appliers, which never carry a ``text``.  Ordering
#: these the other way round is not harmless: the polyglot generator labels
#: ``insert_statement`` with both, so the head was trained on ``new_text`` while the
#: applier read ``text``, and the emission had no effect on the file at all.
_CHUNK_PROP_KEYS: tuple[str, ...] = ("text", "new_text", "value", "to")

#: Every key ``param_target_string`` may choose, values before chunks.  The eval
#: decoder mirrors this rule rather than importing it, so the eval layer keeps
#: working under the end-to-end stub world; ``tests/test_iteration4.py`` pins the two
#: together, and does so behaviourally, not just by comparing the tuples.
_PARAM_PROP_KEYS: tuple[str, ...] = (
    tuple(new for new, _old in _VALUE_PAIRS) + _CHUNK_PROP_KEYS
)


def substitute_value(
    props: dict, new_key: str, old_key: str, new_value: str
) -> str | None:
    """``old_text`` with the recorded old value replaced by ``new_value``.

    ``None`` when the substitution is not defined -- no ``old_text``, an empty old
    value, or an old value that does not occur in it.  A ``None`` means the chunk
    cannot be derived from a value, and the caller must ask for the chunk instead.
    """
    old_text = props.get("old_text")
    if not isinstance(old_text, str):
        return None
    old_value = str(props.get(old_key, ""))
    if not old_value or old_value not in old_text:
        return None
    return old_text.replace(old_value, new_value, 1)


def param_target_key(props: dict) -> str | None:
    """Which prop key the parameter head must fill, or ``None`` for "emit nothing".

    The value key wins when the example carries both the value and its old
    counterpart *and* the chunk is exactly a substitution of that value.  Requiring
    exactness here is what keeps the metric honest: the decoder derives ``new_text``
    from the emission, so a value target is only safe on examples where the gold
    chunk really is that substitution.  Everything else keeps the chunk.
    """
    if not isinstance(props, dict):
        return None
    for new_key, old_key in _VALUE_PAIRS:
        if new_key not in props or old_key not in props:
            continue
        # An op whose applier reads the value directly (rename_symbol) has no chunk
        # to rebuild and needs none.
        if "old_text" not in props:
            return new_key
        if substitute_value(props, new_key, old_key, str(props[new_key])) == props.get(
            "new_text"
        ):
            return new_key
    for key in _CHUNK_PROP_KEYS:
        if key in props:
            return key
    return None


def param_target_rejection_reason(props: dict) -> str:
    """Why :func:`param_target_key` kept the chunk instead of asking for a value.

    A chunk target is not automatically a defect: an op that replaces a whole
    statement has no value to emit, and that is what ``no_value_pair_at_all``
    means.  The interesting answers are the ones that say the record *declares* a
    value pair and then contradicts itself --

    * ``old_value_absent_from_old_text`` and
    * ``substitution_differs_from_new_text``

    -- because ``apply_op`` reads ``new_text`` while the parameter head reads the
    pair, so the two consumers disagree about what the edit is.  Such an example
    is simultaneously unusable as a value target and misleading as a chunk target
    (the kind says ``replace_literal``, the target is three lines).  It is a data
    defect, not a hard example, and ``tests/test_polyglot_synth.py`` fails on it.
    """
    if not isinstance(props, dict):
        return "not_a_dict"
    for new_key, old_key in _VALUE_PAIRS:
        if new_key not in props or old_key not in props:
            continue
        if "old_text" not in props:
            # The applier reads the value directly (rename_symbol); no chunk to check.
            return "value_pair_ok"
        old_text = props.get("old_text")
        if not isinstance(old_text, str):
            return "old_text_not_str"
        old_value = str(props.get(old_key, ""))
        if not old_value:
            return "empty_old_value"
        if old_value not in old_text:
            return "old_value_absent_from_old_text"
        if substitute_value(props, new_key, old_key, str(props[new_key])) != props.get(
            "new_text"
        ):
            return "substitution_differs_from_new_text"
        return "value_pair_ok"
    if any(key in props for key in _CHUNK_PROP_KEYS):
        return "no_value_pair_at_all"
    return "nothing_usable"


def param_target_string(example: dict) -> str:
    """The single *value* the parameter head must produce (learnable, copyable).

    Iteration 5 measured that asking the head to emit the whole ``props`` as JSON
    never worked -- 59/59 and 6/6 held-out payloads came back undecodable, so the
    exact-JSON target is not what a small GRU decoder over byte-BPE can do.  The
    reflex agent only needs to choose *the new value* (a literal, an identifier, a
    short argument string); the executor reconstructs the rest of ``OpSpec.props``
    from the source region it is pointed at.  ``sylphae.model.decode.default_decode_spec``
    is the counterpart: it parses a JSON props object if the head ever emits one, and
    otherwise treats the emission as this value.

    Which key that value goes on is :func:`param_target_key`'s decision, and it is
    *not* simply "the first value-ish key present": for ``replace_literal`` the props
    carry both ``new_text`` (the whole line) and ``new`` (the value), and only the
    latter is what the agent consumes.

    Freeform / no-value ops carry nothing to emit, so their target is the empty
    string and they contribute no ``L_param`` supervision.
    """
    op = example.get("op") or {}
    props = op.get("props") or {}
    if not isinstance(props, dict):
        return ""
    key = param_target_key(props)
    return "" if key is None else str(props[key])


def param_answerable(example: dict) -> bool:
    """Whether the parameter target can actually be read off the request.

    The reflex contract is ``(concrete instruction, code context) -> op`` (DESIGN
    7.1): the planner resolves a qualitative intent into an instruction that
    *names* the new value, and the parameter head's job is to copy that name out
    of the request.  When the value is not in the request there is nothing to
    copy, and 12.12 measured what happens instead: the head learns the prior over
    old-source literals and its argmax is invariant to the request.

    This predicate is therefore the honest definition of "supervised" for
    ``L_param``.  It is deliberately strict -- a *substring* match on the raw
    request, not a semantic guess -- because a loose predicate would quietly
    re-admit the unanswerable rows the mask exists to remove.  Freeform / no-value
    ops have no target at all and are not answerable.
    """
    target = param_target_string(example)
    if not target:
        return False
    request = example.get("request")
    if not isinstance(request, str):
        return False
    return target in request
