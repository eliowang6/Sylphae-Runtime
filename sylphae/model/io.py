"""Reading a trained checkpoint into a model.  Inference-side half only.

``sylphae.train.checkpoint`` owns the *writer*: it packs the model, the optimiser's
per-parameter state and the run metadata into a single ``.safetensors`` file so a
run resumes exactly.  That is a training concern, and it drags ``torch.optim``
and the whole training vocabulary with it.

Inference only needs the other direction.  The reflex agent's ``--run-dir`` flag
and the neural candidate ranker both want the *same* strictness the trainer has
-- load all of it or fail loudly -- and neither of them trains anything.  So the
reader lives here, next to the model it loads into, and it does not import
``sylphae.train`` at all.  That is the whole reason this module exists: without it,
``sylphae.agent`` had an import edge into the training package purely to read a file.

The on-disk key prefixes live here too, because this module is where the format
is read; the writer imports them rather than restating them.  One definition of
the layout, two sides that agree by construction.

The contract is deliberately unforgiving.  Loading "most of" a checkpoint and
continuing is how a run silently ends up running a partially-random model, so a
key or shape that does not match raises :class:`CheckpointError` and names the
difference.

``safetensors`` is imported inside :func:`_read`, not at module scope: the
zero-weight path -- the lexical and AST tools, which is how the runtime is used
without any model -- must not pay for it.  It is not an extra dependency in
practice, because ``torch`` already requires it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "CHECKPOINT_NAME",
    "CheckpointError",
    "META_KEY",
    "MODEL_PREFIX",
    "OPTIM_PREFIX",
    "load_weights",
    "read_metadata",
]

CHECKPOINT_NAME = "checkpoint.safetensors"

#: On-disk key prefixes.  ``MODEL_PREFIX``/``META_KEY`` are read here;
#: ``OPTIM_PREFIX`` is declared here because the writer needs it and the layout
#: is defined in one place.
MODEL_PREFIX = "model."
OPTIM_PREFIX = "optim."
META_KEY = "__meta__"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be read, or does not match the model."""


def read_metadata(path: Path | str) -> dict[str, Any]:
    """Return ``{"step": int, "meta": dict}`` without touching a model.

    Lets a caller see what a run recorded -- which step, which config -- and
    decide whether it wants that checkpoint, before building a model to hold it.
    """
    path = _require_file(path)
    loaded = _read(path)
    meta = _pop_meta(path, loaded)
    return {"step": int(meta.get("step", 0)), "meta": meta.get("meta", {})}


def load_weights(path: Path | str, *, model: torch.nn.Module) -> dict[str, Any]:
    """Load ``model``'s parameters from ``path``; return the stored metadata.

    ``strict`` in the deepest sense: every parameter the model declares must be
    present, and present at the same shape.  A mismatch raises and says which
    parameters disagree, because "the checkpoint loaded but half of it was
    random" is not a failure anyone can notice downstream.
    """
    path = _require_file(path)
    loaded = _read(path)
    meta = _pop_meta(path, loaded)

    model_state = {k[len(MODEL_PREFIX):]: v for k, v in loaded.items()
                   if k.startswith(MODEL_PREFIX)}
    expected_state = model.state_dict()
    expected = set(expected_state.keys())
    actual = set(model_state.keys())
    if expected != actual:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise CheckpointError(
            f"checkpoint/model mismatch: {len(missing)} missing and {len(unexpected)} unexpected "
            f"parameter(s). missing={missing[:8]} unexpected={unexpected[:8]}. Refusing to load "
            f"a partial checkpoint."
        )
    shape_errors = [
        f"{name}: checkpoint {tuple(model_state[name].shape)} vs model "
        f"{tuple(expected_state[name].shape)}"
        for name in sorted(expected)
        if tuple(model_state[name].shape) != tuple(expected_state[name].shape)
    ]
    if shape_errors:
        raise CheckpointError(
            f"checkpoint/model shape mismatch ({len(shape_errors)} parameter(s)): "
            f"{shape_errors[:8]}. Refusing to load a checkpoint that does not fit this model."
        )
    model.load_state_dict(model_state, strict=True)
    return {"step": int(meta.get("step", 0)), "meta": meta.get("meta", {})}


def _require_file(path: Path | str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"no checkpoint at {path}")
    return path


def _read(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file  # noqa: PLC0415 - lazy by design

    return load_file(str(path))


def _pop_meta(path: Path, loaded: dict[str, torch.Tensor]) -> dict[str, Any]:
    if META_KEY not in loaded:
        raise CheckpointError(
            f"{path} has no {META_KEY} tensor; it is not a Sylphae checkpoint "
            f"(no run metadata, so no step count and no model prefix)"
        )
    return json.loads(bytes(loaded.pop(META_KEY).to(torch.uint8).tolist()).decode("utf-8"))
