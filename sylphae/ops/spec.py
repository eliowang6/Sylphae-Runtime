"""``OpSpec`` — one structured edit operation, and its JSON form.

``props`` values are **plain strings** (``INTERFACES.md`` §1).  That is not
cosmetic: the props travel through JSONL into the corpus, and the model's param
head emits tokens, so every payload has to be text.  ``OpSpec`` refuses anything
else rather than coercing it, per the "reject, do not coerce" convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .vocab import OP_TO_ID


class OpApplyError(Exception):
    """Raised by ``apply_op`` when a spec does not fit the source it is applied to.

    Also raised for the whole ``freeform_patch`` kind: a freeform patch is by
    definition *not* replayable by the structured executor.  See
    ``sylphae/ops/apply.py``.
    """


@dataclass
class OpSpec:
    """A single structured edit.

    Attributes:
        kind: one of :data:`sylphae.ops.vocab.OP_KINDS`.
        path: repo-relative posix path of the edited file.
        span: 1-based inclusive line span in the **original** source.  For the
            pure-insertion kinds (see
            :data:`sylphae.ops.vocab.INSERTION_KINDS`) this is ``(p, p)`` and means
            "insert before original line ``p``", where ``p`` may be
            ``len(old_lines) + 1`` (append).  For deletions and replacements it
            covers the original lines consumed.
        props: op-specific payload.  Every key and every value is a ``str``.
    """

    kind: str
    path: str
    span: tuple[int, int]
    props: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in OP_TO_ID:
            raise ValueError(
                f"OpSpec.kind must be one of {sorted(OP_TO_ID)}; got {self.kind!r}"
            )
        if not isinstance(self.path, str) or not self.path:
            raise ValueError(f"OpSpec.path must be a non-empty str; got {self.path!r}")

        span = self.span
        if isinstance(span, list):  # be forgiving about the JSON round-trip only
            span = tuple(span)
            object.__setattr__(self, "span", span)
        if not isinstance(span, tuple) or len(span) != 2:
            raise ValueError(f"OpSpec.span must be a 2-tuple; got {self.span!r}")
        a, b = span
        if not isinstance(a, int) or not isinstance(b, int) or isinstance(a, bool):
            raise ValueError(f"OpSpec.span must be two ints; got {self.span!r}")
        if a < 1 or b < a:
            raise ValueError(
                f"OpSpec.span must satisfy 1 <= a <= b; got ({a}, {b})"
            )

        if not isinstance(self.props, dict):
            raise ValueError(f"OpSpec.props must be a dict; got {type(self.props)!r}")
        for key, value in self.props.items():
            if not isinstance(key, str):
                raise ValueError(f"OpSpec.props keys must be str; got {key!r}")
            if not isinstance(value, str):
                raise ValueError(
                    f"OpSpec.props[{key!r}] must be a str (props go through JSON); "
                    f"got {type(value).__name__} {value!r}"
                )

    # ------------------------------------------------------------------ JSON

    def to_dict(self) -> dict:
        """JSON-serialisable form.  ``span`` becomes a list of two ints."""
        return {
            "kind": self.kind,
            "path": self.path,
            "span": [int(self.span[0]), int(self.span[1])],
            "props": {str(k): str(v) for k, v in self.props.items()},
        }

    @staticmethod
    def from_dict(d: dict) -> "OpSpec":
        """Inverse of :meth:`to_dict`.  Raises ``ValueError`` on a malformed dict."""
        if not isinstance(d, dict):
            raise ValueError(f"OpSpec.from_dict expects a dict; got {type(d)!r}")
        for key in ("kind", "path", "span", "props"):
            if key not in d:
                raise ValueError(f"OpSpec.from_dict: missing key {key!r} in {d!r}")
        return OpSpec(
            kind=d["kind"],
            path=d["path"],
            span=tuple(d["span"]),
            props=dict(d["props"]),
        )

    # ------------------------------------------------------------------ misc

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"OpSpec(kind={self.kind!r}, path={self.path!r}, "
            f"span={self.span!r}, props={self.props!r})"
        )
