"""The hard invariant: ``apply_op(old, detect_op(old, new)) == new``, exactly.

This is the test that decides whether the structured-op vocabulary means anything.
If it goes red, the coverage number computed downstream is a fiction.
"""

from __future__ import annotations

import json

import pytest

from sylphae.ops import OP_KINDS, OpSpec, apply_op, detect_op
from sylphae.ops.vocab import EXECUTABLE_KINDS, NEVER_DETECTED

from ops_corpus import CORPUS, COVERED_KINDS

PATH = "src/pkg/module.py"


def test_corpus_covers_every_detectable_kind():
    """A corpus that stops covering a kind would silently weaken every test below."""
    missing = set(OP_KINDS) - COVERED_KINDS - set(NEVER_DETECTED)
    assert not missing, f"corpus has no example for op kind(s): {sorted(missing)}"
    assert "freeform_patch" in COVERED_KINDS


@pytest.mark.parametrize("name,old,new,expected", CORPUS, ids=[c[0] for c in CORPUS])
def test_detect_labels_the_expected_kind(name, old, new, expected):
    spec = detect_op(old, new, PATH)
    assert spec is not None, f"{name}: detect_op returned None for a real change"
    assert spec.kind == expected, (
        f"{name}: expected kind {expected!r}, got {spec.kind!r} "
        f"(props={spec.props!r}, span={spec.span!r})"
    )
    assert spec.path == PATH


@pytest.mark.parametrize("name,old,new,expected", CORPUS, ids=[c[0] for c in CORPUS])
def test_round_trip_is_exact(name, old, new, expected):
    """The headline invariant, asserted character for character."""
    spec = detect_op(old, new, PATH)
    assert spec is not None
    if spec.kind == "freeform_patch":
        with pytest.raises(Exception):
            apply_op(old, spec)
        return
    assert spec.kind in EXECUTABLE_KINDS
    assert apply_op(old, spec) == new, f"{name}: replay diverged from the target"


def test_every_corpus_example_is_replayable_or_explicitly_freeform():
    """No example may be structured-but-unreplayable, and no freeform may replay."""
    for name, old, new, _expected in CORPUS:
        spec = detect_op(old, new, PATH)
        assert spec is not None
        if spec.kind == "freeform_patch":
            assert spec.kind not in EXECUTABLE_KINDS
        else:
            assert apply_op(old, spec) == new, name


def test_identical_sources_return_none():
    for src in (
        "x = 1\n",
        "",
        "\n\n\n",
        "def f(a, b=2):\n    return a + b\n",
        "import os\n\n\n# comment\n",
    ):
        assert detect_op(src, src, PATH) is None


def test_whitespace_only_difference_is_not_none_and_is_freeform():
    """A trailing-space change is a real change, and not a structured one."""
    spec = detect_op("x = 1\n", "x = 1 \n", PATH)
    assert spec is not None
    assert spec.kind == "freeform_patch"


def test_props_survive_json_round_trip():
    for name, old, new, _expected in CORPUS:
        spec = detect_op(old, new, PATH)
        assert spec is not None
        blob = json.dumps(spec.to_dict())
        revived = OpSpec.from_dict(json.loads(blob))
        assert revived == spec, name
        assert revived.span == spec.span
        for key, value in revived.props.items():
            assert isinstance(key, str) and isinstance(value, str), (name, key, value)
        if revived.kind != "freeform_patch":
            assert apply_op(old, revived) == new, name


def test_freeform_spec_carries_hashes_not_the_new_source():
    """The escape hatch must stay cheap: no full-file payload in props."""
    old = "def f(x):\n    return x\n"
    new = "def f(x):\n    y = x * 2\n    return y\n"
    spec = detect_op(old, new, PATH)
    assert spec.kind == "freeform_patch"
    assert set(spec.props) == {"reason", "n_regions", "old_sha256", "new_sha256"}
    assert len(spec.props["old_sha256"]) == 64
    assert len(spec.props["new_sha256"]) == 64
    assert new not in json.dumps(spec.to_dict())
