"""The runtime must stay separable from training and data synthesis.

Sylphae-Runtime is generated from the runtime half of this tree.  Nothing in the
build stops a later edit from adding ``import sylphae.train`` to a file that
ships, and the failure mode is not a crash -- it is a published package that
drags torch-optimiser vocabulary along with it, or worse, silently pins the
consumer to a private repository.

So the constraint is tested, not documented.  The import graph is the source of
truth; these assertions read it the way a consumer would.

The rules, and the reason each one is a rule rather than a preference:

1. **No runtime file imports ``sylphae.data`` or ``sylphae.train``.**  A shared
   definition is fine -- three of them exist, in ``ops/params.py`` and
   ``model/io.py`` -- but it has to live in a module that is itself shipped.
2. **The runtime does not import ``sylphae.eval``.**  The reflex agent is the
   product; the harness is how we measure it.  An import from the product into
   the measurement means one of the two is in the wrong package.
3. **Every shared definition both sides need lives outside both.**  Named
   explicitly, because "just put it in the model layer" is the move that
   recreates the coupling, and it recreates it in a file that does not import
   anything so nothing flags it.

Only *static* module-level imports are read.  A lazy import inside a function is
still an edge, but the torch-on-the-hot-path work in ``agent/neural.py`` and
``agent/cli.py`` is deliberately lazy and stays that way, so a naive scan would
report a false positive on the two files most carefully written to avoid it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "sylphae"

#: Everything in here ships to Sylphae-Runtime.
RUNTIME_PACKAGES = (
    "agent",
    "coprocessor",
    "exec",
    "ops",
    "ops_env",
    "repo",
    "vision",
)

#: Ships, but only in part -- see RUNTIME_MODEL_MODULES.
RUNTIME_MODEL_MODULES = (
    "backbone",
    "collate",
    "config",
    "decode",
    "fast_decode",
    "io",
    "policy",
    "semantic_grounder",
    "snippet_ranker",
    "ssd",
    "tokenizer",
)

#: Stays behind.  Training-only (optimiser, gradients) or the corpus layer.
TRAINING_PACKAGES = ("data", "eval", "train")


def _iter_runtime_sources() -> list[Path]:
    out = [p for pkg in RUNTIME_PACKAGES for p in (SRC / pkg).rglob("*.py")]
    out += [SRC / "model" / f"{name}.py" for name in RUNTIME_MODEL_MODULES]
    return sorted(p for p in out if "__pycache__" not in p.parts)


def _imported_modules(path: Path) -> set[str]:
    """Every absolute ``sylphae.*`` module this file names, lazy or not.

    Deliberately reads the AST rather than grepping text, so a module named in a
    comment or a docstring -- which is most of the prose in this codebase -- is
    not mistaken for a dependency.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                if alias.name.split(".")[0] == "sylphae":
                    found.add(alias.name)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.level:
                # A relative import: resolve it against the file's own package.
                base = "sylphae" + str(path.parent.relative_to(SRC)).replace("\\", ".")
                base = base.replace(".__init__", "")
                found.add(f"{base}.{node.module}" if node.module else base)
            elif node.module and node.module.split(".")[0] == "sylphae":
                found.add(node.module)

    Visitor().visit(tree)
    return found


def test_every_runtime_package_exists() -> None:
    """A typo in RUNTIME_PACKAGES would make every other test here vacuous."""
    for pkg in RUNTIME_PACKAGES:
        assert (SRC / pkg).is_dir(), f"{pkg} is listed as runtime but does not exist"


@pytest.mark.parametrize("path", _iter_runtime_sources(), ids=lambda p: p.name)
def test_runtime_never_imports_training_or_data(path: Path) -> None:
    """Rule 1.  The constraint that defines the split."""
    offenders = sorted(
        m for m in _imported_modules(path)
        if m.split(".")[1:2] and m.split(".")[1] in TRAINING_PACKAGES
    )
    assert not offenders, (
        f"{path.relative_to(SRC.parent)} imports {offenders}, which does not ship to "
        f"Sylphae-Runtime. A definition both sides need belongs in a shipped module -- "
        f"ops/params.py and model/io.py already hold the three that were needed."
    )


def test_runtime_never_imports_the_eval_package() -> None:
    """Rule 2.  The product must not import the thing that measures it.

    Checked across the whole runtime at once rather than per-file, because the
    failure reads better as a list of edges than as N identical assertions.
    """
    edges: list[str] = []
    for path in _iter_runtime_sources():
        for module in _imported_modules(path):
            if module.split(".")[1:2] == ["eval"]:
                edges.append(f"{path.relative_to(SRC.parent)} -> {module}")
    assert not edges, f"runtime imports the eval package: {edges}"


def test_the_shared_definitions_both_sides_need_are_outside_both() -> None:
    """Rule 3.  The three modules that exist because of this split.

    Each is named with the coupling it exists to break, so that if one is
    deleted the reason goes with it.  The property asserted is *neutrality*: a
    module both sides import is worthless if it leans on either side.
    """
    shared = {
        "ops/params.py": "runtime -> data (the param-label predicates)",
        "model/io.py": "runtime -> train (the checkpoint reader)",
        "model/decode.py": "runtime -> eval (the param decoder)",
    }
    for name, why in shared.items():
        path = SRC / name
        assert path.is_file(), f"{name} is missing: it is what broke {why}"
        leans = sorted(
            m for m in _imported_modules(path)
            if m.split(".")[1:2] and m.split(".")[1] in TRAINING_PACKAGES
        )
        assert not leans, (
            f"{name} imports {leans}. It cannot be the shared home for {why} while "
            f"depending on the package it was extracted from."
        )


def test_the_param_predicates_are_importable_without_torch() -> None:
    """The property that makes the sharing real rather than aspirational.

    ``scripts/validate_synth_corpus.py`` has to ask what a corpus row's
    parameter target is without paying for a torch import (AGENTS.md 2.2).  That
    only holds if ``ops/params.py`` is stdlib-only, so assert it on a real
    import rather than by reading the source for the word "torch" -- which a
    docstring mentioning it would defeat.
    """
    import subprocess
    import sys

    probe = (
        "import sys; import sylphae.ops.params; "
        "assert 'torch' not in sys.modules, 'ops.params pulled torch'; "
        "assert 'numpy' not in sys.modules, 'ops.params pulled numpy'"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=SRC.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"ops/params.py is no longer torch-free:\n{result.stderr.strip()}"
    )
