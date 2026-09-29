import tempfile
from pathlib import Path

from sylphae.ops.polyglot import (
    add_css_properties,
    modify_jsx_class,
    replace_css_property,
)
from sylphae.repo.polyglot import (
    extract_polyglot_symbols,
    iter_project_files,
)


def test_iter_project_files():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "src").mkdir()
        (root / "src" / "index.ts").write_text("export const x = 1;")
        (root / "src" / "App.tsx").write_text("export function App() {}")
        (root / "src" / "style.css").write_text(".btn { color: red; }")
        (root / "pkg").mkdir()
        (root / "pkg" / "main.go").write_text("package main")
        (root / "node_modules").mkdir()
        (root / "node_modules" / "bad.js").write_text("bad")
        (root / ".git").mkdir()
        (root / ".git" / "HEAD").write_text("ref")

        files = iter_project_files(root)
        assert "src/index.ts" in files
        assert "src/App.tsx" in files
        assert "src/style.css" in files
        assert "pkg/main.go" in files
        assert not any("node_modules" in f for f in files)
        assert not any(".git" in f for f in files)


def test_extract_polyglot_symbols():
    css = ".header-actions .btn-primary { color: var(--main-color); } #logo { width: 100px; }"
    syms = extract_polyglot_symbols("style.css", css)
    assert "header-actions" in syms
    assert "btn-primary" in syms
    assert "logo" in syms
    assert "--main-color" in syms

    tsx = """
    import React from 'react';
    export const HeaderNavbar: React.FC = () => {
        return <button className="btn btn-top-right">Click</button>;
    };
    """
    syms_tsx = extract_polyglot_symbols("Header.tsx", tsx)
    assert "HeaderNavbar" in syms_tsx
    assert "btn-top-right" in syms_tsx


def test_replace_css_property():
    css = """
    .btn-top-right {
        background-color: #3b82f6;
        font-size: 14px;
        padding: 8px;
    }
    """
    new_src, spec = replace_css_property(css, "background-color", "#1d4ed8", context_hint="btn-top-right")
    assert spec is not None
    assert "background-color: #1d4ed8;" in new_src
    assert spec.props["old"] == "#3b82f6"
    assert spec.props["new"] == "#1d4ed8"


def test_modify_jsx_class():
    jsx = '<button className="btn bg-blue-400 text-sm px-4">Action</button>'
    new_src, spec = modify_jsx_class(jsx, "bg-blue-400", "bg-blue-600")
    assert spec is not None
    assert 'className="btn bg-blue-600 text-sm px-4"' in new_src


def test_add_css_properties():
    css = """
    .user-card {
        width: 200px;
    }
    """
    new_src, spec = add_css_properties(css, "user-card", {
        "overflow": "hidden",
        "text-overflow": "ellipsis",
    })
    assert spec is not None
    assert "overflow: hidden;" in new_src
    assert "text-overflow: ellipsis;" in new_src
