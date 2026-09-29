"""Polyglot file indexing and symbol extraction across real-world project languages.

Supports:
- Frontend: TypeScript (.ts, .tsx), JavaScript (.js, .jsx), CSS (.css, .scss), HTML (.html), Vue (.vue)
- Systems/Backend: Go (.go), Rust (.rs), Python (.py)
- Configuration: JSON (.json), YAML (.yaml, .yml), TOML (.toml)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from sylphae.repo.ignore import IGNORE_FILE_NAMES, IgnoreRules, load_ignore_rules

__all__ = [
    "POLYGLOT_EXTENSIONS",
    "IGNORED_DIRS",
    "IGNORED_FILE_NAMES",
    "IGNORED_FILE_SUFFIXES",
    "SYLPHAE_ARTIFACT_PREFIX",
    "is_sylphae_artifact",
    "is_ignored_file",
    "iter_project_files",
    "extract_polyglot_symbols",
    "PolyglotFile",
    "IgnoreRules",
    "load_ignore_rules",
]

POLYGLOT_EXTENSIONS: set[str] = {
    # Web / Frontend
    ".ts", ".tsx", ".js", ".jsx", ".css", ".scss", ".html", ".vue", ".svelte",
    # Systems / Backend
    ".py", ".go", ".rs", ".java", ".c", ".cpp", ".h",
    # Database / API schema
    ".sql", ".prisma", ".graphql",
    # Config / Data
    ".json", ".yaml", ".yml", ".toml",
}

#: Directory names that are never project source in *any* repository: build outputs,
#: dependency caches, virtualenvs, editor state.  This table is shared by every project
#: Sylphae indexes, so it may only carry names whose meaning is universal.
#:
#: It used to also carry ``work``, ``scratch``, ``runs`` and ``corpus`` -- this
#: repository's private layout.  Those silently removed a legitimate directory from
#: every other project: ``src/work/card.css`` was never offered as a candidate and
#: nothing said so.  Names of that kind are declarations a repository makes about
#: itself, and are read from its ``.gitignore``/``.sylphaeignore`` instead -- see
#: :mod:`sylphae.repo.ignore`.
#:
#: This is why the shared table is deliberately narrow.  A name added here cannot be
#: argued with by downstream projects, so the bar for adding one is "no reasonable
#: project keeps source there", not "some project we have seen does".
IGNORED_DIRS: set[str] = {
    "node_modules", ".git", "dist", "build", "out", ".next", ".nuxt",
    "venv", ".venv", "__pycache__", "target", "bin", "obj", ".idea",
    ".vscode", "coverage", ".pytest_cache", "tmp", "temp", "vendor",
    ".turbo", ".gradle", ".cargo", "site-packages",
}

IGNORED_FILE_NAMES: set[str] = {
    "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "cargo.lock", "go.sum",
}

IGNORED_FILE_SUFFIXES: tuple[str, ...] = (
    ".map", ".min.js", ".min.css", ".bundle.js", ".chunk.js",
)

#: Every file Sylphae writes into a repo it operates on lives under this name prefix.
#:
#: These files deserve their own rule rather than an entry in ``IGNORED_FILE_NAMES``,
#: because they are wrong candidates for a reason lockfiles are not: they are a
#: *serialisation of the project itself*.  The index records every path and symbol in the
#: tree, so any keyword a request contains is already in its contents -- content search
#: scores it highly.  It also sorts first (dot prefix), so a fallback that has to pick an
#: arbitrary file picks it.  Both effects are sticky and neither has anything to do with
#: what the user asked for.
#:
#: A prefix rather than a filename: the failure mode is "a newly added artifact was
#: excluded from the paths that were updated and not the others", so the rule has to cover
#: artifacts that do not exist yet.
SYLPHAE_ARTIFACT_PREFIX: str = ".sylphae"

#: Sylphae's own *configuration* files.  They carry the artifact prefix because they are
#: Sylphae's, but they are not artifacts: they declare what Sylphae should skip, which is a
#: property of the repository and belongs in version control next to ``.gitignore``.
#: Without this carve-out the prefix rule would swallow them, and a file whose whole
#: purpose is to describe exclusions would itself be excluded.
_SYLPHAE_CONFIG_NAMES: frozenset[str] = frozenset(
    name.lower() for name in IGNORE_FILE_NAMES if name.lower().startswith(SYLPHAE_ARTIFACT_PREFIX)
)


def is_sylphae_artifact(name: str) -> bool:
    """True when ``name`` is one of Sylphae's own runtime artifacts, not project source."""
    base = Path(name).name.lower()
    if base in _SYLPHAE_CONFIG_NAMES:
        return False
    return base.startswith(SYLPHAE_ARTIFACT_PREFIX)


def is_ignored_file(name: str) -> bool:
    """True when ``name`` is never project source.

    The single place this decision is made.  ``iter_project_files``, the content-search
    fallback and the ripgrep path all defer to it, so a discovery path cannot silently
    disagree with the others about what counts as source.
    """
    lowered = Path(name).name.lower()
    return (
        is_sylphae_artifact(name)
        or lowered in IGNORED_FILE_NAMES
        or lowered.endswith(IGNORED_FILE_SUFFIXES)
    )


def iter_project_files(root: Path, extensions: set[str] | None = None) -> list[str]:
    """Return all relevant source and config files relative to root, sorted deterministically.

    The shared table and the repository's own declarations apply together: universal
    directory names come from :data:`IGNORED_DIRS`, everything project-specific comes
    from the project.  Keeping the second half in the project is what lets
    ``src/work/card.css`` survive in a repository that never meant to exclude it.
    """
    root = Path(root).resolve()
    valid_exts = extensions or POLYGLOT_EXTENSIONS
    rules = load_ignore_rules(root)
    result: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
        if rel_dir == ".":
            rel_dir = ""

        kept: list[str] = []
        for name in dirnames:
            if name.startswith(".") or name.lower() in IGNORED_DIRS:
                continue
            rel = f"{rel_dir}/{name}" if rel_dir else name
            if rules.ignores_dir(rel, name):
                continue
            kept.append(name)
        # Prune ignored directories in-place
        dirnames[:] = kept

        for fname in filenames:
            if is_ignored_file(fname):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in valid_exts:
                continue
            full_path = os.path.join(dirpath, fname)
            rel = os.path.relpath(full_path, root).replace(os.sep, "/")
            if rules.ignores_file(rel, fname):
                continue
            result.append(rel)

    result.sort()
    return result


class PolyglotFile:
    """Represents a discovered polyglot file with extracted symbols."""

    def __init__(self, rel_path: str, root: Path):
        self.path = rel_path
        self.root = root
        self.ext = os.path.splitext(rel_path)[1].lower()
        self._content: str | None = None
        self._symbols: list[str] | None = None

    @property
    def content(self) -> str:
        if self._content is None:
            full = self.root / self.path
            try:
                self._content = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                self._content = ""
        return self._content

    @property
    def symbols(self) -> list[str]:
        if self._symbols is None:
            self._symbols = extract_polyglot_symbols(self.path, self.content)
        return self._symbols


def extract_polyglot_symbols(file_path: str, content: str) -> list[str]:
    """Extract candidate identifiers, class names, selectors, and tags from source content."""
    if not content:
        return []

    ext = os.path.splitext(file_path)[1].lower()
    symbols: set[str] = set()

    # 1. CSS / SCSS selectors & properties
    if ext in (".css", ".scss", ".less"):
        # Class names: .btn-primary, .navbar, .header-actions
        for m in re.finditer(r"\.([A-Za-z0-9_-]+)", content):
            symbols.add(m.group(1))
        # IDs: #header, #nav
        for m in re.finditer(r"#([A-Za-z0-9_-]+)", content):
            symbols.add(m.group(1))
        # CSS variables: --primary-color
        for m in re.finditer(r"(--[A-Za-z0-9_-]+)", content):
            symbols.add(m.group(1))
        return sorted(symbols, key=lambda s: -len(s))

    # 2. TypeScript / JavaScript / TSX / JSX / Vue
    if ext in (".ts", ".tsx", ".js", ".jsx", ".vue", ".svelte"):
        # React components / functions: function Navbar, const TopNav = ...
        for m in re.finditer(r"\b(?:function|class|interface|type)\s+([A-Za-z0-9_]+)", content):
            symbols.add(m.group(1))
        for m in re.finditer(r"\b(?:const|let|var)\s+([A-Za-z0-9_]+)\s*[:=]", content):
            symbols.add(m.group(1))
        # JSX component tags: <Button, <Navbar, <Header
        for m in re.finditer(r"<([A-Z][A-Za-z0-9_]*)", content):
            symbols.add(m.group(1))
        # HTML tag class names: className="btn btn-primary top-right"
        for m in re.finditer(r'class(?:Name)?=["\']([^"\']+)["\']', content):
            for token in m.group(1).split():
                if len(token) >= 3:
                    symbols.add(token)

    # 3. Go
    elif ext == ".go":
        # func DoSomething, type Config struct
        for m in re.finditer(r"\bfunc\s+(?:\([^)]+\)\s+)?([A-Za-z0-9_]+)", content):
            symbols.add(m.group(1))
        for m in re.finditer(r"\btype\s+([A-Za-z0-9_]+)\s+(?:struct|interface)", content):
            symbols.add(m.group(1))

    # 4. Rust
    elif ext == ".rs":
        # fn execute, struct Config, impl Service
        for m in re.finditer(r"\b(?:fn|struct|enum|trait|impl)\s+([A-Za-z0-9_]+)", content):
            symbols.add(m.group(1))

    # 5. Python
    elif ext == ".py":
        for m in re.finditer(r"\b(?:def|class)\s+([A-Za-z0-9_]+)", content):
            symbols.add(m.group(1))

    # General fallback: identifiers with underscores or camelCase
    for m in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_-]{2,}\b", content):
        token = m.group(0)
        if not token.isdigit() and len(token) <= 40:
            symbols.add(token)

    return sorted(symbols, key=lambda s: -len(s))
