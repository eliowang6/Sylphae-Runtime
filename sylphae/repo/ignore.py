"""Repository-level ignore declarations.

``polyglot.IGNORED_DIRS`` holds directory names that are never project source in any
repository -- ``node_modules``, ``.git``, ``dist``.  It also held ``work``, ``scratch``,
``runs`` and ``corpus``, which are *this* repository's private layout.  A
general-purpose indexer that carries one project's directory names silently skips
``src/work/card.css`` in every other project: the file is simply never offered as a
candidate, and nothing reports that it was passed over.  AGENTS.md 2.1 rules out
exactly this kind of hardcoding.

Rules that describe one repository belong to that repository, so they are read from it:
``.gitignore``, which every real project already maintains and which answers precisely
the question "what here is not source", plus an optional ``.sylphaeignore`` for exclusions
that should not affect git.

The implemented subset is the one that matters for pruning a directory walk --
comments, blank lines, negation, trailing-slash directory rules, anchoring, and the
``*`` / ``**`` / ``?`` / ``[...]`` wildcards.  There is no index to consult, so no rule
can depend on whether a path is tracked, only on its name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "IGNORE_FILE_NAMES",
    "IgnoreRules",
    "load_ignore_rules",
]

#: Consulted in order.  ``.sylphaeignore`` is second so it can refine ``.gitignore``
#: rather than replace it -- last matching rule wins, as in git.
IGNORE_FILE_NAMES: tuple[str, ...] = (".gitignore", ".sylphaeignore")


@dataclass(frozen=True)
class _Rule:
    pattern: re.Pattern[str]
    negated: bool
    dir_only: bool


def _translate(glob: str) -> str:
    """Translate a gitignore glob body into a regex, anchored to a whole relative path."""
    out: list[str] = []
    i = 0
    n = len(glob)
    while i < n:
        ch = glob[i]
        if ch == "*":
            if glob.startswith("**/", i):
                out.append("(?:.*/)?")
                i += 3
                continue
            if glob.startswith("**", i):
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        elif ch == "[":
            j = i + 1
            if j < n and glob[j] in "!^":
                j += 1
            if j < n and glob[j] == "]":
                j += 1
            while j < n and glob[j] != "]":
                j += 1
            if j >= n:
                # Unterminated class: the character is literal, not a class opener.
                out.append(r"\[")
            else:
                body = glob[i + 1 : j]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append("[" + body + "]")
                i = j + 1
                continue
        elif ch == "\\" and i + 1 < n:
            out.append(re.escape(glob[i + 1]))
            i += 2
            continue
        else:
            out.append(re.escape(ch))
        i += 1
    return "".join(out)


def _compile(line: str) -> _Rule | None:
    """Parse one gitignore line, or ``None`` when it carries no rule."""
    line = line.rstrip("\r\n")
    if not line.strip() or line.lstrip().startswith("#"):
        return None

    negated = line.startswith("!")
    if negated:
        line = line[1:]

    # Trailing whitespace is insignificant unless escaped; only unescaped spaces go.
    line = re.sub(r"(?<!\\)\s+$", "", line)
    if not line:
        return None

    dir_only = line.endswith("/")
    if dir_only:
        line = line[:-1]
    if not line:
        return None

    # A slash anywhere but the end anchors the pattern to the repository root.
    # Without one, the pattern matches at any depth -- ``work`` means ``**/work``.
    anchored = "/" in line
    if line.startswith("/"):
        line = line[1:]
    if not line:
        return None

    prefix = "" if anchored else "(?:.*/)?"
    try:
        return _Rule(re.compile(rf"^{prefix}{_translate(line)}$"), negated, dir_only)
    except re.error:
        return None


def _matches(rules: tuple[_Rule, ...], rel: str, is_dir: bool) -> bool:
    """Whether ``rel`` is ignored, honouring git's last-match-wins ordering."""
    rel = rel.replace("\\", "/").strip("/")
    if not rel:
        return False
    ignored = False
    for rule in rules:
        if rule.dir_only and not is_dir:
            continue
        if rule.pattern.match(rel):
            ignored = not rule.negated
    return ignored


@dataclass(frozen=True)
class IgnoreRules:
    """The ignore rules a repository declares about itself."""

    _rules: tuple[_Rule, ...] = ()

    def __bool__(self) -> bool:
        return bool(self._rules)

    def ignores_dir(self, rel: str, name: str) -> bool:
        """True when a directory should be pruned from the walk.

        ``rel`` is the path from the repository root; ``name`` is its basename.  Both
        are consulted because an anchored rule (``/src/gen``) only matches the full
        path while a bare one matches either.
        """
        return _matches(self._rules, rel, True) or _matches(self._rules, name, True)

    def ignores_file(self, rel: str, name: str) -> bool:
        """True when a file is not project source."""
        return _matches(self._rules, rel, False) or _matches(self._rules, name, False)


def load_ignore_rules(root: Path) -> IgnoreRules:
    """Read the repository's own ignore declarations.

    Cheap on purpose: two small files, read once per walk.  Caching would need to be
    keyed on mtime to stay correct across edits within a session, which costs more than
    the read it avoids.
    """
    collected: list[_Rule] = []
    for name in IGNORE_FILE_NAMES:
        try:
            text = (Path(root) / name).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            rule = _compile(line)
            if rule is not None:
                collected.append(rule)
    return IgnoreRules(tuple(collected))
