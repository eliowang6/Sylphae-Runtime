"""Execute a cross-file rename, having already decided that refusing it is worse.

Round 3 added the consistency gate that *refuses* a rename whose symbol is referenced
elsewhere, and it works: on ``pyv_rich_rename_not_renderable`` it withheld a patch that would
have left 11 references dangling. But refusing is only half of a capability, and a half that
leaves DESIGN.md section 1's ``rename`` shape permanently ``declined (fails safely)`` -- the
safe behaviour with none of the usefulness.

This module is the other half. It rewrites the symbol at every reference site the gate
already found, in one pass, and reports what it did so the change can be audited.

**Why this is a separate module and not a branch in the reflex.** The gate answers "can this
be done in one file"; the executor answers "do it across all of them". They share
``find_references`` but nothing else, and the reflex needs the first without the second --
an agent asked to rename something in one file should still escalate rather than silently
grow the blast radius of a single edit. Keeping them apart is also what lets the executor be
tested against a whole tree while the gate stays a per-file predicate.

**The safety rule that shapes everything here: only resolved identifier positions are
rewritten.** ``find_references`` returns three tiers:

  ``import`` / ``name`` / ``attribute`` / ``arg`` / ``definition``
      An AST node whose own identifier is the symbol. Positioned, and safe to rewrite.
  ``text``
      A regex hit in a non-Python file. Not resolved to an identifier -- it could be inside
      a string literal, a comment, a different language's syntax, or a longer name that
      merely contains the symbol.
  ``config``
      The same, in a config file.

MEASURED 2026-10-02: on the rich fixture, ``NotRenderableError`` has 11 references, of which
10 are AST-resolved and 1 is a ``text`` hit inside ``pyproject.toml``. Rewriting the 10 and
leaving the 1 is *worse* than refusing, because the tree is then inconsistent in a way that
still compiles -- a packaging reference to a class that no longer exists. So a ``text`` or
``config`` hit is a **blocker**, reported with its line, not a silent skip. Refusing with an
exact list beats a partial rename that looks complete.

**What this does not do.** It does not verify the result. It reports the edits; whether the
tree still passes its tests is the caller's question, answered by ``sylphae.exec.verify`` or
``sylphae.exec.verify_eco``. A rename that breaks the build is caught by the same
before/after diff every other patch goes through, which is the point of routing it there
rather than asserting success here.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from sylphae.exec.consistency import Reference, find_references

__all__ = [
    "MultiFileRenameResult",
    "plan_multi_file_rename",
    "apply_multi_file_rename",
]

#: Reference kinds whose position comes from a resolved AST identifier and can therefore be
#: rewritten without guessing. Everything else is a blocker.
_REWRITABLE_KINDS = frozenset({"import", "name", "attribute", "arg", "definition"})


@dataclass
class MultiFileRenameResult:
    """What a cross-file rename did, or why it could not be done."""

    old: str
    new: str
    #: ``{relative path: number of identifiers rewritten}``, only for files actually changed.
    edited: dict[str, int] = field(default_factory=dict)
    #: References that are not AST-resolved, so were not rewritten. Non-empty means the tree
    #: may still reference the old name and the caller should escalate.
    unresolved: list[Reference] = field(default_factory=list)
    #: Unresolved references that are **documentation**, not code.
    #:
    #: MEASURED 2026-10-02 on the rich fixture: after deduplicating the text tier against
    #: the AST tier, exactly two sites remained, and both were docstrings --
    #: ``rich/measure.py:90`` and ``rich/table.py:437``, each an ``errors.NotRenderableError:``
    #: line inside a ``Raises:`` block.
    #:
    #: These are separated from :attr:`unresolved` because they are a different kind of
    #: leftover. A docstring does not execute, so leaving it does not leave a dangling
    #: reference or a broken import; it leaves the *documentation* naming a class that no
    #: longer exists. That is worth reporting and worth a human's five seconds, but it is not
    #: the same as an unresolved code reference, and treating the two alike would mean the
    #: executor refuses every rename in a well-documented codebase -- which is the refusal
    #: this module exists to replace.
    #:
    #: The classification is conservative in the specific way that matters: a line is called
    #: documentation only when it is inside a string literal that ``ast`` reports as a
    #: docstring node, or when the match sits in a comment. Anything else stays in
    #: :attr:`unresolved`.
    doc_leftovers: list[Reference] = field(default_factory=list)
    #: Set when the plan was refused outright.
    blocked: str | None = None

    @property
    def applied(self) -> bool:
        """Whether files were written.

        ``True`` with a non-empty ``unresolved`` is possible and is still a partial result --
        which is why this is not the same question as "is the tree consistent now". A caller
        that needs the second must check ``unresolved``.
        """
        return bool(self.edited)

    @property
    def consistent(self) -> bool:
        """Whether every *code* reference to the old name was rewritten.

        The honest question after a rename, and the one ``applied`` deliberately does not
        answer: a run that rewrote 10 of 11 sites has ``applied=True`` and
        ``consistent=False``.

        Docstring leftovers do not make the tree inconsistent, so they are excluded here and
        surfaced through :attr:`doc_leftovers` instead. They are counted in
        :meth:`summary` so they cannot be dropped silently.
        """
        return bool(self.edited) and not self.unresolved

    def summary(self) -> str:
        if self.blocked:
            return self.blocked
        if not self.edited:
            return (f"{self.old!r} -> {self.new!r}: nothing to rewrite; the symbol has no "
                    f"reference in the tree")
        total = sum(self.edited.values())
        files = len(self.edited)
        parts = [f"{self.old!r} -> {self.new!r}: rewrote {total} occurrence(s) across "
                 f"{files} file(s)"]
        if self.unresolved:
            shown = "; ".join(str(r) for r in self.unresolved[:3])
            parts.append(
                f"but {len(self.unresolved)} code reference(s) could not be resolved: "
                f"{shown}" + (f" (+{len(self.unresolved) - 3} more)"
                              if len(self.unresolved) > 3 else ""))
        if self.doc_leftovers:
            shown = "; ".join(str(r) for r in self.doc_leftovers[:3])
            parts.append(
                f"documentation still names the old symbol in {len(self.doc_leftovers)} "
                f"place(s) ({shown}"
                + (f" (+{len(self.doc_leftovers) - 3} more)"
                   if len(self.doc_leftovers) > 3 else "") + "); not rewritten because a "
                f"docstring is prose, not a reference")
        return "; ".join(parts)


def _documentation_lines(root: Path, files: list[str]) -> set[tuple[str, int]]:
    """``(path, line)`` pairs where a match sits in prose rather than code.

    Two sources, both resolved rather than guessed:

      **String literals.** ``ast`` reports the first line of every string constant but not
      its extent, so the literal's own body is re-scanned with ``tokenize`` to find where it
      ends. Only the lines the literal actually spans count, which is what keeps a
      multi-line docstring from swallowing the code that follows it.

      **Comments.** ``tokenize`` emits COMMENT tokens with a real line number.

    A line that is neither is not documentation. The default is "code", so anything this
    fails to recognise blocks the rename rather than being quietly skipped -- the direction
    that costs a person one extra confirmation instead of a broken tree.
    """
    out: set[tuple[str, int]] = set()
    for rel in files:
        path = root / rel
        # Only Python files can contain a docstring. A ``config.toml`` or a ``.yaml`` holds
        # a *value* that names the symbol -- a packaging entry, an env var name -- and
        # rewriting that needs to understand the file's format, not just where a string
        # happens to sit. MEASURED 2026-10-02: a test that dropped the symbol into a
        # ``config.toml`` and expected a refusal got a clean pass, because ``ast.parse``
        # fails on TOML and the file was treated as unclassifiable rather than as
        # un-rewritable. Values are code to a packaging tool; prose is only prose inside
        # Python.
        if path.suffix != ".py":
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            # An unparseable file is not an opportunity to relax the rule.
            continue

        lines = source.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            start = getattr(node, "lineno", None)
            if start is None:
                continue
            end = getattr(node, "end_lineno", None) or start
            # ``ast`` gives the literal's extent for Python 3.8+, which is exact. The
            # tokenize pass below is only needed for the prefixes (r, f, b) that make a
            # literal's start line differ from the node's first content line.
            for ln in range(start, min(end, len(lines)) + 1):
                out.add((rel, ln))

        try:
            import io
            import tokenize

            for tok in tokenize.generate_tokens(io.StringIO(source).readline):
                if tok.type == tokenize.COMMENT and tok.start[0] == tok.end[0]:
                    out.add((rel, tok.start[0]))
                elif tok.type == tokenize.STRING and tok.start[0] == tok.end[0]:
                    out.add((rel, tok.start[0]))
        except (tokenize.TokenError, IndentationError, SyntaxError):
            pass
    return out


def plan_multi_file_rename(
    root: Path,
    old: str,
    new: str,
    *,
    include_text: bool = True,
) -> tuple[dict[str, list[Reference]], list[Reference], list[Reference]]:
    """Group references by file, beside the ones that cannot be rewritten and the prose ones.

    Three groups, because they need three different responses:

      ``by_file``    resolved identifiers, positioned, safe to rewrite
      ``unresolved`` non-AST sites in *code* -- rewrite needs a language-aware tool, block
      ``docs``       non-AST sites in *prose* -- a human may want to update the wording

    Separated from the write so a caller can inspect the blast radius -- which files, how
    many identifiers -- *before* anything is modified. A capability that can only be invoked
    blind is not auditable, and this project's own record is that the interesting failures are
    the ones nobody looked at first.
    """
    refs = find_references(root, old, include_text=include_text)
    by_file: dict[str, list[Reference]] = {}
    unresolved: list[Reference] = []
    docs: list[Reference] = []
    # The text pass matches by regex over whole lines, so it re-reports every line that the
    # AST pass already resolved. MEASURED on the rich fixture: 22 references, of which 10
    # are AST-resolved and 12 are ``text`` -- but the 12 sit on the *same lines* as 7 of the
    # 10 (``rich/console.py:1326`` is both ``attribute`` and ``text``). Treating the text tier
    # as independent sites made the executor refuse every rename in the fixture, which is
    # the refusal this module exists to replace.
    #
    # So a text hit counts as unresolved only when its line carries no resolved identifier.
    # The lines that do carry one are already covered by the rewrite, and the regex match on
    # them is redundant rather than additional.
    for r in refs:
        if r.kind in _REWRITABLE_KINDS and r.col is not None and r.end_col is not None:
            by_file.setdefault(r.path, []).append(r)
    resolved_lines = {(r.path, r.line) for r in refs
                      if r.kind in _REWRITABLE_KINDS and r.col is not None}
    leftovers = [r for r in refs
                 if r.kind not in _REWRITABLE_KINDS
                 and (r.path, r.line) not in resolved_lines]
    # Split the leftovers by whether they are prose. The classification runs once over the
    # files that actually have a leftover, so the common case pays nothing.
    doc_lines = (_documentation_lines(root, sorted({r.path for r in leftovers}))
                 if leftovers else set())
    for r in leftovers:
        (docs if (r.path, r.line) in doc_lines else unresolved).append(r)
    return by_file, unresolved, docs


def apply_multi_file_rename(
    root: Path,
    old: str,
    new: str,
    *,
    include_text: bool = True,
    dry_run: bool = True,
) -> MultiFileRenameResult:
    """Rewrite ``old`` to ``new`` at every AST-resolved reference.

    ``dry_run`` defaults to ``True`` because this is the first code in the reflex that writes
    to files *it was not asked to edit*, and the reflex's own contract (``dry_run`` in
    ``ReflexAgent.run``) is opt-in for writing. A default of ``False`` here would mean any
    caller that forgot the flag rewrote eleven files.

    Refuses outright -- returning ``blocked`` with nothing written -- when the plan contains
    an unresolved reference. A partial rename that still parses is harder to notice than a
    refusal, and the consistency gate's whole purpose is to avoid that class of quiet
    breakage. The list of blockers is returned so the caller can escalate with it.
    """
    result = MultiFileRenameResult(old=old, new=new)

    if not old or not new or old == new:
        result.blocked = f"rename {old!r} -> {new!r} is a no-op or incomplete"
        return result

    by_file, unresolved, docs = plan_multi_file_rename(
        root, old, new, include_text=include_text)
    result.unresolved = unresolved
    result.doc_leftovers = docs

    if unresolved:
        shown = "; ".join(str(r) for r in unresolved[:5])
        more = f" (+{len(unresolved) - 5} more)" if len(unresolved) > 5 else ""
        result.blocked = (
            f"{old!r} -> {new!r} touches {len(unresolved)} site(s) that are not resolved "
            f"identifiers, so they cannot be rewritten without guessing: {shown}{more}. "
            f"A partial rename would leave the tree referencing a name that no longer "
            f"exists, so nothing was written. These sites need a human, or a "
            f"language-specific rewriter that understands what they are."
        )
        return result

    if not by_file:
        return result

    for rel, refs in by_file.items():
        path = root / rel
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            result.blocked = (f"could not read {rel} to rename {old!r}: {exc}; nothing "
                              f"was written")
            return result

        # Group by line, right-to-left, so earlier offsets stay valid as the line is
        # rewritten. Left-to-right would shift every later column on the same line and
        # corrupt the remainder of it -- the same failure the reference-collection key
        # had, one layer down.
        per_line: dict[int, list[Reference]] = {}
        for r in refs:
            per_line.setdefault(r.line, []).append(r)

        lines = source.splitlines(keepends=True)
        rewritten = 0
        for lineno in sorted(per_line, reverse=True):
            idx = lineno - 1
            if idx < 0 or idx >= len(lines):
                continue
            line = lines[idx]
            hits = sorted(per_line[lineno], key=lambda r: r.col or 0, reverse=True)
            for r in hits:
                # Verify the span actually contains the old name before replacing. The
                # offsets came from an AST parse of this exact text, so a mismatch means
                # the file changed under us -- and replacing anyway is how a rename
                # corrupts an unrelated line.
                segment = line[r.col:r.end_col]
                if segment.strip() != old:
                    result.blocked = (
                        f"{rel}:{lineno} was expected to contain {old!r} but the text at "
                        f"the recorded position is {segment.strip()!r}; the file changed "
                        f"since it was scanned, so nothing was written"
                    )
                    return result
                line = line[:r.col] + new + line[r.end_col:]
                rewritten += 1
            lines[idx] = line

        if not rewritten:
            continue
        if not dry_run:
            try:
                path.write_text("".join(lines), encoding="utf-8")
            except OSError as exc:
                result.blocked = (f"could not write {rel}: {exc}; the rename is incomplete")
                return result
        result.edited[rel] = rewritten

    return result


def renamed_files_still_parse(root: Path, files: list[str]) -> tuple[bool, list[str]]:
    """Whether every rewritten file still parses as Python.

    Not part of :func:`apply_multi_file_rename` on purpose: the reflex's contract is to
    report what it did and let the caller's verification decide, and a parse check here
    would be a second, weaker opinion about a question ``sylphae.exec.verify`` answers with
    the actual test suite. Exposed separately so a caller that only wants a cheap smoke
    check does not have to run pytest for it.
    """
    broken: list[str] = []
    for rel in files:
        path = root / rel
        try:
            ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError) as exc:
            broken.append(f"{rel}: {exc}")
    return (not broken), broken
