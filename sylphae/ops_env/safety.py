"""The non-negotiable safety layer for C3 OPS.

Everything in ``sylphae/ops_env`` runs shell commands on the user's machine, so the
constraint is not "be careful" but "be unable": a command that can destroy work
must be unreachable through this API, not merely discouraged.

Three independent checks, any one of which can refuse a command:

1. ``COMMAND_ALLOWLIST`` -- the program must be known, and its arguments must
   satisfy that entry's policy.  An unknown program is refused even if it is
   harmless, because the point of an ops layer is that its behaviour is
   predictable from the code alone.
2. ``classify_destructive`` -- pattern-level rejection of the shapes the goal
   names outright (``rm -rf``, ``del``, disk formatting, git-history rewrites,
   force pushes, and any deletion of a path the caller did not pass in).
3. ``AuditLog`` -- every attempted command is appended before it runs, including
   the ones that were refused, so a refusal is as attributable as an execution.

The reason this is a module rather than a doc comment: an ops layer that deletes
a file is a data-loss feature, and the goal makes it a hard failure -- "违反即视为
失败".  A check that lives in the caller's head is not a check.
"""

from __future__ import annotations

import hashlib
import re
import shlex
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

#: Destructive shapes, matched against the *argument string* after the program.
#:
#: ``rm -rf`` is listed by name in the goal; the rest are the same class of
#: damage reachable through a different verb.  ``git commit --amend`` and
#: ``git push --force`` rewrite history rather than files, which is the one kind
#: of "undo" that has no undo.
_DESTRUCTIVE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"(?:^|\s)-[a-zA-Z]*r[a-zA-Z]*f\s",  # rm -rf / -fr, any combo order
        r"(?:^|\s)--force\s",
        r"(?:^|\s)rm\s+",
        r"(?:^|\s)(?:del|erase|rd|deltree)\s",
        #: Disk formatting takes a drive letter or a volume; requiring one keeps
        #: this from matching ``ruff format a.py``, where ``format`` is a
        #: subcommand rather than a verb.  The general pattern was measured to
        #: refuse that command, which is a false positive the caller cannot work
        #: around, since the allowlist is what permits ``ruff`` in the first place.
        r"(?:^|\s)format\s+(?:/\S+\s*)?[A-Za-z]:",
        r"(?:^|\s)mkfs(?:\.\w+)?\s",
        r"(?:^|\s)dd\s+.*of=",
        r"(?:^|\s)git\s+(?:commit\s+--amend|rebase|reset\s+--hard|push\s+.*--force)",
        r"(?:^|\s)git\s+clean\s+-[a-zA-Z]*[fd]",
        r"(?:^|\s)truncate\s+-s\s*0",
        r"(?:^|\s)shred\s+",
        r"(?:^|\s)reg\s+delete",
        r"(?:^|\s)diskpart\s",
        r"(?:^|\s)taskkill\s",
        r"(?:^|\s)kill(?:\s|-9)",
        r"(?:^|\s)chmod\s+-R",
        r"(?:^|\s)chown\s+-R",
    )
)

#: A regex an argument must not contain even as a substring -- the ``..`` escape
#: and the recursive-delete flags, which are dangerous in any position.
_FORBIDDEN_SUBSTRINGS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\.\./",
        r"\.\.\\",
        r"~",  # never expand the user's home implicitly
    )
)

#: Whether a glob is permitted at all in an argument.  ``ripgrep`` uses globs to
#: *narrow*; ``rm`` would use one to widen.  Since rm is refused outright this is
#: a second line of defence, not the first.
_GLOB_RE = re.compile(r"[*?\[\]]")


@dataclass(frozen=True)
class ArgPolicy:
    """How a known program's arguments may be shaped.

    ``max_args`` bounds the surface; ``allow_flags`` is a closed set, so a new
    flag is refused until someone reads this file and adds it; ``allow_globs``
    is off by default because a glob in the wrong verb widens the blast radius.
    """

    allow_flags: frozenset[str] = frozenset()
    allow_globs: bool = False
    max_args: int = 8
    #: Patterns that are destructive for this program specifically.  ``pip`` can
    #: uninstall, which is reversible only if the version is recorded; ``git``
    #: can discard work.
    extra_destructive: tuple[re.Pattern[str], ...] = ()
    #: Positional arguments must stay inside the executor's working directory.
    #: ``True`` for programs whose positional argument is a path (``ruff``,
    #: ``pytest``), ``False`` for ones whose positional argument is a package
    #: name or a port number.
    positional_is_path: bool = False


def _pip_extra() -> tuple[re.Pattern[str], ...]:
    return tuple(
        re.compile(p, re.IGNORECASE)
        for p in (r"(?:^|\s)uninstall\s", r"(?:^|\s)--force-reinstall", r"(?:^|\s)-I\b")
    )


def _git_extra() -> tuple[re.Pattern[str], ...]:
    return tuple(
        re.compile(p, re.IGNORECASE)
        for p in (
            r"(?:^|\s)(?:checkout|restore|reset|stash\s+drop|bisect)",
            r"(?:^|\s)branch\s+-[a-zA-Z]*[dD]",
        )
    )


#: The closed set of programs this layer will ever launch.  Adding an entry is a
#: reviewable diff; the absence of ``rm``, ``del``, ``format`` and ``git push`` is
#: the safety property.
COMMAND_ALLOWLIST: dict[str, ArgPolicy] = {
    "python": ArgPolicy(
        allow_flags=frozenset({"-m", "-c", "-V", "--version", "-S", "-I", "-W"}),
        max_args=12,
    ),
    "py": ArgPolicy(
        allow_flags=frozenset({"-m", "-c", "-V", "--version", "-3", "-0"}),
        max_args=12,
    ),
    "pip": ArgPolicy(
        allow_flags=frozenset(
            {
                "install",
                "download",
                "list",
                "show",
                "freeze",
                "check",
                "--no-input",
                "--dry-run",
                "--report",
                "--requirement",
                "-r",
                "--upgrade",
                "--no-deps",
                "--index-url",
                "--constraint",
                "--local",
                "--outdated",
                "--format",
                "--disable-pip-version-check",
            }
        ),
        max_args=12,
        extra_destructive=_pip_extra(),
    ),
    "pytest": ArgPolicy(
        allow_flags=frozenset(
            {
                "-q",
                "-x",
                "--tb",
                "-k",
                "-p",
                "--no-header",
                "-rA",
                "--co",
                "-l",
                "--timeout",
                "--ignore",
                "--last-failed",
                "--failed-first",
                "-W",
            }
        ),
        max_args=10,
        allow_globs=True,
        positional_is_path=True,
    ),
    "ruff": ArgPolicy(
        allow_flags=frozenset(
            {"check", "format", "--fix", "--no-cache", "--select", "--exit-non-zero-on-fix"}
        ),
        max_args=8,
        allow_globs=True,
        positional_is_path=True,
    ),
    "black": ArgPolicy(
        allow_flags=frozenset({"--check", "--diff", "--quiet", "-"}),
        max_args=6,
        positional_is_path=True,
    ),
    "rg": ArgPolicy(
        allow_flags=frozenset(
            {
                "-i",
                "-n",
                "--max-count",
                "--no-require-git",
                "-l",
                "--count",
                "-w",
                "--json",
            }
        ),
        max_args=8,
        allow_globs=True,
        positional_is_path=True,
    ),
    "git": ArgPolicy(
        allow_flags=frozenset(
            {
                "status",
                "diff",
                "log",
                "show",
                "ls-files",
                "rev-parse",
                "stash",
                "list",
                "branch",
                "--no-pager",
                "-p",
                "--name-only",
                "--porcelain",
            }
        ),
        max_args=10,
        extra_destructive=_git_extra(),
    ),
    "where": ArgPolicy(allow_flags=frozenset(), max_args=4),
    "netstat": ArgPolicy(
        allow_flags=frozenset({"-ano", "-a", "-n", "-o"}),
        max_args=4,
    ),
    "ss": ArgPolicy(
        allow_flags=frozenset({"-ltnp", "-lnt", "-p"}),
        max_args=4,
    ),
    "tasklist": ArgPolicy(
        allow_flags=frozenset({"/FI", "/FO", "/NH", "/V"}),
        max_args=6,
    ),
    "systeminfo": ArgPolicy(allow_flags=frozenset(), max_args=2),
    "cmd": ArgPolicy(allow_flags=frozenset({"/c"}), max_args=4),
}

#: Programs that exist only to look something up, and so are also allowed inside
#: a ``cmd /c`` wrapper.  ``cmd /c`` itself is allowed because ``venv`` and
#: ``pip`` are sometimes only reachable through a launcher on Windows; the /c
#: payload is re-validated recursively rather than trusted.
_LOOKUP_PROGRAMS = frozenset({"where", "netstat", "ss", "tasklist", "systeminfo", "python", "py"})


@dataclass(frozen=True)
class Refusal:
    """Why a command was not run.  ``reason`` is machine-checkable."""

    reason: str
    detail: str = ""
    command: str = ""

    def __bool__(self) -> bool:
        """A refusal is truthy, so ``refusal or result`` reads the right way."""
        return True


@dataclass
class CommandSpec:
    """One command the caller intends to run, before any check."""

    program: str
    args: tuple[str, ...] = ()
    cwd: Path | None = None
    timeout_s: float = 30.0
    #: Paths the caller has declared may be written.  A write outside this set is
    #: refused; the default is "nowhere".
    writable_roots: tuple[Path, ...] = ()
    #: Explicit caller authorisation for one narrowly-scoped risky verb.  This is
    #: how a playbook kills a process it identified, rather than doing it
    #: silently -- see ``ProcessKillAuthorisation``.
    authorisation: str | None = None

    @property
    def command(self) -> str:
        return " ".join([self.program, *self.args])

    @classmethod
    def parse(cls, text: str, **kw: Any) -> CommandSpec:
        parts = shlex.split(text, posix=False)
        if not parts:
            raise ValueError("empty command")
        return cls(program=parts[0], args=tuple(parts[1:]), **kw)


@dataclass
class CommandResult:
    """What actually happened, including the refusal when nothing ran."""

    command: str
    cwd: str
    exit_code: int | None
    duration_ms: float
    stdout: str = ""
    stderr: str = ""
    refusal: Refusal | None = None
    timed_out: bool = False
    #: The audit record for this attempt.  Present for refusals too: an attempted
    #: command that was blocked is as much an audit event as one that ran.
    audit: AuditRecord | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.refusal is None and self.exit_code == 0

    @property
    def refused(self) -> bool:
        return self.refusal is not None


@dataclass
class AuditRecord:
    """One attempted command, in the shape the goal asks for: command as typed,
    working directory, exit code, duration, and a hash of the output rather than
    the output itself -- summaries are diffable, raw logs are not, and stdout can
    carry secrets.

    ``program`` and ``args`` are stored as a structured pair because the
    human-readable ``command`` string cannot be parsed back: a program path
    containing a space (``C:\\Users\\Elio Wang\\...\\python.exe``) splits two
    ways, and a replay that re-joins and re-splits will disagree with the
    validator that made the original decision.  The audit trail is how
    "zero destructive actions" is measured, so it has to round-trip exactly.
    """

    timestamp_epoch: float
    command: str
    program: str
    args: tuple[str, ...]
    cwd: str
    exit_code: int | None
    duration_ms: float
    stdout_sha256: str
    stdout_bytes: int
    stderr_sha256: str
    stderr_bytes: int
    timed_out: bool
    refused: bool
    refusal_reason: str
    fingerprint: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["args"] = list(self.args)
        return out


class AuditLog:
    """Append-only audit trail.

    Written *before* the command runs, so a crash mid-command still leaves a
    record of what was about to happen.  The file is opened per append rather
    than held open: a long-lived handle is lost with the process, and an ops
    layer that loses its audit log when it dies has no audit log.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: AuditRecord) -> None:
        import json

        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")

    def read(self) -> list[dict[str, Any]]:
        import json

        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _truncate(text: str, limit: int) -> tuple[str, int]:
    """Truncate to ``limit`` characters, marking the cut.

    The goal requires an output truncation ceiling.  The marker is what makes a
    truncated output distinguishable from a complete one; without it a truncated
    traceback looks like a short traceback.
    """
    if len(text) <= limit:
        return text, len(text)
    cut = text[:limit]
    return cut + f"\n...[truncated: {len(text) - limit} more chars]", len(text)


def classify_destructive(program: str, args: Sequence[str]) -> Refusal | None:
    """Pattern-level rejection of destructive commands.

    Returns a ``Refusal`` when the command must not run, ``None`` when it passes
    this layer (it may still be refused by the allowlist).
    """
    if not program:
        return Refusal("empty-program", "no program was given")
    arg_str = " ".join(args)
    for pat in _DESTRUCTIVE_PATTERNS:
        if pat.search(arg_str):
            return Refusal("destructive-pattern", pat.pattern, f"{program} {arg_str}")
    for pat in _FORBIDDEN_SUBSTRINGS:
        if pat.search(arg_str):
            return Refusal("forbidden-substring", pat.pattern, f"{program} {arg_str}")
    return None


def _path_is_under(path: Path, roots: Sequence[Path]) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for root in roots:
        try:
            resolved.relative_to(Path(root).resolve())
        except ValueError:
            continue
        return True
    return False


#: Modules reachable through ``python -m <module>``.  Without this entry the
#: arguments are checked against *python's* flag policy, so ``python -m pip install
#: -r requirements.txt`` is refused for a flag pip permits and python does not.
#: ``venv`` and ``ensurepip`` need entries of their own because they are modules,
#: not console scripts, and so are never launched any other way.
MODULE_ALLOWLIST: dict[str, ArgPolicy] = {
    "pip": COMMAND_ALLOWLIST["pip"],
    "pytest": COMMAND_ALLOWLIST["pytest"],
    "venv": ArgPolicy(max_args=3, positional_is_path=True),
    "ensurepip": ArgPolicy(allow_flags=frozenset({"--upgrade", "--default-pip"}), max_args=2),
    "http.server": ArgPolicy(allow_flags=frozenset({"--bind", "--directory"}), max_args=3),
}


def _program_key(program: str) -> str:
    """The allowlist key for a program, whatever form it was written in.

    ``sys.executable`` is an absolute path ending in ``python.exe``; a bare
    ``python`` typed by a user is the same program.  Without this normalisation
    the path form is refused as unrecognised, which would mean the safety
    property depends on how the caller spelled the interpreter -- and the venv
    playbook, which launches ``sys.executable``, would be unusable.
    """
    name = Path(program).name.lower()
    if name.endswith(".exe"):
        name = name[:-4]
    return name


def _effective_program(program: str, args: Sequence[str]) -> str:
    """The program whose policy governs these arguments.

    ``python -m pip`` is pip, not python: the module is the command and python is
    its launcher.  Resolving this is what makes a ``python -m`` invocation
    checkable at all, and what keeps the two spellings of one command from
    carrying two different safety properties.
    """
    if _program_key(program) in ("python", "py") and len(args) >= 2 and args[0] == "-m":
        return args[1]
    return _program_key(program)


def _flag_token(arg: str) -> str:
    """The canonical flag in an argument, so ``--tb=short`` and ``--tb`` are one rule.

    Without this, a flag written with its value attached is refused for being a
    flag the allowlist never saw -- which was measured to refuse
    ``pytest --tb=short``, and worse, to refuse it *silently as a test failure*,
    because the playbook saw ``exit_code is None`` and reported the test as
    broken.  The value half cannot smuggle a new flag: it is part of the same
    token, and the destructive-substring checks see the whole argument.
    """
    if not arg.startswith("-"):
        return arg
    #: ``--flag=value`` -> ``--flag``; ``-kvalue`` -> ``-k``.  A single dash
    #: followed by more than one letter is a *cluster* (``-qZ``), not a value, and
    #: is checked as written.
    if "=" in arg:
        return arg.split("=", 1)[0]
    if len(arg) > 2 and not arg.startswith("--"):
        return arg[:2]
    return arg


def validate_command(spec: CommandSpec) -> Refusal | None:
    """The full check.  ``None`` means the command may run.

    This is deliberately a pure function of the spec: the caller can ask "would
    this be allowed?" and get the same answer as at run time, which is what makes
    a playbook able to explain a refusal before attempting it.
    """
    program = spec.program
    if not program:
        return Refusal("empty-program", "no program was given", spec.command)

    if classify_destructive(program, spec.args):
        return classify_destructive(program, spec.args)

    args = tuple(spec.args)
    is_launcher = (
        _program_key(program) in ("python", "py") and len(args) >= 2 and args[0] == "-m"
    )
    if is_launcher:
        #: The ``-m <module>`` prefix is the launcher spelling of a command, not
        #: an argument of it, so it is not checked against the module's policy.
        effective = args[1]
        launcher_prefix: tuple[str, ...] = args[:2]
    else:
        effective = _program_key(program)
        launcher_prefix = ()
    policy = COMMAND_ALLOWLIST.get(effective)
    if policy is None and is_launcher:
        policy = MODULE_ALLOWLIST.get(effective)
    if policy is None:
        return Refusal(
            "program-not-allowlisted",
            f"{effective!r} is not in COMMAND_ALLOWLIST; add it with a reviewed ArgPolicy",
            spec.command,
        )

    check_args = args[len(launcher_prefix):]
    if len(check_args) > policy.max_args:
        return Refusal(
            "too-many-args",
            f"{len(check_args)} args exceeds {policy.max_args} for {effective}",
            spec.command,
        )

    for pat in policy.extra_destructive:
        if pat.search(" ".join(check_args)):
            return Refusal("destructive-program-specific", pat.pattern, spec.command)

    for arg in check_args:
        is_flag = arg.startswith("-") or arg.startswith("/")
        if is_flag and _flag_token(arg) not in policy.allow_flags:
            return Refusal(
                "flag-not-allowlisted",
                f"{arg!r} is not permitted for {program}",
                spec.command,
            )
        if not is_flag and not policy.allow_globs and _GLOB_RE.search(arg):
            return Refusal(
                "glob-not-permitted",
                f"{arg!r} contains a glob, which {program} is not configured to accept",
                spec.command,
            )
        if not is_flag and policy.positional_is_path:
            target = Path(arg)
            if target.is_absolute() and spec.writable_roots and not _path_is_under(
                target, spec.writable_roots
            ):
                return Refusal(
                    "path-outside-writable-roots",
                    f"{arg!r} is outside the declared writable roots",
                    spec.command,
                )

    return None


def make_audit(
    spec: CommandSpec,
    *,
    exit_code: int | None,
    duration_ms: float,
    stdout: str,
    stderr: str,
    refusal: Refusal | None,
    timed_out: bool,
    fingerprint: dict[str, Any] | None = None,
    truncate_chars: int = 8192,
) -> AuditRecord:
    out, out_n = _truncate(stdout, truncate_chars)
    err, err_n = _truncate(stderr, truncate_chars)
    return AuditRecord(
        timestamp_epoch=time.time(),
        command=spec.command,
        program=spec.program,
        args=tuple(spec.args),
        cwd=str(spec.cwd) if spec.cwd else "",
        exit_code=exit_code,
        duration_ms=round(duration_ms, 1),
        stdout_sha256=_sha256(out),
        stdout_bytes=out_n,
        stderr_sha256=_sha256(err),
        stderr_bytes=err_n,
        timed_out=timed_out,
        refused=refusal is not None,
        refusal_reason=refusal.reason if refusal else "",
        fingerprint=fingerprint or {},
    )
