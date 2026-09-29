"""The one place ``sylphae.ops_env`` actually launches a process.

Every command goes through here, so the safety checks cannot be bypassed by a
caller that forgot to call them.  The order matters: audit first, refuse second,
run third -- a refusal is written to the audit log *before* it is returned, which
means an attempted-but-blocked command is indistinguishable in the trail from one
that was attempted and failed.

Synchronous, not async, on purpose.  ``asyncio.create_subprocess_exec`` on Windows
spends most of its time in the same process-creation path that was measured
blocking the event loop for 441 ms in ``sylphae.agent.tools``; an ops command is
expected to take seconds, so the caller wants a bounded wait and a captured exit
code, not a coroutine that can be cancelled mid-write.  Callers wrap this in
``asyncio.to_thread`` when they need concurrency.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sylphae.ops_env.safety import (
    AuditLog,
    CommandResult,
    CommandSpec,
    Refusal,
    make_audit,
    validate_command,
)

#: The output ceiling, as characters of decoded text.  A diagnostic that exceeds
#: this is truncated and hashed; the hash is what makes the audit entry
#: reproducible without retaining a possibly-secret-laden transcript.
_TRUNCATE_CHARS = 8192


@dataclass
class EnvExecutor:
    """Runs allowlisted commands, capturing everything an audit needs.

    ``writable_roots`` is the *only* notion of "where may this write".  A
    playbook receives a scratch directory and passes it here, so a command's
    positional path argument is checked against a caller-chosen boundary rather
    than against the filesystem at large.
    """

    audit: AuditLog
    writable_roots: tuple[Path, ...] = ()
    default_timeout_s: float = 60.0
    fingerprint: dict[str, Any] | None = None

    def would_allow(self, spec: CommandSpec) -> Refusal | None:
        """Whether this command would run, without running it.

        Playbooks use this to *explain* a refusal in the report they hand back,
        rather than reporting a bare ``Refused``.  A user who reads "I would have
        run ``pip install --force-reinstall x`` but that verb is destructive" is
        being treated as an adult; one who reads "refused" is not.
        """
        spec = self._with_defaults(spec)
        return validate_command(spec)

    def run(self, spec: CommandSpec) -> CommandResult:
        spec = self._with_defaults(spec)
        refusal = validate_command(spec)
        if refusal is not None:
            record = make_audit(
                spec,
                exit_code=None,
                duration_ms=0.0,
                stdout="",
                stderr="",
                refusal=refusal,
                timed_out=False,
                fingerprint=self.fingerprint or {},
                truncate_chars=_TRUNCATE_CHARS,
            )
            self.audit.append(record)
            return CommandResult(
                command=spec.command,
                cwd=str(spec.cwd) if spec.cwd else "",
                exit_code=None,
                duration_ms=0.0,
                refusal=refusal,
                audit=record,
            )

        import time

        started = time.perf_counter()
        timed_out = False
        exit_code: int | None = None
        stdout = ""
        stderr = ""
        try:
            completed = subprocess.run(
                [spec.program, *spec.args],
                cwd=str(spec.cwd) if spec.cwd else None,
                capture_output=True,
                timeout=spec.timeout_s or self.default_timeout_s,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            exit_code = completed.returncode
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = None
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        except FileNotFoundError:
            #: The program is allowlisted but not installed.  This is a
            #: *diagnostic*, not an error to hide: an ops layer that reports
            #: "ruff is missing" has done half of category 7's job.
            stderr = f"{spec.program!r} is not installed on this machine"
        except OSError as exc:
            stderr = f"could not launch {spec.program!r}: {exc}"

        duration_ms = (time.perf_counter() - started) * 1000.0
        record = make_audit(
            spec,
            exit_code=exit_code,
            duration_ms=duration_ms,
            stdout=stdout,
            stderr=stderr,
            refusal=None,
            timed_out=timed_out,
            fingerprint=self.fingerprint or {},
            truncate_chars=_TRUNCATE_CHARS,
        )
        self.audit.append(record)
        return CommandResult(
            command=spec.command,
            cwd=str(spec.cwd) if spec.cwd else "",
            exit_code=exit_code,
            duration_ms=round(duration_ms, 1),
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            audit=record,
        )

    def _with_defaults(self, spec: CommandSpec) -> CommandSpec:
        if spec.writable_roots or not self.writable_roots:
            return spec
        return CommandSpec(
            program=spec.program,
            args=spec.args,
            cwd=spec.cwd,
            timeout_s=spec.timeout_s or self.default_timeout_s,
            writable_roots=self.writable_roots,
            authorisation=spec.authorisation,
        )
