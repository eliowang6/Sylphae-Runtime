"""Byte-faithful text I/O for source files this project edits.

An edit is supposed to change the region it names and nothing else. Two things defeat that
silently on this platform, and both were measured through a public entry point before this
module existed:

* **Newline translation.** ``Path.read_text`` uses universal newlines, so a CRLF file
  arrives as LF; ``Path.write_text`` translates ``\\n`` to ``os.linesep`` on the way out, so
  an LF file leaves as CRLF. A one-line edit therefore rewrites every line of the file --
  and with ``core.autocrlf=true``, which is set on this machine, ``git status`` cannot show
  it. MEASURED 2026-09-23: a 3-line LF file read and rewritten through
  ``read_text``/``write_text`` came back with 3 CRLF lines.
* **Undecodable bytes.** ``errors="replace"`` turns a byte that is not valid UTF-8 into
  U+FFFD, and the replacement is written back, so the original byte is gone.
  MEASURED 2026-09-23 through ``sylphae.coprocessor.executor``: a latin-1 ``0xe9`` came back as
  ``ef bf bd`` while the step still reported ``applied=True syntax_ok=True error=None``.

``newline=""`` disables both translations and ``errors="surrogateescape"`` lets an
undecodable byte round-trip through a Python ``str`` unchanged. Together they make
``read_source`` / ``write_source`` a round trip: for any file this can read,
``write_source(p, read_source(p))`` leaves the bytes identical.

**Writers must use this module.** ``sylphae/exec/patch.py`` and ``sylphae/coprocessor/executor.py``
both edit files in place and both go through here. ``sylphae/exec/verify.py`` defines the same
encoding constants but only *reads* (and hands the encoding to a subprocess), so it carries
no byte-fidelity obligation and is deliberately left alone rather than changed without a
measurement.

``write_source`` is atomic: the text goes to a temporary file in the *same directory* and is
then ``os.replace``-d over the target, so a crash or a full disk leaves either the old file
or the new file, never a half-written one. It does not create missing parent directories --
callers that create a file are expected to have decided the directory should exist.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

__all__ = [
    "ENCODING",
    "ERRORS",
    "NEWLINE",
    "line_ending",
    "read_source",
    "with_ending",
    "write_source",
]

ENCODING = "utf-8"
ERRORS = "surrogateescape"
NEWLINE = ""

#: CRLF, as it appears inside a string read with ``newline=""``.
CRLF = "\r\n"
LF = "\n"


def read_source(path: Path) -> str:
    """Read ``path`` without translating newlines or discarding undecodable bytes.

    Raises ``OSError`` if the file cannot be read; callers decide how to report that.
    """
    with open(path, "r", encoding=ENCODING, errors=ERRORS, newline=NEWLINE) as handle:
        return handle.read()


def write_source(path: Path, text: str) -> None:
    """Atomically replace ``path`` with ``text``, byte for byte, newlines included."""
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".sylphae-", suffix=".tmp"
    )
    try:
        with os.fdopen(
            fd, "w", encoding=ENCODING, errors=ERRORS, newline=NEWLINE
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        # Never leave the temporary file behind, whatever went wrong.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def line_ending(text: str) -> str:
    """The newline convention ``text`` already uses, for text inserted into it.

    Decided by majority, with ties going to CRLF, so a file that is mostly CRLF keeps CRLF
    even if it contains a stray bare LF. An empty file, and a file that uses only bare LF,
    get ``"\\n"`` -- there is no evidence for anything else, and guessing CRLF would mean a
    one-line edit on a Unix file rewrites every line.

    MEASURED 2026-09-23: without this, appending a generated block to a CRLF file produced a
    file with CRLF in the body and LF in the appended block.
    """
    crlf = text.count(CRLF)
    bare_lf = text.count(LF) - crlf
    if crlf and crlf >= bare_lf:
        return CRLF
    return LF


def with_ending(text: str, ending: str) -> str:
    """``text`` with every line break written as ``ending``.

    Idempotent: an existing CRLF is normalised to LF before the substitution, so calling
    this twice, or calling it on text that already uses ``ending``, does not double the
    carriage returns.
    """
    return text.replace(CRLF, LF).replace(LF, ending)
