"""Tests for ``sylphae/exec/textio.py`` -- the byte-faithful I/O contract.

An edit is supposed to change the region it names and nothing else. Two things break that
silently, and both are asserted here against raw bytes rather than against a re-read
string, because a re-read through ``read_text`` would hide exactly the change being tested.

.. note::
   These tests are *red on Windows only* for the newline cases. ``Path.write_text``
   translates ``\\n`` to ``os.linesep``, which is the defect; on a platform whose
   ``os.linesep`` is ``\\n`` the same code is harmless, so the assertion would pass there
   without the contract. The undecodable-byte cases are platform independent.

Run from the repository root:

    python -m pytest tests/test_exec_textio.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sylphae.exec.textio import (  # noqa: E402
    line_ending,
    read_source,
    with_ending,
    write_source,
)

CRLF_SOURCE = b"a = 1\r\nb = 2\r\nc = 3\r\n"
LF_SOURCE = b"a = 1\nb = 2\nc = 3\n"
#: 0xe9 is 'e-acute' in latin-1 and is not valid UTF-8 on its own.
LATIN1_SOURCE = b"# caf\xe9 latte\nx = 1\n"


def _target(tmp_path: Path, payload: bytes, name: str = "src.py") -> Path:
    path = tmp_path / name
    path.write_bytes(payload)
    return path


# --------------------------------------------------------------------------------------
# read_source
# --------------------------------------------------------------------------------------


def test_read_source_keeps_crlf_visible_to_the_caller(tmp_path: Path) -> None:
    """Universal newlines would hand back LF, and the caller could never write it back."""
    text = read_source(_target(tmp_path, CRLF_SOURCE))
    assert text.count("\r\n") == 3, "the CRLF endings must survive the read"


def test_read_source_keeps_lf_visible_to_the_caller(tmp_path: Path) -> None:
    text = read_source(_target(tmp_path, LF_SOURCE))
    assert "\r" not in text


def test_read_source_keeps_an_undecodable_byte(tmp_path: Path) -> None:
    """``errors="replace"`` would give U+FFFD, which cannot be written back as 0xe9."""
    text = read_source(_target(tmp_path, LATIN1_SOURCE))
    assert "\udce9" in text, "the byte must survive as its surrogateescape surrogate"


def test_read_source_raises_for_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        read_source(tmp_path / "absent.py")


# --------------------------------------------------------------------------------------
# write_source
# --------------------------------------------------------------------------------------


def test_write_source_writes_lf_where_lf_was_asked_for(tmp_path: Path) -> None:
    """The defect: on Windows ``write_text`` turns this into CRLF, rewriting every line."""
    path = tmp_path / "out.py"
    write_source(path, "a = 1\nb = 2\n")
    assert path.read_bytes() == b"a = 1\nb = 2\n"


def test_write_source_writes_crlf_where_crlf_was_asked_for(tmp_path: Path) -> None:
    path = tmp_path / "out.py"
    write_source(path, "a = 1\r\nb = 2\r\n")
    assert path.read_bytes() == b"a = 1\r\nb = 2\r\n"


def test_write_source_keeps_an_undecodable_byte(tmp_path: Path) -> None:
    path = tmp_path / "out.py"
    write_source(path, "x = 1  # caf\udce9\n")
    assert path.read_bytes() == b"x = 1  # caf\xe9\n"


def test_write_source_leaves_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "out.py"
    write_source(path, "x = 1\n")
    assert [p.name for p in tmp_path.iterdir()] == ["out.py"]


def test_write_source_does_not_create_a_missing_directory(tmp_path: Path) -> None:
    """Documented behaviour: the caller decides whether the directory should exist."""
    with pytest.raises(OSError):
        write_source(tmp_path / "absent" / "out.py", "x = 1\n")


def test_write_source_reports_a_failed_write_instead_of_swallowing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full disk must surface, and must not leave a half-written target behind."""
    path = _target(tmp_path, LF_SOURCE)

    real_replace = __import__("os").replace

    def exploding_replace(src: str, dst: object) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr("sylphae.exec.textio.os.replace", exploding_replace)
    with pytest.raises(OSError):
        write_source(path, "replacement\n")
    monkeypatch.setattr("sylphae.exec.textio.os.replace", real_replace)

    assert path.read_bytes() == LF_SOURCE, "the original file must be untouched"
    assert [p.name for p in tmp_path.iterdir()] == ["src.py"], (
        "the temporary file must not be left behind"
    )


# --------------------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [CRLF_SOURCE, LF_SOURCE, LATIN1_SOURCE, b"", b"no trailing newline", b"mixed\r\nand\nendings\n"],
    ids=["crlf", "lf", "latin1", "empty", "no-trailing-newline", "mixed"],
)
def test_round_trip_is_byte_identical(tmp_path: Path, payload: bytes) -> None:
    """The property the whole module exists for: writing back what was read changes nothing."""
    path = _target(tmp_path, payload)
    write_source(path, read_source(path))
    assert path.read_bytes() == payload


# --------------------------------------------------------------------------------------
# line_ending
# --------------------------------------------------------------------------------------


def test_line_ending_reads_the_convention_the_file_already_uses() -> None:
    assert line_ending("a\r\nb\r\n") == "\r\n"
    assert line_ending("a\nb\n") == "\n"


def test_line_ending_of_an_empty_file_is_lf() -> None:
    """There is no evidence for CRLF, and guessing it would rewrite a Unix file."""
    assert line_ending("") == "\n"
    assert line_ending("a = 1") == "\n"


def test_line_ending_goes_to_the_majority() -> None:
    """A single stray ending must not flip the whole file's convention."""
    assert line_ending("a\r\nb\r\nc\r\nd\n") == "\r\n", "3 CRLF beats 1 bare LF"
    assert line_ending("a\r\nb\nc\nd\n") == "\n", "1 CRLF loses to 3 bare LF"


def test_line_ending_ties_go_to_crlf() -> None:
    assert line_ending("a\r\nb\n") == "\r\n"


# --------------------------------------------------------------------------------------
# with_ending
# --------------------------------------------------------------------------------------


def test_with_ending_converts_lf_to_the_files_convention() -> None:
    assert with_ending("a = 1\nb = 2\n", "\r\n") == "a = 1\r\nb = 2\r\n"


def test_with_ending_is_idempotent() -> None:
    """Applying it twice must not produce ``\\r\\r\\n``."""
    once = with_ending("a = 1\nb = 2\n", "\r\n")
    assert with_ending(once, "\r\n") == once


def test_with_ending_leaves_single_line_text_alone() -> None:
    assert with_ending("2.1.1", "\r\n") == "2.1.1"


def test_with_ending_normalises_away_the_other_convention() -> None:
    """An LF file searched with a CRLF snippet must still match."""
    assert with_ending("a = 1\r\nb = 2\r\n", "\n") == "a = 1\nb = 2\n"
