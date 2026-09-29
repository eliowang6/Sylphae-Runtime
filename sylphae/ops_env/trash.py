"""Deletion with undo semantics, because the goal makes it a hard constraint:

    用户文件删除只允许走回收站语义

A delete that bypasses the recycle bin is data loss waiting for a test to run it,
so this module's contract is that a caller who asks for a delete either gets
something recoverable or gets a refusal -- never a silent permanent removal.

On Windows the recycle bin is reachable through ``SHFileOperation`` with
``FOF_ALLOWUNDO``.  On other platforms ``send2trash`` is used when installed, and
when it is not installed the delete is *refused* rather than falling back to
``unlink``: a fallback that permanently deletes would mean "the safety property
holds on my laptop and not on the server".
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrashResult:
    """What happened to a deletion request.

    ``recovered`` names the recycle-bin entry when the delete was undoable, so an
    audit reader can find the thing that was removed without grepping a log.
    """

    path: str
    moved_to: str | None
    permanent: bool
    refused: bool
    reason: str = ""


def _windows_send_to_trash(path: Path) -> TrashResult:
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):  # pragma: no cover - platform-specific
        _fields_ = [
            ("hwnd", wintypes.HWND),
            ("wFunc", wintypes.UINT),
            ("pFrom", wintypes.LPCWSTR),
            ("pTo", wintypes.LPCWSTR),
            ("fFlags", wintypes.WORD),
            ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", wintypes.LPVOID),
            ("lpszProgressTitle", wintypes.LPCWSTR),
        ]

    FO_DELETE = 0x0003
    FOF_ALLOWUNDO = 0x0040
    FOF_NOCONFIRMATION = 0x0010
    FOF_SILENT = 0x0004

    op = SHFILEOPSTRUCTW()
    op.hwnd = None
    op.wFunc = FO_DELETE
    # The API wants a double-null-terminated list of names.
    op.pFrom = str(path) + "\0\0"
    op.pTo = None
    op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT
    op.fAnyOperationsAborted = False
    op.hNameMappings = None
    op.lpszProgressTitle = None

    code = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if code != 0:
        return TrashResult(
            str(path), None, False, True, f"SHFileOperationW returned {code}"
        )
    return TrashResult(str(path), "recycle-bin", False, False)


def _posix_send_to_trash(path: Path) -> TrashResult:
    try:
        import send2trash  # noqa: PLC0415 - optional dependency
    except ImportError:
        return TrashResult(
            str(path),
            None,
            False,
            True,
            "send2trash is not installed, and permanent deletion is not a fallback",
        )
    try:
        send2trash.send2trash(str(path))
    except Exception as exc:  # noqa: BLE001 - report, do not raise
        return TrashResult(str(path), None, False, True, f"send2trash failed: {exc}")
    return TrashResult(str(path), "trash", False, False)


def send_to_trash(path: Path) -> TrashResult:
    """Delete ``path`` through the recycle bin, or refuse.

    Never permanently removes.  A directory is handled by the same path: the
    recycle bin accepts directories, and a separate code path for them would be a
    second chance to get it wrong.
    """
    path = Path(path)
    if not path.exists():
        return TrashResult(str(path), None, False, True, "path does not exist")
    if path.is_symlink():
        #: A symlink points somewhere the caller did not name.  Removing the link
        #: itself is safe, but removing *through* it is not, so the link is
        #: refused rather than followed -- and ``shutil.rmtree`` on a symlinked
        #: directory would remove the target's contents, which is exactly the
        #: failure this module exists to prevent.
        return TrashResult(
            str(path), None, False, True, "refusing to delete a symlink"
        )
    import sys

    if sys.platform == "win32":
        return _windows_send_to_trash(path)
    return _posix_send_to_trash(path)


def move_to_scratch(path: Path, scratch: Path) -> TrashResult:
    """Move ``path`` under ``scratch`` instead of deleting it.

    A scratch move is the portable undo: it works on every platform, needs no
    optional dependency, and is what a playbook uses when the recycle bin is not
    acceptable (a test asserting the file is gone cannot see a recycle-bin entry)
    but permanence still is not.
    """
    path = Path(path)
    scratch = Path(scratch)
    if not path.exists():
        return TrashResult(str(path), None, False, True, "path does not exist")
    scratch.mkdir(parents=True, exist_ok=True)
    target = scratch / path.name
    counter = 1
    while target.exists():
        target = scratch / f"{path.stem}.{counter}{path.suffix}"
        counter += 1
    try:
        shutil.move(str(path), str(target))
    except OSError as exc:
        return TrashResult(str(path), None, False, True, f"move failed: {exc}")
    return TrashResult(str(path), str(target), False, False)
