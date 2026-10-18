"""Writes that survive a crash, and a lock that keeps two writers apart.

A whole file is written to a temporary file in the same directory, flushed,
fsynced, moved over the old one with os.replace, and the directory is fsynced.
A reader sees the old file or the new one, never half of either.

A log record is one os.write of one line under an exclusive lock, then fsync.
If the log does not end in a newline (an earlier write was torn), a newline is
written first, so the torn fragment stays its own line and the new record is whole.

The lock is fcntl.flock on a ``.lock`` file beside the data. It is advisory: every
AngularAI process honors it, other programs need not. Windows has no flock here,
and the lock is a no-op there.
"""

import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

if sys.platform != "win32":
    import fcntl

LOCK_NAME = ".lock"


@contextmanager
def locked(root: Path, *, shared: bool = False) -> Iterator[None]:
    """Hold the cortex lock. Writers take it exclusive, readers shared."""

    if sys.platform == "win32":
        yield
        return
    path = root / LOCK_NAME
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        if not shared:
            raise
        # A reader of a cortex it cannot write still reads, without the lock.
        yield
        return
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def atomic_write(path: Path, data: bytes, mode: int = 0o600) -> None:
    """Replace path with data, or leave it as it was."""

    parent = path.parent
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    fsync_dir(parent)


def append_line(path: Path, line: bytes, mode: int = 0o600) -> None:
    """Append one record. Call it under an exclusive lock."""

    if b"\n" in line:
        raise ValueError("A record is one line.")
    descriptor = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, mode)
    try:
        size = os.fstat(descriptor).st_size
        data = line + b"\n"
        if size:
            os.lseek(descriptor, size - 1, os.SEEK_SET)
            if os.read(descriptor, 1) != b"\n":
                data = b"\n" + data
        _write_all(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, mode)


def fsync_dir(path: Path) -> None:
    if sys.platform == "win32":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        view = view[written:]
