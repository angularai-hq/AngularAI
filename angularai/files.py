"""File access inside the agent's directory that a swapped path cannot redirect.

Every open walks from the root one component at a time, each with O_NOFOLLOW and a
directory file descriptor (openat), so a link put in place after the path was checked
is refused instead of followed. Linux and macOS support this.

A read opens with O_NONBLOCK, so a FIFO cannot hang it, and then refuses anything
that fstat does not call a regular file. Reads stop at a byte cap.

A write never writes into an existing file. It writes a new file beside it and
renames it over the name (renameat). A hard link to the old file, inside the
directory or not, keeps the old content: a write changes this name only. The mode of
the old file is kept.

A hard link cannot be told apart from its other names by path, so a read follows it.
Only the user, or a program outside the agent, can make one that points out: run's
cp checks that its source is inside.
"""

import codecs
import errno
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_FILE = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
_SNIFF = 8192


class Refused(Exception):
    """A sentence for the model. The file was not opened, read, or changed."""


def _parts(root: Path, path: Path) -> tuple[str, ...]:
    parts = path.relative_to(root).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise Refused("That path leaves the directory.")
    return parts


@contextmanager
def _directory(root: Path, parts: tuple[str, ...], create: bool = False) -> Iterator[int]:
    """A descriptor for root/parts, walked without following a single link."""

    fd = os.open(root, _DIR)
    try:
        for part in parts:
            try:
                child = os.open(part, _DIR, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise Refused("That path is not there.") from None
                os.mkdir(part, 0o777, dir_fd=fd)
                child = os.open(part, _DIR, dir_fd=fd)
            except OSError as exc:
                raise _changed(exc) from exc
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def _changed(exc: OSError) -> Refused:
    if exc.errno in (errno.ELOOP, errno.ENOTDIR, errno.EMLINK):
        return Refused("That path is a link or changed while it was used. It was not followed.")
    return Refused("That path could not be opened.")


@contextmanager
def open_regular(root: Path, path: Path) -> Iterator[int]:
    """A read descriptor for a regular file inside root, or Refused."""

    parts = _parts(root, path)
    with _directory(root, parts[:-1]) as parent:
        try:
            fd = os.open(parts[-1], _FILE, dir_fd=parent)
        except FileNotFoundError:
            raise Refused("That path is not there.") from None
        except OSError as exc:
            # A socket will not open at all (ENXIO). Name what is there instead.
            try:
                found = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False).st_mode
            except OSError:
                found = 0
            if found and not stat.S_ISLNK(found) and not stat.S_ISREG(found):
                raise Refused(
                    f"That path is not a regular file ({_kind(found)}). It was not read."
                ) from exc
            raise _changed(exc) from exc
    try:
        mode = os.fstat(fd).st_mode
        if not stat.S_ISREG(mode):
            raise Refused(f"That path is not a regular file ({_kind(mode)}). It was not read.")
        os.set_blocking(fd, True)
        yield fd
    finally:
        os.close(fd)


def _kind(mode: int) -> str:
    if stat.S_ISDIR(mode):
        return "a directory"
    if stat.S_ISFIFO(mode):
        return "a FIFO"
    if stat.S_ISSOCK(mode):
        return "a socket"
    if stat.S_ISCHR(mode) or stat.S_ISBLK(mode):
        return "a device"
    return "not a file"


def read_text(root: Path, path: Path, cap: int) -> tuple[str, bool]:
    """At most cap bytes of a text file, and whether it was cut. Binary is Refused."""

    with open_regular(root, path) as fd:
        size = os.fstat(fd).st_size
        chunks: list[bytes] = []
        total = 0
        while total < cap:
            chunk = os.read(fd, min(65536, cap - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        cut = total >= cap and bool(os.read(fd, 1))
    data = b"".join(chunks)
    text = decode(data, final=not cut)
    if text is None:
        raise Refused(f"That file is binary ({size} bytes). It was not read.")
    return text, cut


def decode(data: bytes, final: bool = True) -> str | None:
    """UTF-8 text, or None for binary: a NUL in the first 8 KiB, or bytes that are not UTF-8.

    When final is False the data was cut, so a character split at the end is dropped.
    """

    if b"\x00" in data[:_SNIFF]:
        return None
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    try:
        return decoder.decode(data, final=final)
    except UnicodeDecodeError:
        return None


def write_text(root: Path, path: Path, text: str, *, overwrite: bool, create: bool) -> None:
    """Write text to path through a new file renamed over the name."""

    parts = _parts(root, path)
    with _directory(root, parts[:-1], create=create) as parent:
        name = parts[-1]
        try:
            found = os.stat(name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            found = None
        if found is not None:
            if stat.S_ISLNK(found.st_mode):
                raise Refused("That path is a link. A write does not go through one.")
            if stat.S_ISDIR(found.st_mode):
                raise Refused("That path is a directory.")
            if not stat.S_ISREG(found.st_mode):
                raise Refused(f"That path is not a regular file ({_kind(found.st_mode)}).")
            if not overwrite:
                raise Refused("That file exists. Edit it, or set overwrite.")
        temporary = f".{name}.{os.getpid()}.{os.urandom(4).hex()}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(temporary, flags, 0o666, dir_fd=parent)
        try:
            try:
                if found is not None:
                    os.fchmod(fd, stat.S_IMODE(found.st_mode))
                data = text.encode("utf-8")
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view) :]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.rename(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
        except BaseException:
            try:
                os.unlink(temporary, dir_fd=parent)
            except OSError:
                pass
            raise
