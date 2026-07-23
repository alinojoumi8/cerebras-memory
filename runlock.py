"""Small cross-process lock used by manual and scheduled ingestion."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator


class IngestionAlreadyRunning(RuntimeError):
    pass


class DistillationAlreadyRunning(RuntimeError):
    pass


def _lock(handle: BinaryIO) -> None:
    handle.seek(0)
    if handle.read(1) == b"":
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except ImportError:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle: BinaryIO) -> None:
    handle.seek(0)
    try:
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    except ImportError:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def ingestion_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        try:
            _lock(handle)
        except (OSError, BlockingIOError) as exc:
            raise IngestionAlreadyRunning("Another Cerebras Memory refresh is already running") from exc
        try:
            yield
        finally:
            _unlock(handle)
    finally:
        handle.close()


@contextmanager
def distillation_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        try:
            _lock(handle)
        except (OSError, BlockingIOError) as exc:
            raise DistillationAlreadyRunning(
                "Another Cerebras Memory distillation is already running"
            ) from exc
        try:
            yield
        finally:
            _unlock(handle)
    finally:
        handle.close()
