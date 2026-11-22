"""
Non-blocking cross-process lock file (O_CREAT|O_EXCL) with dead-holder and stale-lock reclaim.

The lock file holds ``"<pid> <hostname> <token> <iso time>"``. A contender reclaims an
existing lock when

  * its holder ran on THIS host and that PID is no longer alive (killed / crashed run,
    server restarted while a daemon-thread run was in progress) -> reclaimed immediately;
  * or the file is older than ``stale_seconds`` (fallback for other hosts sharing the
    volume, unreadable/legacy lock files, and PID reuse).

Reclaims are serialised by a short-lived ``<lock>.reclaim`` guard file so two contenders
cannot both remove the same dead lock and then each create their own. ``release()`` only
deletes the lock file if it still carries this holder's token, so a holder whose lock was
reclaimed by age never deletes the new holder's lock.

PID liveness uses ``OpenProcess``/``GetExitCodeProcess`` on Windows (``os.kill(pid, 0)``
would TERMINATE the process there) and ``os.kill(pid, 0)`` on POSIX. Any doubt (access
denied, unreadable file, foreign host) counts as "alive", so a live holder is never
displaced by the liveness path. The ``stale_seconds`` age rule applies only to holders
whose liveness can't be proven (foreign host, unreadable file); a provably live local
holder is only reclaimed after LIVE_HOLDER_CEILING_SECONDS, to cover PID reuse.
"""

import ctypes
import logging
import os
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

RECLAIM_GUARD_SUFFIX = ".reclaim"
RECLAIM_GUARD_STALE_SECONDS = 60.0     # a guard left by a process killed mid-reclaim
LIVE_HOLDER_CEILING_SECONDS = 24 * 3600  # live local PID: only reclaimed past this (PID reuse)
MAX_WINDOWS_PID = 2 ** 32 - 1

# Win32 constants
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_INVALID_PARAMETER = 87


@dataclass(frozen=True)
class LockInfo:
    """Parsed lock file. ``host``/``token`` are None for legacy ``"<pid> <time>"`` files."""
    raw: str
    pid: Optional[int]
    host: Optional[str]
    token: Optional[str]


def current_host() -> str:
    """Hostname as stored in lock files (whitespace-free)."""
    return "".join((socket.gethostname() or "unknown-host").split())


def _win_pid_alive(pid: int) -> bool:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # No such process -> ERROR_INVALID_PARAMETER. Anything else (e.g. access denied
        # for another user's process) means it exists.
        return ctypes.get_last_error() != _ERROR_INVALID_PARAMETER
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def pid_alive(pid: int) -> bool:
    """True when a process with this PID exists on this host (errs on the side of True)."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        if pid > MAX_WINDOWS_PID:
            return False
        try:
            return _win_pid_alive(pid)
        except (OSError, AttributeError):
            return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # PermissionError: exists but owned by someone else
        return True
    return True


def read_lock_info(path: Path) -> Optional[LockInfo]:
    """Parsed lock file, or None when there is no lock file."""
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    except OSError:
        return LockInfo(raw="", pid=None, host=None, token=None)
    parts = raw.split()
    pid = int(parts[0]) if parts and parts[0].isdigit() else None
    host, token = (parts[1], parts[2]) if len(parts) >= 4 else (None, None)
    return LockInfo(raw=raw, pid=pid, host=host, token=token)


class FileLock:
    """
    Non-blocking lock. ``acquire()`` returns False when another live holder (thread or
    process) has it. See the module docstring for the reclaim rules.
    """

    def __init__(self, path: Path, stale_seconds: float):
        self.path = Path(path)
        self.stale_seconds = stale_seconds
        self._held = False
        self._token: Optional[str] = None
        self._guard = threading.Lock()

    # -- inspection ---------------------------------------------------------
    def _age_exceeded(self, limit_seconds: Optional[float] = None) -> bool:
        limit = self.stale_seconds if limit_seconds is None else limit_seconds
        try:
            return (time.time() - self.path.stat().st_mtime) > limit
        except FileNotFoundError:
            return False  # released meanwhile; the next create attempt succeeds

    @staticmethod
    def _holder_dead(info: LockInfo) -> bool:
        return (info.pid is not None and info.host is not None
                and info.host == current_host() and not pid_alive(info.pid))

    @staticmethod
    def _holder_alive_here(info: LockInfo) -> bool:
        return (info.pid is not None and info.host is not None
                and info.host == current_host() and pid_alive(info.pid))

    def _reclaim_reason(self, info: LockInfo) -> Optional[str]:
        if self._holder_dead(info):
            return f"holder process {info.pid} is no longer running"
        if self._holder_alive_here(info):
            # A provably live local holder (e.g. a retrain running > stale_seconds) is never
            # displaced by the normal age rule; only the much longer ceiling, which covers
            # PID reuse by an unrelated process, can reclaim it.
            if self._age_exceeded(LIVE_HOLDER_CEILING_SECONDS):
                return f"live-PID lock older than {LIVE_HOLDER_CEILING_SECONDS}s (assumed PID reuse)"
            return None
        if self._age_exceeded():
            return f"lock older than {int(self.stale_seconds)}s"
        return None

    def is_locked(self) -> bool:
        """True when a live (non-reclaimable) holder has the lock. Read-only."""
        info = read_lock_info(self.path)
        return info is not None and self._reclaim_reason(info) is None

    # -- acquire / release --------------------------------------------------
    def _try_create(self) -> bool:
        token = uuid.uuid4().hex
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"{os.getpid()} {current_host()} {token} {datetime.now().isoformat()}\n")
        self._token = token
        return True

    def _enter_reclaim_guard(self) -> Optional[Path]:
        guard = self.path.with_name(self.path.name + RECLAIM_GUARD_SUFFIX)
        try:
            os.close(os.open(str(guard), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return guard
        except FileExistsError:
            try:
                if time.time() - guard.stat().st_mtime > RECLAIM_GUARD_STALE_SECONDS:
                    guard.unlink(missing_ok=True)  # left by a process killed mid-reclaim
            except FileNotFoundError:
                pass
            return None

    def _remove_if_reclaimable(self) -> Tuple[bool, Optional[str]]:
        """Remove a dead/stale lock file. Returns (lock file now absent, reason)."""
        guard = self._enter_reclaim_guard()
        if guard is None:
            return False, None
        try:
            info = read_lock_info(self.path)
            if info is None:
                return True, None
            reason = self._reclaim_reason(info)
            if reason is None:
                return False, None
            graveyard = self.path.with_name(f"{self.path.name}.stale.{os.getpid()}.{threading.get_ident()}")
            try:
                os.replace(self.path, graveyard)
            except FileNotFoundError:
                return True, None
            moved = read_lock_info(graveyard)
            if moved is not None and moved.raw != info.raw and not self.path.exists():
                # The holder released and someone re-acquired between our read and the
                # rename: put the fresh lock back instead of stealing it.
                os.replace(graveyard, self.path)
                return False, None
            graveyard.unlink(missing_ok=True)
            return True, reason
        finally:
            guard.unlink(missing_ok=True)

    def acquire(self) -> bool:
        with self._guard:
            if self._held:
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self._try_create():
                self._held = True
                return True
            absent, reason = self._remove_if_reclaimable()
            if absent and self._try_create():
                if reason:
                    logger.warning("Reclaimed lock file %s (%s)", self.path, reason)
                self._held = True
                return True
            return False

    def release(self) -> None:
        with self._guard:
            if not self._held:
                return
            self._held = False
            info = read_lock_info(self.path)
            if info is not None and not info.raw:
                # Unreadable: ownership can't be proven, so leave it for the reclaim rules.
                logger.warning("Lock file %s could not be read on release; not removing it", self.path)
                return
            if info is not None and self._token and info.token != self._token:
                logger.warning("Lock file %s now belongs to another holder; not removing it", self.path)
                return
            self.path.unlink(missing_ok=True)
