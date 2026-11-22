"""
Opt-in in-process scheduler for the daily data update (no extra dependencies).

Enable with ``LOADLY_AUTO_UPDATE=1``; ``LOADLY_UPDATE_TIME`` (HH:MM, Asia/Kolkata, default
13:00) sets the daily run time and ``LOADLY_UPDATE_STARTUP_DELAY_S`` (default 30) the delay
of the catch-up run after startup. A daemon thread does all the work, so startup and
request handling are never blocked; exceptions are logged and never propagate. With several
uvicorn workers every worker runs a scheduler, but the update's file lock lets only one
actually execute; the others merely refresh their in-process caches when the files change.
"""

import datetime as dt
import logging
import os
import threading
from typing import Callable, Optional, Tuple

logger = logging.getLogger("loadly.daily_update")

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
ENABLE_ENV = "LOADLY_AUTO_UPDATE"
TIME_ENV = "LOADLY_UPDATE_TIME"
STARTUP_DELAY_ENV = "LOADLY_UPDATE_STARTUP_DELAY_S"
DEFAULT_TIME = "13:00"
DEFAULT_STARTUP_DELAY_S = 30.0
CACHE_CHECK_INTERVAL_S = 300.0
JOIN_TIMEOUT_S = 5.0


def parse_update_time(value: str) -> Tuple[int, int]:
    """'13:00' -> (13, 0). Raises ValueError for anything else."""
    hour_s, minute_s = value.strip().split(":")
    hour, minute = int(hour_s), int(minute_s)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"out-of-range time: {value!r}")
    return hour, minute


def scheduler_settings() -> Tuple[bool, str]:
    """(enabled, 'HH:MM') from the environment; an invalid time falls back to the default."""
    enabled = os.getenv(ENABLE_ENV, "").strip().lower() in ("1", "true", "yes")
    raw = os.getenv(TIME_ENV, DEFAULT_TIME)
    try:
        parse_update_time(raw)
        return enabled, raw.strip()
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", TIME_ENV, raw, DEFAULT_TIME)
        return enabled, DEFAULT_TIME


def next_scheduled_run(now: Optional[dt.datetime] = None, time_str: str = DEFAULT_TIME) -> dt.datetime:
    """Next HH:MM IST strictly after ``now`` (today if still ahead, otherwise tomorrow)."""
    hour, minute = parse_update_time(time_str)
    now_ist = (now or dt.datetime.now(IST)).astimezone(IST)
    candidate = now_ist.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return candidate if candidate > now_ist else candidate + dt.timedelta(days=1)


class DailyUpdateScheduler:
    """Daemon thread: one catch-up run after startup, then one run per day at ``time_str`` IST."""

    def __init__(
        self,
        run_fn: Callable[[], object],
        time_str: str = DEFAULT_TIME,
        startup_delay_s: float = DEFAULT_STARTUP_DELAY_S,
        cache_refresh_fn: Optional[Callable[[], object]] = None,
        clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(IST),
    ):
        self._run_fn = run_fn
        self._time_str = time_str
        self._startup_delay_s = startup_delay_s
        self._cache_refresh_fn = cache_refresh_fn
        self._clock = clock
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="loadly-daily-update", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Signal the loop and wait up to JOIN_TIMEOUT_S. A run still in progress is not
        interrupted; if the process then exits, the daemon thread dies without releasing
        data/daily_update.lock, and the next process reclaims it at once because the
        recorded PID is no longer alive (utils/file_lock.py).
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=JOIN_TIMEOUT_S)

    def _safe(self, fn: Callable[[], object], label: str) -> None:
        try:
            fn()
        except Exception:  # a failing job must never take the server down
            logger.exception("%s failed", label)

    def _loop(self) -> None:
        if self._stop.wait(self._startup_delay_s):
            return
        self._safe(self._run_fn, "Startup catch-up update")
        while not self._stop.is_set():
            due = next_scheduled_run(self._clock(), self._time_str)
            while not self._stop.is_set():
                remaining = (due - self._clock()).total_seconds()
                if remaining <= 0:
                    break
                if self._stop.wait(min(remaining, CACHE_CHECK_INTERVAL_S)) or self._cache_refresh_fn is None:
                    continue
                self._safe(self._cache_refresh_fn, "Cache refresh")
            if self._stop.is_set():
                return
            self._safe(self._run_fn, "Scheduled daily update")
            self._safe(self._cache_refresh_fn or (lambda: None), "Cache refresh")
