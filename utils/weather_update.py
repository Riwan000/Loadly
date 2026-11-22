"""
Incremental weather refresh for the daily update job.

Fills ``data/zenodo_df.csv`` and the per-state files in ``data/14983362/`` (the files
inference reads) for every date up to ``end`` that has no row yet, using Open-Meteo.
Unit conversion (Kelvin, J/m^2, wind vectors, 0-1 cloud cover) is delegated to the
existing ``fetch_recent_weather.process_hourly_to_daily`` so new rows are identical in
meaning to the history (verified against existing rows for overlapping dates).

Source rule
  * The archive API serves complete 24-hour days even for yesterday, but for recent days
    those values are forecast-model fill, not final ERA5 (ERA5 lags ~5 days). Therefore
    EVERY date within WEATHER_PROVISIONAL_DAYS of the run date is PROVISIONAL, whichever
    API it came from. Provisional (date, state) pairs are tracked in the sidecar
    ``data/weather_provisional.json`` ({date: [states]}).
  * The forecast API with ``past_days`` is used only for pairs the archive cannot serve
    completely; those rows are provisional regardless of age.
  * Every run re-fetches the archive for provisional dates AND for every stored date
    inside the provisional window (so rows written before this rule existed are adopted
    automatically). A stored row is replaced only when the archive values differ; once a
    date is older than the window and the archive serves it completely, its row is
    refreshed one last time and the provisional marker is dropped (FINAL).
  * a day is accepted only with 24 hourly values for every variable; a day with gaps
    is never half-filled and never interpolated.
"""

import csv
import datetime as dt
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests

from config import ALL_STATES, STATE_COORDINATES
from utils.csv_upsert import frame_to_rows, read_header, scan_keys, upsert_csv
from utils.fetch_psp_reports import UpstreamError, request_with_retry
from utils.fetch_recent_weather import STATE_FILE_MAP, process_hourly_to_daily

logger = logging.getLogger("loadly.daily_update")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = [
    "temperature_2m", "dewpoint_2m", "apparent_temperature", "windspeed_10m",
    "winddirection_10m", "shortwave_radiation", "cloudcover",
]
TIMEZONE = "Asia/Kolkata"
LOCATIONS_PER_REQUEST = 12     # Open-Meteo accepts comma-separated coordinate lists
MAX_DAYS_PER_REQUEST = 31
MAX_PAST_DAYS = 92             # forecast API limit
HOURS_PER_DAY = 24
REQUEST_DELAY_S = 1.0
# ERA5 final data lags ~5 days; anything this recent (in days before the run date) may
# still change and is kept provisional / re-fetched.
WEATHER_PROVISIONAL_DAYS = 7
# Values are recomputed by the same pipeline, so identical inputs give identical floats;
# the tolerance only absorbs float noise when deciding whether a refresh changed a row.
REFRESH_ABS_TOLERANCE = 1e-9

# Sanitised error codes (detail goes to the log only).
ERR_UPSTREAM = "upstream_unavailable"
ERR_FORECAST_WINDOW = "forecast_window_exceeded"

SOURCE_ARCHIVE = "archive"
SOURCE_FORECAST = "forecast"

PairKey = Tuple[str, str]  # (date, state)


def is_settled(day: dt.date, today: dt.date) -> bool:
    """True when ``day`` is older than the provisional window (archive values are final)."""
    return (today - day).days > WEATHER_PROVISIONAL_DAYS


@dataclass(frozen=True)
class WeatherPaths:
    zenodo: Path
    state_dir: Path
    provisional: Path


def load_provisional(path: Path) -> Dict[str, Set[str]]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    entries = raw.get("provisional") if isinstance(raw, dict) else None
    if not isinstance(entries, dict):
        return {}
    # Malformed entries (non-date keys, non-list values) are dropped, never fatal.
    out: Dict[str, Set[str]] = {}
    for d, states in entries.items():
        if not isinstance(d, str) or not isinstance(states, list):
            continue
        try:
            dt.date.fromisoformat(d)
        except ValueError:
            continue
        out[d] = {s for s in states if isinstance(s, str)}
    return out


def save_provisional(path: Path, pairs: Dict[str, Set[str]]) -> None:
    payload = {"provisional": {d: sorted(s) for d, s in sorted(pairs.items()) if s}}
    tmp = Path(path).with_name(Path(path).name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def scan_coverage(zenodo: Path) -> Dict[str, Set[str]]:
    """date -> set of states that have a row in zenodo_df.csv."""
    coverage: Dict[str, Set[str]] = {}
    for date, state in scan_keys(zenodo, ["date", "state"]):
        coverage.setdefault(date, set()).add(state)
    return coverage


def _contiguous_chunks(days: List[dt.date]) -> List[Tuple[dt.date, dt.date]]:
    """Split dates into contiguous (start, end) ranges of at most MAX_DAYS_PER_REQUEST days."""
    chunks: List[Tuple[dt.date, dt.date]] = []
    for day in sorted(days):
        if chunks and (day - chunks[-1][1]).days == 1 and (day - chunks[-1][0]).days < MAX_DAYS_PER_REQUEST:
            chunks[-1] = (chunks[-1][0], day)
        else:
            chunks.append((day, day))
    return chunks


def _fetch_locations(session, url: str, params: dict, states: List[str]) -> Dict[str, dict]:
    """One request for several states; returns {state: hourly dict}."""
    params = dict(params)
    params["latitude"] = ",".join(str(STATE_COORDINATES[s]["lat"]) for s in states)
    params["longitude"] = ",".join(str(STATE_COORDINATES[s]["lon"]) for s in states)
    params["hourly"] = ",".join(HOURLY_VARS)
    params["timezone"] = TIMEZONE
    try:
        payload = request_with_retry(session, "GET", url, params=params).json()
    except ValueError as exc:
        raise UpstreamError(f"Open-Meteo returned invalid JSON: {exc}") from exc
    results = payload if isinstance(payload, list) else [payload]
    if len(results) != len(states) or not all(isinstance(r, dict) for r in results):
        raise UpstreamError(f"Open-Meteo returned {len(results)} locations for {len(states)} requested")
    return {s: r.get("hourly") or {} for s, r in zip(states, results)}


def _complete_daily_rows(hourly: dict, wanted: Set[str]) -> pd.DataFrame:
    """Daily ERA5-style rows for wanted dates that have all 24 hours of every variable."""
    if not hourly or "time" not in hourly:
        return pd.DataFrame()
    df = pd.DataFrame(hourly)
    df["date"] = df["time"].astype(str).str[:10]
    df = df[df["date"].isin(wanted)]
    keep: Set[str] = set()
    for day, grp in df.groupby("date"):
        if len(grp) == HOURS_PER_DAY and not grp[HOURLY_VARS].isna().any().any():
            keep.add(day)
    df = df[df["date"].isin(keep)].drop(columns="date")
    return process_hourly_to_daily(df.to_dict("list")) if not df.empty else pd.DataFrame()


def _fetch_source(
    session, source: str, states: List[str], days: List[dt.date], today: dt.date
) -> Tuple[Dict[PairKey, pd.Series], List[str]]:
    """Fetch complete daily rows for (date, state) pairs. Returns ({(date, state): row}, errors)."""
    rows: Dict[PairKey, pd.Series] = {}
    errors: List[dict] = []
    if source == SOURCE_ARCHIVE:
        plan = [(ARCHIVE_URL, {"start_date": a.isoformat(), "end_date": b.isoformat()})
                for a, b in _contiguous_chunks(days)]
    else:
        past = (today - min(days)).days + 1
        if past > MAX_PAST_DAYS:
            logger.error("Forecast API cannot serve dates older than %d days (oldest %s)", MAX_PAST_DAYS, min(days))
            errors.append({"code": ERR_FORECAST_WINDOW})
            return rows, errors
        plan = [(FORECAST_URL, {"past_days": past, "forecast_days": 1})]

    wanted = {d.isoformat() for d in days}
    for url, params in plan:
        for i in range(0, len(states), LOCATIONS_PER_REQUEST):
            batch = states[i:i + LOCATIONS_PER_REQUEST]
            try:
                hourly_by_state = _fetch_locations(session, url, params, batch)
            except RuntimeError as exc:
                logger.error("Open-Meteo %s request %s failed: %s", source, params, exc)
                if {"code": ERR_UPSTREAM} not in errors:
                    errors.append({"code": ERR_UPSTREAM})
                continue
            finally:
                time.sleep(REQUEST_DELAY_S)
            for state, hourly in hourly_by_state.items():
                daily = _complete_daily_rows(hourly, wanted)
                for _, rec in daily.iterrows():
                    rows[(rec["date"], state)] = rec
    return rows, errors


def needed_dates(
    coverage: Dict[str, Set[str]], provisional: Dict[str, Set[str]], end: dt.date,
    backfill_from: Optional[dt.date] = None, today: Optional[dt.date] = None,
) -> List[dt.date]:
    """
    Dates <= end after the last fully covered date that lack a state row, plus provisional
    dates, plus (when ``today`` is given) every stored date inside the provisional window.
    """
    all_states = set(ALL_STATES)
    complete = [d for d, s in coverage.items() if s >= all_states]
    start = dt.date.fromisoformat(max(complete)) + dt.timedelta(days=1) if complete else None
    if backfill_from is not None:
        start = backfill_from if start is None else min(start, backfill_from)
    needed: Set[dt.date] = set()
    if start is not None:
        day = start
        while day <= end:
            if coverage.get(day.isoformat(), set()) < all_states:
                needed.add(day)
            day += dt.timedelta(days=1)
    needed |= {dt.date.fromisoformat(d) for d, s in provisional.items() if s and d <= end.isoformat()}
    if today is not None:
        needed |= {dt.date.fromisoformat(d) for d in coverage
                   if d <= end.isoformat() and not is_settled(dt.date.fromisoformat(d), today)}
    return sorted(needed)


def _stored_rows(zenodo: Path, dates: Set[str]) -> Dict[PairKey, List[str]]:
    """Exact stored text of the zenodo rows for ``dates`` (keyed by (date, state))."""
    out: Dict[PairKey, List[str]] = {}
    if not dates:
        return out
    with Path(zenodo).open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        di, si = header.index("date"), header.index("state")
        for row in reader:
            if row and row[di] in dates:
                out[(row[di], row[si])] = row
    return out


def _values_differ(stored: Optional[List[str]], fresh: List[str]) -> bool:
    if stored is None or len(stored) != len(fresh):
        return True
    for old, new in zip(stored, fresh):
        if old == new:
            continue
        try:
            a, b = float(old), float(new)
        except ValueError:
            return True
        if math.isnan(a) != math.isnan(b) or (not math.isnan(a) and abs(a - b) > REFRESH_ABS_TOLERANCE):
            return True
    return False


def _plan_changes(needed, states, coverage, provisional, archive_rows, forecast_rows,
                  stored, header, today):
    """
    Decide per (date, state): add an archive row, refresh a provisional row with newer
    archive values, or add a forecast (provisional) row. Returns
    (new_pairs, replace_keys, remaining provisional markers, finalized dates).
    """
    new_pairs: List[Tuple[str, str, pd.Series, str]] = []
    replace_keys: Set[PairKey] = set()
    remaining: Dict[str, Set[str]] = {d: set(s) for d, s in provisional.items()}
    finalized: Set[str] = set()
    for day in needed:
        d, settled = day.isoformat(), is_settled(day, today)
        for s in states:
            present = s in coverage.get(d, set())
            marked = s in provisional.get(d, set())
            arch = archive_rows.get((d, s))
            if present:
                if arch is None or (settled and not marked):
                    continue            # nothing newer to apply / already final
                fresh = frame_to_rows(pd.DataFrame([{**arch.to_dict(), "state": s}]), header)[0]
                if _values_differ(stored.get((d, s)), fresh):
                    new_pairs.append((d, s, arch, SOURCE_ARCHIVE))
                    replace_keys.add((d, s))
                if settled:
                    remaining.get(d, set()).discard(s)
                    finalized.add(d)
                else:
                    remaining.setdefault(d, set()).add(s)
            elif arch is not None:
                new_pairs.append((d, s, arch, SOURCE_ARCHIVE))
                if not settled:
                    remaining.setdefault(d, set()).add(s)
            elif (d, s) in forecast_rows:
                new_pairs.append((d, s, forecast_rows[(d, s)], SOURCE_FORECAST))
                remaining.setdefault(d, set()).add(s)
    return new_pairs, replace_keys, remaining, finalized


def _write_rows(paths: WeatherPaths, new_pairs, replace_keys: Set[PairKey]) -> None:
    if not new_pairs:
        return
    frame = pd.DataFrame([{**row.to_dict(), "state": state} for _d, state, row, _src in new_pairs])
    for state, grp in frame.groupby("state"):
        state_file = paths.state_dir / STATE_FILE_MAP[state]
        if not state_file.exists():
            logger.warning("Per-state weather file missing, not created: %s", state_file)
            continue
        keys = {(d,) for d, s in replace_keys if s == state}
        upsert_csv(state_file, grp.drop(columns="state"), ["date"], keys)
    # Aggregate file last: its coverage drives what the next run considers missing.
    upsert_csv(paths.zenodo, frame, ["date", "state"], replace_keys)


def update_weather(
    paths: WeatherPaths, end: dt.date, session: requests.Session, today: dt.date,
    backfill_from: Optional[dt.date] = None, dry_run: bool = False,
) -> dict:
    """Bring weather up to ``end``. Returns the summary dict described by ``summary`` below."""
    summary = {
        "archive": [], "provisional": [], "upgraded_to_archive": [], "finalized": [], "still_missing": [],
        "provisional_pending": [], "last_weather_date": None, "errors": [],
    }
    if not paths.zenodo.exists():
        raise FileNotFoundError(f"{paths.zenodo} not found (is the data/ volume mounted?)")
    coverage = scan_coverage(paths.zenodo)
    provisional = load_provisional(paths.provisional)
    summary["last_weather_date"] = max(coverage) if coverage else None
    summary["provisional_pending"] = sorted(d for d, s in provisional.items() if s)
    needed = needed_dates(coverage, provisional, end, backfill_from, today)
    if not needed:
        return summary
    if dry_run:
        # Only dates lacking rows; complete provisional-window dates would merely be re-checked.
        summary["still_missing"] = [d.isoformat() for d in needed
                                    if coverage.get(d.isoformat(), set()) < set(ALL_STATES)]
        return summary

    states = list(ALL_STATES)
    archive_rows, errs = _fetch_source(session, SOURCE_ARCHIVE, states, needed, today)
    summary["errors"].extend(errs)

    # Forecast API fills only pairs that are absent from the file and not served by the archive.
    gap_days = sorted({
        day for day in needed for s in states
        if s not in coverage.get(day.isoformat(), set()) and (day.isoformat(), s) not in archive_rows
    })
    forecast_rows: Dict[PairKey, pd.Series] = {}
    if gap_days:
        forecast_rows, errs = _fetch_source(session, SOURCE_FORECAST, states, gap_days, today)
        summary["errors"].extend(errs)

    stored = _stored_rows(paths.zenodo, {d.isoformat() for d in needed if d.isoformat() in coverage})
    new_pairs, replace_keys, remaining, finalized = _plan_changes(
        needed, states, coverage, provisional, archive_rows, forecast_rows,
        stored, read_header(paths.zenodo), today)

    # Mark provisional pairs BEFORE writing rows so a crash can never leave an unmarked
    # forecast-derived row; drop markers of upgraded pairs only after the rows are in.
    save_provisional(paths.provisional, {
        d: remaining.get(d, set()) | provisional.get(d, set()) for d in set(remaining) | set(provisional)})
    _write_rows(paths, new_pairs, replace_keys)
    save_provisional(paths.provisional, remaining)

    _summarise(summary, new_pairs, replace_keys, needed, coverage, remaining)
    summary["finalized"] = sorted(d for d in finalized if not remaining.get(d))
    return summary


def _summarise(summary, new_pairs, replace_keys, needed, coverage, remaining) -> None:
    summary["archive"] = sorted({d for d, s, _r, src in new_pairs if src == SOURCE_ARCHIVE and (d, s) not in replace_keys})
    # Stored rows whose values changed and were replaced by newer archive values.
    summary["upgraded_to_archive"] = sorted({d for d, _s in replace_keys})
    summary["provisional"] = sorted({d for d, _s, _r, src in new_pairs if src == SOURCE_FORECAST})
    summary["provisional_pending"] = sorted(d for d, s in remaining.items() if s)
    added = {(d, s) for d, s, _r, _src in new_pairs if (d, s) not in replace_keys}
    for day in needed:
        d = day.isoformat()
        have = len(coverage.get(d, set())) + sum(1 for (dd, _s) in added if dd == d)
        if have < len(ALL_STATES):
            summary["still_missing"].append(d)
    dates = set(coverage) | {d for d, _s in added}
    summary["last_weather_date"] = max(dates) if dates else None
