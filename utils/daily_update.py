"""
Idempotent daily data refresh: Grid-India demand + Open-Meteo weather + ledger reconcile.

Usage:
    python -m utils.daily_update                       # up to yesterday (IST), 14-day catch-up
    python -m utils.daily_update --end 2026-11-20 --catchup 30
    python -m utils.daily_update --weather-backfill-from 2026-08-30
    python -m utils.daily_update --dry-run             # report only, write nothing (see below)

What one run does
  1. Demand: dates in [end - catchup + 1, end] missing from data/daily_energy_met_MU.csv are
     downloaded from Grid-India (data/psp/), parsed and APPENDED to the five demand files
     (schemas and column order preserved). Dates with no usable report stay missing and are
     reported with a reason; nothing is ever interpolated.
     REVISION RULE: existing rows are never touched, except that a newer revision (higher
     upload Id) of a stored date's spreadsheet report is downloaded and, if it passes the
     same plausibility checks as a first parse and its values differ from the stored row,
     the row is replaced. The comparison is always stored row vs. NEWEST LOCAL report, so
     an interrupted revision is completed by the next run. Only the last --catchup days
     and only dates on/after 2024-04-29 (first PSP-sourced date) are considered.
  2. Weather: dates after the last fully covered date (or from --weather-backfill-from)
     are added to data/zenodo_df.csv and data/14983362/<State>.csv. See weather_update.py
     for the archive-vs-provisional rule.
  3. The prediction ledger is reconciled for added/revised demand dates.
  4. In-process data caches are cleared so a running server serves the new data.

Training data (data/processed/feature_engineered_dataset.csv) is NOT rebuilt here. It derives from
preprocessed_merged_features_clean.csv (notebooks/01_data_preprocessing.ipynb); to refresh it, rerun that notebook
and then `python utils/train_pipeline.py` (or POST /api/mlops/retrain).
Runs are serialised by data/daily_update.lock (utils/file_lock.py: a lock left by a dead
process on this host is reclaimed at once, any other after LOCK_STALE_SECONDS). A run that
finds the lock held is recorded in data/daily_update_skipped.json and shown by the status API.

Errors: the persisted/returned summary holds only sanitised entries
{"stage", "code", "message"[, "date"]} with codes from ERROR_MESSAGES; exception text,
paths and URLs go to logs/daily_update.log only.

Dry run (--dry-run): writes nothing to data/ and nothing to logs/ (log lines go to stderr
via the CLI's console handler only) and takes no lock. It still QUERIES the Grid-India
listing API (read-only POSTs, to report what would be downloaded); it downloads no report
and makes no Open-Meteo request.
"""

import argparse
import datetime as dt
import json
import logging
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.csv_upsert import read_header, scan_keys, upsert_csv  # noqa: E402
from utils.fetch_psp_reports import (  # noqa: E402
    PDF_EXT, SPREADSHEET_EXTS, UpstreamError, already_present, collect_items, download_best,
    item_ext, make_session, rank_candidates,
)
from utils.file_lock import FileLock  # noqa: E402
from utils.parse_psp_reports import parse_psp_xls, select_files_per_date  # noqa: E402
from utils.replace_fabricated_demand import GEN_REGION_COLS, validate_parsed  # noqa: E402
from utils.weather_update import WeatherPaths, load_provisional, update_weather  # noqa: E402

logger = logging.getLogger("loadly.daily_update")

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
FIRST_REVISABLE_DATE = "2024-04-29"
VALUE_TOLERANCE = 0.005
DEFAULT_CATCHUP_DAYS = 14
LOCK_STALE_SECONDS = 2 * 3600
LOG_MAX_BYTES = 1_000_000
LOG_BACKUPS = 3
PSP_ID_RE = re.compile(r"_PSP_.*?(\d+)(?:\.\w+)?$")

REASON_NOT_PUBLISHED = "not published yet"
REASON_NOT_LISTED = "no report listed on Grid-India"
REASON_DOWNLOAD_FAILED = "download failed"
REASON_DOWNLOAD_404 = "download failed: file listed but not found (404)"
REASON_IMPLAUSIBLE = "report parsed but incomplete or implausible"
REASON_UNPARSEABLE = "report could not be parsed"
REASON_NO_USABLE = "no usable report"
REASON_WOULD_DOWNLOAD = "would be downloaded"
MISSING_REASONS = frozenset({
    REASON_NOT_PUBLISHED, REASON_NOT_LISTED, REASON_DOWNLOAD_FAILED, REASON_DOWNLOAD_404,
    REASON_IMPLAUSIBLE, REASON_UNPARSEABLE, REASON_NO_USABLE, REASON_WOULD_DOWNLOAD,
})
QUARANTINE_SUFFIX = ".bad"

# Sanitised error vocabulary: the only error content that is persisted or served.
STAGES = frozenset({"demand", "weather", "ledger", "job"})
ERROR_MESSAGES = {
    "upstream_unavailable": "An upstream data source could not be reached or answered unusably.",
    "data_missing": "A required data file is missing.",
    "revision_download_failed": "A newer report revision could not be downloaded.",
    "forecast_window_exceeded": "Some weather dates are too old for the forecast API.",
    "ledger_reconcile_failed": "Prediction ledger reconciliation failed.",
    "internal_error": "Unexpected internal error; see the server log.",
}
RUN_STATUSES = frozenset({"ok", "error", "skipped"})
SKIP_REASON_LOCK_HELD = "lock_held"
ISO_DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
ISO_DATETIME_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:[+-][0-9]{2}:[0-9]{2})?")
WEATHER_ADDED_KEYS = ("archive", "provisional", "upgraded_to_archive", "finalized")


@dataclass(frozen=True)
class DataPaths:
    """Every file the job reads or writes (injectable so tests never touch data/)."""
    psp_dir: Path
    demand_main: Path
    demand_mirror: Path
    state_mu_1: Path
    state_mu_2: Path
    long_state_mu2: Path
    zenodo: Path
    state_weather_dir: Path
    provisional: Path
    status_file: Path
    lock_file: Path
    log_file: Path
    skip_file: Path

    @classmethod
    def under(cls, root: Path) -> "DataPaths":
        data = Path(root) / "data"
        return cls(
            psp_dir=data / "psp",
            demand_main=data / "daily_energy_met_MU.csv",
            demand_mirror=data / "14983362" / "daily_energy_met_MU.csv",
            state_mu_1=data / "State_mu" / "state_mu_1.csv",
            state_mu_2=data / "State_mu" / "state_mu_2.csv",
            long_state_mu2=data / "processed" / "long_state_mu2.csv",
            zenodo=data / "zenodo_df.csv",
            state_weather_dir=data / "14983362",
            provisional=data / "weather_provisional.json",
            status_file=data / "daily_update_status.json",
            lock_file=data / "daily_update.lock",
            log_file=Path(root) / "logs" / "daily_update.log",
            skip_file=data / "daily_update_skipped.json",
        )

    @property
    def weather(self) -> WeatherPaths:
        return WeatherPaths(self.zenodo, self.state_weather_dir, self.provisional)


def default_paths() -> DataPaths:
    return DataPaths.under(ROOT_DIR)


def today_ist(now: Optional[dt.datetime] = None) -> dt.date:
    return (now or dt.datetime.now(IST)).astimezone(IST).date()


def make_error(stage: str, code: str, date: Optional[str] = None) -> dict:
    """A sanitised error entry: fixed stage/code/message only (plus an ISO date)."""
    stage = stage if stage in STAGES else "job"
    code = code if code in ERROR_MESSAGES else "internal_error"
    entry = {"stage": stage, "code": code, "message": ERROR_MESSAGES[code]}
    if isinstance(date, str) and ISO_DATE_RE.fullmatch(date):
        entry["date"] = date
    return entry


def classify_exception(exc: BaseException) -> str:
    """Error code for an exception (its text is never stored)."""
    if isinstance(exc, FileNotFoundError):
        return "data_missing"
    if isinstance(exc, UpstreamError):
        return "upstream_unavailable"
    try:
        import requests
        if isinstance(exc, requests.RequestException):
            return "upstream_unavailable"
    except ImportError:  # pragma: no cover
        pass
    return "internal_error"


def _add_error(errors: List[dict], entry: dict) -> None:
    if entry not in errors:
        errors.append(entry)


# ---------------------------------------------------------------------------
# Demand
# ---------------------------------------------------------------------------
def _local_sheet_id(psp_dir: Path, day: dt.date) -> int:
    """Highest upload Id among local spreadsheet reports of ``day`` (0 when none)."""
    prefix = f"{day.strftime('%d.%m.%y')}_NLDC_PSP"
    best = 0
    for ext in SPREADSHEET_EXTS:
        for path in Path(psp_dir).glob(f"{prefix}*.{ext}"):
            m = PSP_ID_RE.search(path.name)
            best = max(best, int(m.group(1)) if m else 0)
    return best


def _parse_local_reports(psp_dir: Path, days: List[str]):
    """
    Parse the newest local report per date.
    Returns (wide, long, gen DataFrames, unparsed dates, {unparsed date: failing file}).
    """
    chosen = select_files_per_date(Path(psp_dir))
    wide, long_rows, gen_rows, unparsed, failed = [], [], [], [], {}
    for day in days:
        path = chosen.get(day)
        res = parse_psp_xls(str(path)) if path else None
        if res is None:
            unparsed.append(day)
            if path:
                failed[day] = Path(path)
            continue
        wide.append(res["state_demand_wide"])
        long_rows.extend(res["state_demand_long"])
        gen_rows.extend(res["generation_long"])
    return pd.DataFrame(wide), pd.DataFrame(long_rows), pd.DataFrame(gen_rows), unparsed, failed


def _quarantine(path: Path) -> None:
    """Rename an unparseable report to ``*.bad`` so it no longer blocks its date."""
    target = path.with_name(path.name + QUARANTINE_SUFFIX)
    try:
        os.replace(path, target)
        logger.warning("Quarantined unparseable report %s -> %s", path.name, target.name)
    except OSError as exc:
        logger.error("Could not quarantine %s: %s", path.name, exc)


def _retry_unparseable(paths: DataPaths, failed: Dict[str, Path], ranked: dict, session) -> Set[str]:
    """
    Quarantine each unparseable local report of a missing date and, for files that were
    already on disk before this run, download the newest listed candidate again (atomic
    .part -> rename) so the caller can parse it once more. Returns the re-downloaded dates.
    """
    retried: Set[str] = set()
    for iso, bad in sorted(failed.items()):
        _quarantine(bad)
        cands = ranked.get(dt.date.fromisoformat(iso))
        if not cands:
            continue
        name, err = download_best(session, dt.date.fromisoformat(iso), cands, paths.psp_dir)
        if name is None:
            logger.error("Re-download of %s after a parse failure failed: %s", iso, err)
            continue
        retried.add(iso)
        time.sleep(0.3)
    return retried


def _stored_rows(path: Path, dates: Set[str]) -> Dict[str, Dict[str, str]]:
    import csv
    out: Dict[str, Dict[str, str]] = {}
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            if rec["Date"] in dates:
                out[rec["Date"]] = rec
    return out


def _row_differs(stored: Dict[str, str], parsed: pd.Series) -> bool:
    for col, new in parsed.items():
        if col == "Date" or col not in stored:
            continue
        old = float(stored[col]) if stored[col] != "" else math.nan
        new = float(new) if pd.notna(new) else math.nan
        if math.isnan(old) != math.isnan(new):
            return True
        if not math.isnan(old) and abs(old - new) > VALUE_TOLERANCE:
            return True
    return False


def _state_mu_1_frame(path: Path, df_long: pd.DataFrame, dates: Set[str]) -> pd.DataFrame:
    """Every day lists the same entity set as the latest earlier day (as replace_fabricated_demand does)."""
    orig = pd.read_csv(path, usecols=["Region", "States", "date"])
    base = orig[~orig["date"].isin(dates)]
    template = base[base["date"] == base["date"].max()][["Region", "States"]]
    frames = []
    for day, grp in df_long[df_long["date"].isin(dates)].groupby("date"):
        rows = template.merge(grp.drop(columns=["Region"]), on="States", how="left")
        rows["date"] = day
        rows["Shortage during maximum Demand(MW)"] = rows["Shortage during maximum Demand(MW)"].fillna(0.0)
        frames.append(rows)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _long_gen_frame(df_gen: pd.DataFrame) -> pd.DataFrame:
    if df_gen.empty:
        return pd.DataFrame()
    long_new = df_gen.melt(id_vars=["source", "date"], value_vars=GEN_REGION_COLS,
                           var_name="region", value_name="regional_generation_mu")
    long_new["region"] = long_new["region"].str.lower()
    return long_new.dropna(subset=["regional_generation_mu"]).sort_values(["date", "region", "source"])


def _write_demand_files(paths: DataPaths, wide: pd.DataFrame, df_long: pd.DataFrame, df_gen: pd.DataFrame,
                        add: Set[str], revise: Set[str]) -> None:
    """Upsert into all demand files; the main file goes LAST so an interrupted run is retried."""
    dates = add | revise
    keys = {(d,) for d in revise}
    gen = df_gen[df_gen["date"].isin(dates)] if not df_gen.empty else df_gen
    if paths.long_state_mu2.exists():
        upsert_csv(paths.long_state_mu2, _long_gen_frame(gen), ["date"], keys)
    if paths.state_mu_2.exists() and not gen.empty:
        upsert_csv(paths.state_mu_2, gen, ["date"], keys)
    if paths.state_mu_1.exists() and not df_long.empty:
        upsert_csv(paths.state_mu_1, _state_mu_1_frame(paths.state_mu_1, df_long, dates), ["date"], keys)
    new_wide = wide[wide["Date"].isin(dates)]
    for target in (paths.demand_mirror, paths.demand_main):
        if target.exists():
            upsert_csv(target, new_wide, ["Date"], keys)


def update_demand(paths: DataPaths, start: dt.date, end: dt.date, today: dt.date,
                  session, dry_run: bool = False) -> dict:
    """Add missing dates / apply newer revisions in [start, end]. See module docstring."""
    out = {"added": [], "revised": [], "still_missing": [], "would_fetch": [], "errors": []}
    if not paths.demand_main.exists():
        raise FileNotFoundError(f"{paths.demand_main} not found (is the data/ volume mounted?)")
    window = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    main_dates = {k[0] for k in scan_keys(paths.demand_main, ["Date"])}
    missing = [d for d in window if d.isoformat() not in main_dates]

    missing_iso = {d.isoformat() for d in missing}
    ranked = rank_candidates(collect_items(session, start, end), start, end)
    if not dry_run:
        paths.psp_dir.mkdir(parents=True, exist_ok=True)
    reasons: Dict[str, str] = {}
    recheck: List[str] = []   # dates already stored whose local report must be re-compared
    downloaded_now: Set[str] = set()

    for day in window:
        iso, cands = day.isoformat(), ranked.get(day)
        is_missing = iso in missing_iso
        if not cands:
            if is_missing:
                reasons[iso] = REASON_NOT_PUBLISHED if day >= today - dt.timedelta(days=2) else REASON_NOT_LISTED
            continue
        if is_missing:
            listed_sheets = [c for c in cands if item_ext(c) in SPREADSHEET_EXTS]
            if already_present(paths.psp_dir, day, SPREADSHEET_EXTS[0]):
                continue                    # parse the local spreadsheet (retried below if bad)
            if already_present(paths.psp_dir, day, PDF_EXT):
                if not listed_sheets:
                    continue                # only a PDF exists anywhere: parse the local one
                wanted = listed_sheets      # a spreadsheet is now listed: prefer it over the PDF
            else:
                wanted = cands
        elif iso >= FIRST_REVISABLE_DATE:
            local_id = _local_sheet_id(paths.psp_dir, day)
            wanted = [c for c in cands if item_ext(c) in SPREADSHEET_EXTS and c["Id"] > local_id]
            recheck.append(iso)
            if not wanted:
                continue
        else:
            continue
        out["would_fetch"].append(iso)
        if dry_run:
            continue
        name, err = download_best(session, day, wanted, paths.psp_dir)
        if name is not None:
            downloaded_now.add(iso)
        elif is_missing:
            logger.error("Download failed for %s: %s", iso, err)
            reasons[iso] = REASON_DOWNLOAD_404 if err and "HTTP 404" in err else REASON_DOWNLOAD_FAILED
        else:
            logger.error("Newer revision of %s could not be downloaded: %s", iso, err)
            _add_error(out["errors"], make_error("demand", "revision_download_failed", iso))
        time.sleep(0.3)

    if dry_run:
        out["still_missing"] = [{"date": d.isoformat(), "reason": reasons.get(d.isoformat(), REASON_WOULD_DOWNLOAD)}
                                for d in missing]
        return out

    to_parse = sorted({d.isoformat() for d in missing} | set(recheck))
    wide, df_long, df_gen, unparsed, failed = _parse_local_reports(paths.psp_dir, to_parse)
    # L2: an unparseable report of a MISSING date is quarantined; one that was already on
    # disk is downloaded again and parsed once more (a fresh download that fails is retried
    # by the next run). Stored dates (revision re-checks) are left alone.
    failed = {d: f for d, f in failed.items() if d in missing_iso}
    if failed:
        retried = _retry_unparseable(
            paths, {d: f for d, f in failed.items() if d not in downloaded_now}, ranked, session)
        for d in set(failed) & downloaded_now:
            _quarantine(failed[d])
        if retried:
            w2, l2, g2, unparsed2, failed2 = _parse_local_reports(paths.psp_dir, sorted(retried))
            for f in failed2.values():
                _quarantine(f)
            wide = pd.concat([wide, w2], ignore_index=True)
            df_long = pd.concat([df_long, l2], ignore_index=True)
            df_gen = pd.concat([df_gen, g2], ignore_index=True)
            unparsed = [d for d in unparsed if d not in retried] + unparsed2
    if not wide.empty:
        wide_ok, rejected = validate_parsed(wide)
    else:
        wide_ok, rejected = wide, []
    ok_dates = set(wide_ok["Date"]) if not wide_ok.empty else set()

    add = {d.isoformat() for d in missing if d.isoformat() in ok_dates}
    stored = _stored_rows(paths.demand_main, ok_dates - add) if ok_dates - add else {}
    by_date = wide_ok.set_index("Date") if not wide_ok.empty else wide_ok
    revise = {d for d in stored if _row_differs(stored[d], by_date.loc[d])}

    for d in missing:
        iso = d.isoformat()
        if iso in add:
            continue
        if iso in rejected:
            reasons[iso] = REASON_IMPLAUSIBLE
        elif iso in unparsed and iso not in reasons:
            reasons[iso] = REASON_UNPARSEABLE
        reasons.setdefault(iso, REASON_NO_USABLE)

    if add or revise:
        _write_demand_files(paths, wide_ok, df_long, df_gen, add, revise)
    out["added"] = sorted(add)
    out["revised"] = sorted(revise)
    out["still_missing"] = [{"date": d.isoformat(), "reason": reasons[d.isoformat()]}
                            for d in missing if d.isoformat() not in add]
    return out


# ---------------------------------------------------------------------------
# Caches, tail dates, status
# ---------------------------------------------------------------------------
_last_signature: Optional[Tuple] = None


def _data_signature(paths: DataPaths) -> Tuple:
    sig = []
    for p in (paths.demand_main, paths.zenodo):
        try:
            st = p.stat()
            sig.append((st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append(None)
    return tuple(sig)


def refresh_caches_if_data_changed(paths: Optional[DataPaths] = None) -> bool:
    """Clear data_loading caches when the demand/weather files changed (e.g. another worker updated them)."""
    global _last_signature
    sig = _data_signature(paths or default_paths())
    if sig == _last_signature:
        return False
    _last_signature = sig
    from data_loading import clear_data_caches
    clear_data_caches()
    return True


def clear_caches_and_mark(paths: DataPaths) -> None:
    """Clear data_loading caches and remember the data signature they now correspond to."""
    global _last_signature
    from data_loading import clear_data_caches
    clear_data_caches()
    _last_signature = _data_signature(paths)


def last_csv_date(path: Path) -> Optional[str]:
    """Date (first column) of the last non-empty line, read from the file tail."""
    try:
        with Path(path).open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            lines = [ln for ln in f.read().decode("utf-8", "ignore").splitlines() if ln.strip()]
        return lines[-1].split(",", 1)[0] if lines else None
    except OSError:
        return None


def _write_json_atomic(path: Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(path).with_name(Path(path).name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def read_last_summary(paths: Optional[DataPaths] = None) -> Optional[dict]:
    try:
        return json.loads(Path((paths or default_paths()).status_file).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _iso_date_or_none(value) -> Optional[str]:
    if not (isinstance(value, str) and ISO_DATE_RE.fullmatch(value)):
        return None
    try:
        dt.date.fromisoformat(value)
    except ValueError:
        return None
    return value


def _iso_datetime_or_none(value) -> Optional[str]:
    if not (isinstance(value, str) and ISO_DATETIME_RE.fullmatch(value)):
        return None
    try:
        dt.datetime.fromisoformat(value)  # regex shape alone admits e.g. 2026-99-99
    except ValueError:
        return None
    return value


def _date_list(value) -> List[str]:
    return sorted({d for d in value if _iso_date_or_none(d)}) if isinstance(value, list) else []


def _public_errors(value) -> List[dict]:
    """Rebuild error entries from the fixed vocabulary; anything else (e.g. legacy free text) -> internal_error."""
    out: List[dict] = []
    for entry in value if isinstance(value, list) else []:
        if isinstance(entry, dict):
            _add_error(out, make_error(str(entry.get("stage")), str(entry.get("code")), entry.get("date")))
        else:
            _add_error(out, make_error("job", "internal_error"))
    return out


def public_summary(last: dict) -> dict:
    """
    The part of a run summary that the PUBLIC status endpoint may show. Every value is
    rebuilt from validated pieces (ISO dates, fixed reason/code vocabularies, ints), so a
    status file written by an older version, or tampered with, cannot leak paths, URLs or
    exception text.
    """
    weather = last.get("weather_dates_added") if isinstance(last.get("weather_dates_added"), dict) else {}
    missing = []
    for item in last.get("dates_still_missing") if isinstance(last.get("dates_still_missing"), list) else []:
        if isinstance(item, dict) and _iso_date_or_none(item.get("date")):
            reason = item.get("reason")
            missing.append({"date": item["date"], "reason": reason if reason in MISSING_REASONS else REASON_NO_USABLE})
    reconciled = last.get("ledger_reconciled")
    return {
        "dates_added": _date_list(last.get("dates_added")),
        "dates_revised": _date_list(last.get("dates_revised")),
        "dates_still_missing": missing,
        "weather_dates_added": {k: _date_list(weather.get(k)) for k in WEATHER_ADDED_KEYS},
        "ledger_reconciled": reconciled if isinstance(reconciled, int) and not isinstance(reconciled, bool) else None,
        "errors": _public_errors(last.get("errors")),
    }


def _read_skip_record(paths: DataPaths) -> Optional[dict]:
    try:
        rec = json.loads(Path(paths.skip_file).read_text(encoding="utf-8"))
        return rec if isinstance(rec, dict) else None
    except (OSError, ValueError):
        return None


def is_update_running(paths: Optional[DataPaths] = None) -> bool:
    """True while a live process holds the daily-update lock."""
    return FileLock((paths or default_paths()).lock_file, LOCK_STALE_SECONDS).is_locked()


def get_update_status(paths: Optional[DataPaths] = None, now: Optional[dt.datetime] = None) -> dict:
    """
    Read-only status for the PUBLIC API: last run (sanitised, see ``public_summary``),
    whether a run is in progress, the last lock-contention skip, next scheduled run and
    newest demand/weather dates. Contains no paths, URLs or exception text.
    """
    from utils.update_scheduler import next_scheduled_run, scheduler_settings
    paths = paths or default_paths()
    last = read_last_summary(paths)
    last = last if isinstance(last, dict) else None
    enabled, time_str = scheduler_settings()
    provisional = sorted(d for d, s in load_provisional(paths.provisional).items() if s and _iso_date_or_none(d))
    last_run_at = _iso_datetime_or_none((last or {}).get("finished_at"))
    skip = _read_skip_record(paths) or {}
    skipped_at = _iso_datetime_or_none(skip.get("skipped_at"))
    if skipped_at and last_run_at and dt.datetime.fromisoformat(skipped_at) <= dt.datetime.fromisoformat(last_run_at):
        skipped_at = None   # a later run finished since; the skip is no longer news
    status = (last or {}).get("status")
    return {
        "running": is_update_running(paths),
        "last_run_at": last_run_at,
        "last_run_status": status if status in RUN_STATUSES else None,
        "last_run_summary": public_summary(last) if last else None,
        "last_skipped_at": skipped_at,
        "last_skip_reason": SKIP_REASON_LOCK_HELD if skipped_at else None,
        "scheduler_enabled": enabled,
        "next_scheduled_run": next_scheduled_run(now, time_str).isoformat() if enabled else None,
        "last_demand_date": _iso_date_or_none(last_csv_date(paths.demand_main)),
        "last_weather_date": _iso_date_or_none(last_csv_date(paths.zenodo)),
        "weather_provisional_dates": provisional,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _attach_log(path: Path) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    return handler


def _reconcile_ledger(dates: List[str], ingest_fn: Callable) -> int:
    days = sorted(dt.date.fromisoformat(d) for d in dates)
    result = ingest_fn(target_date=days[-1], catchup_days=(days[-1] - days[0]).days + 1)
    return int(result.get("reconciled_forecasts", 0))


def run_daily_update(
    end_date: Optional[dt.date] = None,
    catchup_days: int = DEFAULT_CATCHUP_DAYS,
    dry_run: bool = False,
    weather_backfill_from: Optional[dt.date] = None,
    paths: Optional[DataPaths] = None,
    session=None,
    ingest_fn: Optional[Callable] = None,
    now: Optional[dt.datetime] = None,
    lock: Optional[FileLock] = None,
) -> dict:
    """
    Run the full daily update once and return a summary dict (also logged and persisted).

    ``lock``: an already-acquired FileLock on ``paths.lock_file`` (used by
    ``start_background_update``); it is released when the run ends.
    """
    paths = paths or default_paths()
    today = today_ist(now)
    end = end_date or today - dt.timedelta(days=1)
    start = end - dt.timedelta(days=max(catchup_days, 1) - 1)
    summary = {
        "status": "ok", "dry_run": dry_run, "started_at": dt.datetime.now(IST).isoformat(timespec="seconds"),
        "end_date": end.isoformat(), "window_start": start.isoformat(),
        "dates_added": [], "dates_revised": [], "dates_still_missing": [],
        "weather_dates_added": {k: [] for k in WEATHER_ADDED_KEYS},
        "weather_still_missing": [], "weather_provisional_pending": [],
        "ledger_reconciled": None, "last_demand_date": None, "last_weather_date": None, "errors": [],
    }
    if not dry_run and lock is None:
        lock = FileLock(paths.lock_file, LOCK_STALE_SECONDS)
        if not lock.acquire():
            summary.update(status="skipped", reason=SKIP_REASON_LOCK_HELD)
            _record_skip(paths, summary)
            return summary

    # Dry run: no log FILE (the CLI's console handler still prints to stderr).
    handler = None if dry_run else _attach_log(paths.log_file)
    try:
        logger.info("Daily update start: end=%s window_start=%s dry_run=%s", end, start, dry_run)
        session = session or make_session()
        _run_steps(summary, paths, start, end, today, session, dry_run, weather_backfill_from, ingest_fn)
    except Exception as exc:  # never let a bad run crash the server's scheduler thread
        logger.exception("Daily update failed")
        _add_error(summary["errors"], make_error("job", classify_exception(exc)))
    finally:
        summary["finished_at"] = dt.datetime.now(IST).isoformat(timespec="seconds")
        summary["status"] = "error" if summary["errors"] else "ok"
        summary["last_demand_date"] = last_csv_date(paths.demand_main)
        summary["last_weather_date"] = last_csv_date(paths.zenodo)
        logger.info("Daily update finished: %s", json.dumps(summary, default=str))
        if not dry_run:
            try:
                _write_json_atomic(paths.status_file, summary)
            except OSError as exc:
                logger.error("Could not write status file: %s", exc)
            lock.release()
        if handler is not None:
            logger.removeHandler(handler)
            handler.close()
    return summary


def _record_skip(paths: DataPaths, summary: dict) -> None:
    """Persist a lock-contention skip separately, so the last real run's summary is kept."""
    logger.warning("Daily update skipped: another run holds %s", paths.lock_file.name)
    try:
        _write_json_atomic(paths.skip_file, {"skipped_at": summary["started_at"], "reason": SKIP_REASON_LOCK_HELD})
    except OSError as exc:
        logger.error("Could not write skip record: %s", exc)


_background_thread: Optional[threading.Thread] = None


def start_background_update(job: Callable[[FileLock], object], paths: Optional[DataPaths] = None) -> bool:
    """
    Acquire the daily-update lock in the CALLER's thread and, if that succeeds, run
    ``job(lock)`` in a daemon thread (the job must pass the lock to ``run_daily_update``;
    it is released here in any case). Returns False, starting nothing, when a run is
    already in progress. If the process dies mid-run the lock's PID check frees it.
    """
    global _background_thread
    lock = FileLock((paths or default_paths()).lock_file, LOCK_STALE_SECONDS)
    if not lock.acquire():
        return False

    def target() -> None:
        try:
            job(lock)
        except Exception:
            logger.exception("Background data update failed")
        finally:
            lock.release()

    thread = threading.Thread(target=target, name="loadly-data-update-manual", daemon=True)
    try:
        thread.start()
    except BaseException:
        lock.release()
        raise
    _background_thread = thread
    return True


def join_background_update(timeout: Optional[float] = None) -> None:
    """Wait for the last background run started by ``start_background_update`` (tests/CLI)."""
    if _background_thread is not None:
        _background_thread.join(timeout)


def _run_steps(summary, paths, start, end, today, session, dry_run, backfill_from, ingest_fn) -> None:
    try:
        demand = update_demand(paths, start, end, today, session, dry_run)
        summary["dates_added"] = demand["added"]
        summary["dates_revised"] = demand["revised"]
        summary["dates_still_missing"] = demand["still_missing"]
        for entry in demand["errors"]:
            _add_error(summary["errors"], entry)
        if dry_run:
            summary["dates_would_fetch"] = demand["would_fetch"]
    except Exception as exc:
        logger.exception("Demand update failed")
        _add_error(summary["errors"], make_error("demand", classify_exception(exc)))

    try:
        weather = update_weather(paths.weather, end, session, today, backfill_from, dry_run)
        summary["weather_dates_added"] = {k: weather[k] for k in WEATHER_ADDED_KEYS}
        summary["weather_still_missing"] = weather["still_missing"]
        summary["weather_provisional_pending"] = weather["provisional_pending"]
        for entry in weather["errors"]:
            _add_error(summary["errors"], make_error("weather", str(entry.get("code"))))
    except Exception as exc:
        logger.exception("Weather update failed")
        _add_error(summary["errors"], make_error("weather", classify_exception(exc)))

    changed = summary["dates_added"] + summary["dates_revised"]
    if dry_run:
        return
    if changed:
        try:
            if ingest_fn is None:
                from utils.ingestion_worker import run_ingestion_pipeline as ingest_fn
            summary["ledger_reconciled"] = _reconcile_ledger(changed, ingest_fn)
        except Exception:
            logger.exception("Ledger reconciliation failed")
            _add_error(summary["errors"], make_error("ledger", "ledger_reconcile_failed"))
    try:
        clear_caches_and_mark(paths)
    except Exception as exc:  # data_loading is optional for the CLI
        logger.warning("Could not clear data caches: %s", exc)


def _parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Daily Grid-India demand + Open-Meteo weather refresh.")
    parser.add_argument("--end", type=_parse_date, help="last date to fill (default: yesterday, IST)")
    parser.add_argument("--catchup", type=int, default=DEFAULT_CATCHUP_DAYS, help="days back to check for missing demand")
    parser.add_argument("--weather-backfill-from", type=_parse_date, help="also fill missing weather from this date")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change; writes no data and no log file (logs to stderr). "
                             "Still queries the Grid-India listing API; downloads nothing, no weather requests.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    summary = run_daily_update(args.end, args.catchup, args.dry_run, args.weather_backfill_from)
    print(json.dumps(summary, indent=2, default=str))
    return 1 if summary["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
