"""
Hardening of the daily data update (review 2026-11-22): sanitised public status (H1),
provisional-window weather (M1), PID-aware lock (M2), listing-record validation (L1),
unparseable-report recovery (L2), background POST (L3) and log-free dry run (L4).

Everything runs against tmp_path trees with fake HTTP sessions (no network, no data/).
"""

import datetime as dt
import json
import os
import subprocess
import sys
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from config import ALL_STATES
from server import app
from utils import daily_update, fetch_psp_reports, file_lock, model_retrainer, weather_update
from utils.daily_update import DataPaths, run_daily_update
from utils.file_lock import FileLock, current_host, pid_alive
from utils.fetch_psp_reports import local_filename, rank_candidates, safe_destination, validate_item

# Shared fakes/fixtures (the autouse fixture makes sleeps instant and fakes the PSP parser).
from tests.test_daily_update import (  # noqa: F401
    ARCHIVE_TEMP_C, BAD_CONTENT, END, NOW, FakeSession, fast_and_fake, listing_for, new_ingest,
    paths, psp_item, reports_17_to_20, run, sha, zenodo_rows,
)

client = TestClient(app)
LEAK_PATH = r"C:\Users\someone\secret\data\daily_energy_met_MU.csv"
LEAK_TEXT = f"kaboom at {LEAK_PATH} via http://internal.example/api"


def dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def lock_line(pid: int, host: str) -> str:
    return f"{pid} {host} deadbeef {dt.datetime.now().isoformat()}\n"


# ---------------------------------------------------------------------------
# H1: the public status endpoint never leaks exception text, paths or URLs
# ---------------------------------------------------------------------------
def assert_no_leak(text: str) -> None:
    assert "/" not in text and "\\" not in text
    assert "http" not in text.lower()
    assert "kaboom" not in text and "internal.example" not in text and "giving up" not in text


def test_public_status_after_failing_run_has_only_sanitised_errors(paths, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError(LEAK_TEXT)
    monkeypatch.setattr(daily_update, "default_paths", lambda: paths)
    monkeypatch.setattr(daily_update, "collect_items", boom)
    summary = run(paths, FakeSession(fail_weather=True))
    assert summary["status"] == "error"
    codes = {(e["stage"], e["code"]) for e in summary["errors"]}
    assert codes == {("demand", "internal_error"), ("weather", "upstream_unavailable")}
    assert_no_leak(json.dumps(summary["errors"]))
    assert_no_leak(paths.status_file.read_text(encoding="utf-8"))
    log = paths.log_file.read_text(encoding="utf-8")
    assert "kaboom" in log and "open-meteo" in log            # full detail is kept in the log only

    res = client.get("/api/data/update-status")
    assert res.status_code == 200
    assert_no_leak(res.text)
    body = res.json()
    assert body["last_run_status"] == "error"
    assert {e["code"] for e in body["last_run_summary"]["errors"]} == {"internal_error", "upstream_unavailable"}


def test_public_status_sanitises_legacy_or_tampered_status_file(paths, monkeypatch):
    monkeypatch.setattr(daily_update, "default_paths", lambda: paths)
    paths.status_file.parent.mkdir(parents=True, exist_ok=True)
    paths.status_file.write_text(json.dumps({
        "status": LEAK_TEXT, "finished_at": LEAK_PATH,
        "dates_added": ["2026-11-19", LEAK_PATH], "dates_revised": [LEAK_TEXT],
        "dates_still_missing": [{"date": "2026-11-20", "reason": f"download failed ({LEAK_TEXT})"},
                                {"date": LEAK_PATH, "reason": "not published yet"}],
        "weather_dates_added": {"archive": [LEAK_PATH], LEAK_TEXT: ["2026-11-19"]},
        "ledger_reconciled": LEAK_TEXT,
        "errors": [f"demand update failed: {LEAK_TEXT}",
                   {"stage": LEAK_TEXT, "code": LEAK_TEXT, "message": LEAK_TEXT, "date": LEAK_PATH}],
        "extra_field": LEAK_TEXT,
    }), encoding="utf-8")
    paths.skip_file.write_text(json.dumps({"skipped_at": LEAK_TEXT, "reason": LEAK_TEXT}), encoding="utf-8")

    res = client.get("/api/data/update-status")
    assert res.status_code == 200
    assert_no_leak(res.text)
    body = res.json()
    summary = body["last_run_summary"]
    assert body["last_run_status"] is None and body["last_run_at"] is None
    assert summary["dates_added"] == ["2026-11-19"] and summary["dates_revised"] == []
    assert summary["dates_still_missing"] == [{"date": "2026-11-20", "reason": daily_update.REASON_NO_USABLE}]
    assert summary["errors"] == [{"stage": "job", "code": "internal_error",
                                  "message": daily_update.ERROR_MESSAGES["internal_error"], "date": None}]
    assert summary["ledger_reconciled"] is None
    assert body["last_skipped_at"] is None


def test_download_failure_reason_is_fixed_text(paths):
    session = FakeSession(listing=listing_for(psp_item(dt.date(2026, 11, 17), 49000)), broken_files=["17.11.26"])
    summary = run(paths, session)
    missing = {m["date"]: m["reason"] for m in summary["dates_still_missing"]}
    assert missing["2026-11-17"] == daily_update.REASON_DOWNLOAD_404
    assert_no_leak(json.dumps(summary["dates_still_missing"]))


# ---------------------------------------------------------------------------
# M1: recent weather is provisional even from the archive; refreshed; finalised
# ---------------------------------------------------------------------------
def test_recent_archive_rows_are_provisional_and_refreshed_when_values_change(paths):
    run(paths, reports_17_to_20())
    prov = weather_update.load_provisional(paths.provisional)
    assert set(prov) == {"2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20"}
    assert all(states == set(ALL_STATES) for states in prov.values())

    # ERA5 later revises the recent days: rows are replaced in place, still provisional.
    summary = run(paths, reports_17_to_20(archive_temp=ARCHIVE_TEMP_C + 1.0))
    assert summary["weather_dates_added"]["upgraded_to_archive"] == ["2026-11-17", "2026-11-18",
                                                                     "2026-11-19", "2026-11-20"]
    rows = zenodo_rows(paths, "2026-11-19")
    assert len(rows) == len(ALL_STATES)
    assert rows["2m_temperature_mean"].eq(ARCHIVE_TEMP_C + 1.0 + 273.15).all()
    assert weather_update.load_provisional(paths.provisional)["2026-11-19"] == set(ALL_STATES)


def test_marker_is_dropped_once_the_date_leaves_the_window(paths):
    run(paths, reports_17_to_20())
    later = NOW + dt.timedelta(days=weather_update.WEATHER_PROVISIONAL_DAYS + 2)   # today 2026-11-30
    before = sha(paths.zenodo)
    summary = run_daily_update(end_date=END, catchup_days=1, paths=paths, session=reports_17_to_20(),
                               ingest_fn=new_ingest()[0], now=later)
    assert summary["weather_dates_added"]["finalized"] == ["2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20"]
    assert summary["weather_provisional_pending"] == []
    assert weather_update.load_provisional(paths.provisional) == {}
    assert sha(paths.zenodo) == before                          # archive unchanged -> no rewrite

    # Final rows are never fetched again.
    session = reports_17_to_20()
    run_daily_update(end_date=END, catchup_days=1, paths=paths, session=session,
                     ingest_fn=new_ingest()[0], now=later)
    assert not session.weather_calls()


def test_stored_recent_rows_without_marker_are_adopted_as_provisional(paths):
    """One-time migration: rows written before the window rule existed get marked and refreshed."""
    assert not paths.provisional.exists()                       # fixture rows 17/18 are unmarked
    summary = run(paths, reports_17_to_20(), catchup=1)
    assert {"2026-11-17", "2026-11-18"} <= set(summary["weather_provisional_pending"])
    assert summary["weather_dates_added"]["upgraded_to_archive"] == ["2026-11-17", "2026-11-18"]


def test_settled_rows_outside_the_window_are_not_touched(paths):
    before = sha(paths.zenodo)
    session = reports_17_to_20()
    later = dt.datetime(2026, 11, 30, 14, 0, tzinfo=NOW.tzinfo)
    run_daily_update(end_date=dt.date(2026, 11, 18), catchup_days=1, paths=paths, session=session,
                     ingest_fn=new_ingest()[0], now=later)
    assert not session.weather_calls()
    assert sha(paths.zenodo) == before


# ---------------------------------------------------------------------------
# M2: PID-aware lock
# ---------------------------------------------------------------------------
def test_pid_alive_detects_own_and_dead_process():
    assert pid_alive(os.getpid())
    assert not pid_alive(dead_pid())
    assert not pid_alive(0)


def test_lock_of_dead_process_on_this_host_is_reclaimed_immediately(tmp_path):
    path = tmp_path / "x.lock"
    path.write_text(lock_line(dead_pid(), current_host()))
    lock = FileLock(path, stale_seconds=3600)
    assert not lock.is_locked()
    assert lock.acquire()
    assert path.read_text().split()[0] == str(os.getpid())
    assert not list(tmp_path.glob("x.lock.*"))                 # no graveyard / guard left behind
    lock.release()
    assert not path.exists()


def test_lock_of_live_process_or_other_host_is_kept_until_stale(tmp_path):
    path = tmp_path / "x.lock"
    path.write_text(lock_line(os.getpid(), current_host()))
    assert FileLock(path, 3600).is_locked()
    assert not FileLock(path, 3600).acquire()
    path.write_text(lock_line(dead_pid(), "some-other-host"))
    assert not FileLock(path, 3600).acquire()                 # cannot check a foreign PID
    old = time.time() - 7200
    os.utime(path, (old, old))
    assert FileLock(path, 3600).acquire()                     # age fallback still works


def test_live_local_holder_is_not_displaced_by_age_rule(tmp_path):
    # A retrain/update running longer than stale_seconds must not let a second run start.
    path = tmp_path / "x.lock"
    path.write_text(lock_line(os.getpid(), current_host()))
    old = time.time() - 7200
    os.utime(path, (old, old))
    assert FileLock(path, 3600).is_locked()
    assert not FileLock(path, 3600).acquire()
    ancient = time.time() - file_lock.LIVE_HOLDER_CEILING_SECONDS - 60
    os.utime(path, (ancient, ancient))
    assert FileLock(path, 3600).acquire()                     # PID-reuse ceiling still reclaims


def test_release_keeps_an_unreadable_lock_file(tmp_path, monkeypatch):
    path = tmp_path / "x.lock"
    lock = FileLock(path, 3600)
    assert lock.acquire()
    monkeypatch.setattr(file_lock, "read_lock_info",
                        lambda p: file_lock.LockInfo(raw="", pid=None, host=None, token=None))
    lock.release()
    assert path.exists()                                      # ownership unprovable -> keep


@pytest.mark.parametrize("payload", [
    {"provisional": ["2026-11-20"]},                          # list instead of dict
    {"provisional": {"2026-11-20": "Kerala"}},                # string instead of list
    {"provisional": {"2026-99-99": ["Kerala"]}},              # impossible date
    ["not", "a", "dict"],
])
def test_malformed_provisional_file_is_ignored_not_fatal(paths, monkeypatch, payload):
    monkeypatch.setattr(daily_update, "default_paths", lambda: paths)
    paths.provisional.parent.mkdir(parents=True, exist_ok=True)
    paths.provisional.write_text(json.dumps(payload), encoding="utf-8")
    assert weather_update.load_provisional(paths.provisional) == {}
    res = client.get("/api/data/update-status")
    assert res.status_code == 200
    assert res.json()["weather_provisional_dates"] == []


def test_impossible_skip_datetime_does_not_500(paths, monkeypatch):
    monkeypatch.setattr(daily_update, "default_paths", lambda: paths)
    paths.status_file.parent.mkdir(parents=True, exist_ok=True)
    paths.status_file.write_text(json.dumps({"status": "ok", "finished_at": "2026-11-22T00:00:00"}),
                                 encoding="utf-8")
    paths.skip_file.write_text(json.dumps({"skipped_at": "2026-99-99T00:00:00"}), encoding="utf-8")
    res = client.get("/api/data/update-status")
    assert res.status_code == 200
    assert res.json()["last_skipped_at"] is None


def test_release_does_not_delete_a_lock_that_was_reclaimed_by_someone_else(tmp_path):
    path = tmp_path / "x.lock"
    lock = FileLock(path, 3600)
    assert lock.acquire()
    path.write_text(lock_line(os.getpid(), current_host()))   # another holder took over
    lock.release()
    assert path.exists()


def test_stale_reclaim_guard_is_cleared(tmp_path):
    path = tmp_path / "x.lock"
    path.write_text(lock_line(dead_pid(), current_host()))
    guard = tmp_path / ("x.lock" + file_lock.RECLAIM_GUARD_SUFFIX)
    guard.write_text("")
    old = time.time() - file_lock.RECLAIM_GUARD_STALE_SECONDS - 5
    os.utime(guard, (old, old))
    lock = FileLock(path, 3600)
    assert not lock.acquire()          # first attempt clears the abandoned guard
    assert lock.acquire()
    lock.release()


def test_run_after_crashed_run_is_not_skipped(paths):
    paths.lock_file.parent.mkdir(parents=True, exist_ok=True)
    paths.lock_file.write_text(lock_line(dead_pid(), current_host()))
    assert run(paths, reports_17_to_20())["status"] == "ok"
    assert not paths.lock_file.exists()


def test_skip_due_to_lock_is_recorded_and_visible(paths, monkeypatch):
    monkeypatch.setattr(daily_update, "default_paths", lambda: paths)
    holder = FileLock(paths.lock_file, daily_update.LOCK_STALE_SECONDS)
    assert holder.acquire()
    try:
        summary = run(paths, reports_17_to_20())
        assert summary["status"] == "skipped" and summary["reason"] == daily_update.SKIP_REASON_LOCK_HELD
        body = client.get("/api/data/update-status").json()
        assert body["running"] is True
        assert body["last_skipped_at"] == summary["started_at"]
        assert body["last_skip_reason"] == daily_update.SKIP_REASON_LOCK_HELD
        assert not paths.status_file.exists()                  # the last real run's summary is kept
    finally:
        holder.release()
    assert client.get("/api/data/update-status").json()["running"] is False


def test_retrain_lock_of_dead_process_is_reclaimed():
    model_retrainer.RETRAIN_LOCK_PATH.write_text(lock_line(dead_pid(), current_host()))
    assert model_retrainer._acquire_retrain_lock()
    model_retrainer._release_retrain_lock()
    assert not model_retrainer.RETRAIN_LOCK_PATH.exists()


# ---------------------------------------------------------------------------
# L1: listing records are validated; traversal is impossible
# ---------------------------------------------------------------------------
GOOD = {"Id": 49523, "Field2": "2026-11-21", "FilePath": "files/grdw/2026/11/20.11.26_NLDC_PSP_523.xls",
        "Title_": "20.11.26_NLDC_PSP"}


@pytest.mark.parametrize("override", [
    {"Id": "..\\..\\evil"}, {"Id": "../../evil"}, {"Id": "12a"}, {"Id": -5}, {"Id": 0}, {"Id": None},
    {"Id": True}, {"Id": "1" * 13}, {"Id": "\u0661\u0662"}, {"Id": "123\n"},
    {"Field2": "21-11-2026"}, {"Field2": None}, {"Field2": "2026-13-45"},
    {"FilePath": "files/grdw/../../etc/20.11.26_NLDC_PSP.xls"}, {"FilePath": "../20.11.26_NLDC_PSP.xls"},
    {"FilePath": "/files/grdw/2026/11/20.11.26_NLDC_PSP.xls"}, {"FilePath": "https://evil/x.xls"},
    {"FilePath": "files/grdw/2026/11/20.11.26_NLDC_PSP.exe"}, {"FilePath": "files/grdw/2026\\11\\x.xls"},
    {"FilePath": "files/grdw/.hidden/x.xls"}, {"FilePath": "files/grdw/2026/11/x.xls\n"},
    {"FilePath": "files/other/2026/11/x.xls"},
], ids=lambda o: repr(o))
def test_malformed_listing_records_are_rejected(override):
    assert validate_item({**GOOD, **override}) is None


def test_valid_listing_records_are_normalised():
    assert validate_item(GOOD)["Id"] == 49523
    legacy = {**GOOD, "Id": "39283",
              "FilePath": "files/grdw/uploads/daily-reports/psp-reports/2024-2025/31.05.24_NLDC_PSP.xls"}
    assert validate_item(legacy)["Id"] == 39283


def test_malformed_record_is_skipped_without_aborting_the_ranking():
    bad = {**GOOD, "Id": "..\\..\\evil"}
    ranked = rank_candidates([bad, GOOD, "not-a-dict"], dt.date(2026, 11, 20), dt.date(2026, 11, 20))
    assert [c["Id"] for c in ranked[dt.date(2026, 11, 20)]] == [49523]


def test_file_names_are_built_only_from_validated_parts(tmp_path):
    assert local_filename(dt.date(2026, 11, 20), GOOD) == "20.11.26_NLDC_PSP_49523.xls"
    with pytest.raises(ValueError):
        local_filename(dt.date(2026, 11, 20), {**GOOD, "Id": "..\\..\\evil"})
    if os.name == "nt":
        # Backslash is a path separator only on Windows; on POSIX it is a literal name character.
        with pytest.raises(ValueError):
            safe_destination(tmp_path, "..\\evil.xls")
    else:
        assert safe_destination(tmp_path, "..\\evil.xls").parent == tmp_path.resolve()
    with pytest.raises(ValueError):
        safe_destination(tmp_path, "../evil.xls")
    assert safe_destination(tmp_path, "a.xls") == (tmp_path / "a.xls").resolve()


def test_run_with_a_malformed_record_still_adds_the_good_dates(paths):
    days = [dt.date(2026, 11, d) for d in (17, 18, 19, 20)]
    items = [psp_item(d, 49000 + i) for i, d in enumerate(days)]
    items.append({**psp_item(days[0], 1), "Id": "..\\..\\evil"})
    summary = run(paths, FakeSession(listing=listing_for(*items)))
    assert summary["dates_added"] == [d.isoformat() for d in days]
    assert not list(paths.psp_dir.parent.parent.rglob("*evil*"))


# ---------------------------------------------------------------------------
# L2: unparseable local reports do not wedge a date
# ---------------------------------------------------------------------------
def write_report(paths, name, content=b"x"):
    paths.psp_dir.mkdir(parents=True, exist_ok=True)
    path = paths.psp_dir / name
    path.write_bytes(content * (fetch_psp_reports.MIN_VALID_FILE_BYTES + 10))
    return path


def test_unparseable_local_report_is_quarantined_and_downloaded_again(paths):
    bad = write_report(paths, "17.11.26_NLDC_PSP_49000.xls", BAD_CONTENT)
    session = reports_17_to_20()
    summary = run(paths, session)
    assert "2026-11-17" in summary["dates_added"]
    assert (paths.psp_dir / (bad.name + daily_update.QUARANTINE_SUFFIX)).exists()
    assert not (paths.psp_dir / bad.name).read_bytes().startswith(BAD_CONTENT)
    assert [c for c in session.calls if "webcdn" in c[1] and "17.11.26" in c[1]]


def test_local_pdf_does_not_block_a_spreadsheet_listed_later(paths):
    write_report(paths, "17.11.26_NLDC_PSP_48000.pdf")
    session = reports_17_to_20()
    summary = run(paths, session)
    assert "2026-11-17" in summary["dates_added"]
    assert (paths.psp_dir / "17.11.26_NLDC_PSP_49000.xls").exists()
    assert pd.read_csv(paths.demand_main).set_index("Date").loc["2026-11-17", "S0"] == pytest.approx(149.0)


def test_persistently_bad_report_is_quarantined_and_reported(paths, monkeypatch):
    write_report(paths, "17.11.26_NLDC_PSP_49000.xls", BAD_CONTENT)
    session = reports_17_to_20()
    real_request = session.request

    def bad_cdn(method, url, **kw):
        resp = real_request(method, url, **kw)
        if "webcdn" in url and "17.11.26" in url:
            resp.content = BAD_CONTENT * (fetch_psp_reports.MIN_VALID_FILE_BYTES + 10)
        return resp
    monkeypatch.setattr(session, "request", bad_cdn)
    summary = run(paths, session)
    assert {"date": "2026-11-17", "reason": daily_update.REASON_UNPARSEABLE} in summary["dates_still_missing"]
    assert not list(paths.psp_dir.glob("17.11.26_NLDC_PSP*.xls"))       # nothing left to wedge the date
    assert list(paths.psp_dir.glob("17.11.26_NLDC_PSP*.bad"))


# ---------------------------------------------------------------------------
# L3: POST /api/data/update runs in the background
# ---------------------------------------------------------------------------
def test_post_update_returns_202_then_409_while_running(paths, admin_headers, monkeypatch):
    import threading
    import api.routes as routes
    monkeypatch.setattr(daily_update, "default_paths", lambda: paths)
    gate, started = threading.Event(), threading.Event()
    seen = {}

    def slow_run(catchup_days, lock):
        seen["lock_held_by_job"] = lock is not None and paths.lock_file.exists()
        started.set()
        gate.wait(10)
        return {"status": "ok"}
    monkeypatch.setattr(routes, "run_daily_update", slow_run)

    res = client.post("/api/data/update", json={"catchup_days": 2}, headers=admin_headers)
    assert res.status_code == 202 and res.json() == {"status": "started"}
    assert started.wait(10)
    assert client.get("/api/data/update-status").json()["running"] is True
    again = client.post("/api/data/update", json={"catchup_days": 2}, headers=admin_headers)
    assert again.status_code == 409
    gate.set()
    daily_update.join_background_update(timeout=10)
    assert seen["lock_held_by_job"]
    assert not paths.lock_file.exists()
    assert client.get("/api/data/update-status").json()["running"] is False


def test_background_update_runs_the_real_job_and_releases_the_lock(paths):
    def job(lock):
        return run_daily_update(end_date=END, catchup_days=4, paths=paths, session=reports_17_to_20(),
                                ingest_fn=new_ingest()[0], now=NOW, lock=lock)
    assert daily_update.start_background_update(job, paths)
    daily_update.join_background_update(timeout=30)
    assert json.loads(paths.status_file.read_text(encoding="utf-8"))["status"] == "ok"
    assert not paths.lock_file.exists()
