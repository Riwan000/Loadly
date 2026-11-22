"""
Daily update job: demand (Grid-India) + weather (Open-Meteo) + ledger + scheduler + status API.

No network and no writes to data/ or artifacts/: every file lives under tmp_path and all
HTTP goes through FakeSession. The PSP parser is replaced by a deterministic fake (real
spreadsheets are gitignored data).
"""

import datetime as dt
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.routes as routes
from config import ALL_STATES
from server import app
from utils import daily_update, fetch_psp_reports, update_scheduler, weather_update
from utils.csv_upsert import upsert_csv
from utils.daily_update import DataPaths, run_daily_update
from utils.fetch_recent_weather import STATE_FILE_MAP
from utils.file_lock import FileLock
from utils.parse_psp_reports import parse_date_from_filename

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
NOW = dt.datetime(2026, 11, 21, 14, 0, tzinfo=IST)          # "today" = 2026-11-21, end = 2026-11-20
END = dt.date(2026, 11, 20)
STATE_COLS = [f"S{i}" for i in range(32)]
DEMAND_HEADER = ["Date", *STATE_COLS, "Total"]
WEATHER_COLS = [
    "date", "10m_u_component_of_wind_min", "10m_u_component_of_wind_mean", "10m_u_component_of_wind_max",
    "10m_v_component_of_wind_min", "10m_v_component_of_wind_mean", "10m_v_component_of_wind_max",
    "2m_temperature_min", "2m_temperature_mean", "2m_temperature_max",
    "2m_dewpoint_temperature_min", "2m_dewpoint_temperature_mean", "2m_dewpoint_temperature_max",
    "surface_solar_radiation_downwards_min", "surface_solar_radiation_downwards_mean",
    "surface_solar_radiation_downwards_max", "total_cloud_cover_min", "total_cloud_cover_mean",
    "total_cloud_cover_max", "utci_min", "utci_mean", "utci_max",
]
ARCHIVE_TEMP_C, FORECAST_TEMP_C = 25.0, 27.0


# ---------------------------------------------------------------------------
# Fakes and fixtures
# ---------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status=200, payload=None, content=b""):
        self.status_code, self._payload, self.content = status, payload, content

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Routes Grid-India listing/CDN calls and Open-Meteo archive/forecast calls."""

    def __init__(self, listing=None, broken_files=(), archive_last_date=END, partial_day=None,
                 archive_temp=ARCHIVE_TEMP_C, fail_weather=False):
        self.headers = {}
        self.archive_temp = archive_temp
        self.fail_weather = fail_weather              # Open-Meteo answers 500 (URL ends up in the error)
        self.listing = listing or {}                  # "2026-27/11" -> [items]
        self.broken_files = set(broken_files)         # FilePath substrings answering 404
        self.archive_last_date = archive_last_date
        self.partial_day = partial_day                # archive day with a missing hour
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url))
        if "webapi.grid-india.in" in url:
            body = kwargs["json"]
            key = f"{body['_fileDate']}/{body['_month']}"
            return FakeResponse(payload={"retData": self.listing.get(key, [])})
        if "webcdn.grid-india.in" in url:
            if any(b in url for b in self.broken_files):
                return FakeResponse(status=404)
            return FakeResponse(content=b"x" * (fetch_psp_reports.MIN_VALID_FILE_BYTES + 10))
        if "open-meteo.com" in url:
            if self.fail_weather:
                return FakeResponse(status=500)
            return FakeResponse(payload=self._weather(url, kwargs["params"]))
        raise AssertionError(f"unexpected URL {url}")

    def weather_calls(self):
        return [c for c in self.calls if "open-meteo" in c[1]]

    def _weather(self, url, params):
        archive = "archive-api" in url
        if archive:
            first = dt.date.fromisoformat(params["start_date"])
            last = dt.date.fromisoformat(params["end_date"])
        else:
            last = NOW.date()
            first = last - dt.timedelta(days=params["past_days"])
        n_loc = len(str(params["latitude"]).split(","))
        hours = []
        day = first
        while day <= last:
            hours.extend(f"{day.isoformat()}T{h:02d}:00" for h in range(24))
            day += dt.timedelta(days=1)
        temp = self.archive_temp if archive else FORECAST_TEMP_C
        out = []
        for _ in range(n_loc):
            hourly = {
                "time": hours,
                "temperature_2m": [temp] * len(hours),
                "dewpoint_2m": [temp - 5.0] * len(hours),
                "apparent_temperature": [temp + 2.0] * len(hours),
                "windspeed_10m": [36.0] * len(hours),           # 10 m/s
                "winddirection_10m": [90.0] * len(hours),       # from the east -> u = -10, v ~ 0
                "shortwave_radiation": [500.0] * len(hours),
                "cloudcover": [50.0] * len(hours),
            }
            for i, t in enumerate(hours):
                d = dt.date.fromisoformat(t[:10])
                if archive and d > self.archive_last_date:
                    for var in list(hourly)[1:]:
                        hourly[var][i] = None
                if archive and self.partial_day and t == f"{self.partial_day}T05:00":
                    hourly["temperature_2m"][i] = None
            out.append({"hourly": hourly})
        return out if n_loc > 1 else out[0]


def psp_item(day, item_id, ext="xls"):
    stem = f"{day.strftime('%d.%m.%y')}_NLDC_PSP"
    return {"Id": item_id, "FilePath": f"files/grdw/{day.year}/{day.month:02d}/{stem}_{item_id % 1000}.{ext}",
            "Title_": stem, "Field2": (day + dt.timedelta(days=1)).isoformat()}


def listing_for(*items):
    return {"2026-27/11": list(items)}


def fake_parse(path):
    """Deterministic stand-in for the PSP parser; values encode the file's upload Id."""
    day = parse_date_from_filename(str(path))
    if day is None or day in UNPARSEABLE or Path(path).read_bytes().startswith(BAD_CONTENT):
        return None
    file_id = int(re.search(r"_PSP_(\d+)", Path(path).name).group(1))
    wide = {"Date": day, **{c: 100.0 + file_id / 1000 for c in STATE_COLS}, "Total": 4000.0 + file_id / 1000}
    long_rows = [{"date": day, "Region": "NR", "States": "S0", "Max.Demand Met during the day(MW)": 5.0,
                  "Shortage during maximum Demand(MW)": 0.0, "Energy Met (MU)": 100.0 + file_id / 1000}]
    gen = [{"date": day, "source": "Coal", "NR": 1.0, "WR": 2.0, "SR": 3.0, "ER": 4.0, "NER": 5.0, "All India": 15.0}]
    return {"date": day, "state_demand_wide": wide, "state_demand_long": long_rows, "generation_long": gen}


UNPARSEABLE: set = set()
BAD_CONTENT = b"BAD"


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([",".join(header)] + [",".join(map(str, r)) for r in rows]) + "\n")


def demand_row(day, value=90.0):
    return [day, *[value] * len(STATE_COLS), value * 40]


def weather_row(day, state=None, temp_k=290.0):
    vals = [temp_k] * (len(WEATHER_COLS) - 1)
    return [day, *vals] + ([state] if state else [])


@pytest.fixture
def paths(tmp_path):
    p = DataPaths.under(tmp_path / "root")
    old_days = ["2026-11-15", "2026-11-16"]
    write_csv(p.demand_main, DEMAND_HEADER, [demand_row("2026-03-01", 80.0)] + [demand_row(d) for d in old_days])
    write_csv(p.demand_mirror, DEMAND_HEADER, [demand_row(d) for d in old_days])
    sm1_header = ["Region", "States", "Max.Demand Met during the day(MW)", "Shortage during maximum Demand(MW)",
                  "Energy Met (MU)", "date"]
    write_csv(p.state_mu_1, sm1_header, [["NR", "S0", 5.0, 0.0, 90.0, "2026-11-16"], ["NR", "S1", 5.0, 0.0, 90.0, "2026-11-16"]])
    write_csv(p.state_mu_2, ["source", "NR", "WR", "SR", "ER", "NER", "All India", "date"],
              [["Coal", 1.0, 2.0, 3.0, 4.0, 5.0, 15.0, "2026-11-16"]])
    write_csv(p.long_state_mu2, ["date", "region", "source", "regional_generation_mu"],
              [["2026-11-16", "nr", "Coal", 1.0]])
    # Weather complete through 2026-11-18 for every state.
    zen_header = [*WEATHER_COLS, "state"]
    write_csv(p.zenodo, zen_header, [weather_row(d, s) for d in ("2026-11-17", "2026-11-18") for s in sorted(ALL_STATES)])
    for state in ALL_STATES:
        write_csv(p.state_weather_dir / STATE_FILE_MAP[state], WEATHER_COLS,
                  [weather_row(d) for d in ("2026-11-17", "2026-11-18")])
    return p


@pytest.fixture(autouse=True)
def fast_and_fake(monkeypatch):
    UNPARSEABLE.clear()
    for mod in (fetch_psp_reports, weather_update, daily_update):
        monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(daily_update, "parse_psp_xls", fake_parse)


def new_ingest():
    calls = []

    def ingest(target_date, catchup_days):
        calls.append((target_date, catchup_days))
        return {"reconciled_forecasts": 7}
    return ingest, calls


def reports_17_to_20(**kw):
    days = [dt.date(2026, 11, d) for d in (17, 18, 19, 20)]
    return FakeSession(listing=listing_for(*[psp_item(d, 49000 + i) for i, d in enumerate(days)]), **kw)


def run(paths, session, ingest=None, **kw):
    ingest = ingest or new_ingest()[0]
    return run_daily_update(end_date=kw.pop("end", END), catchup_days=kw.pop("catchup", 4), paths=paths,
                            session=session, ingest_fn=ingest, now=NOW, **kw)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dates_in(path, col="Date"):
    return pd.read_csv(path)[col].tolist()


# ---------------------------------------------------------------------------
# Demand
# ---------------------------------------------------------------------------
def test_adds_missing_dates_to_every_demand_file_keeping_schema(paths):
    headers = {p: Path(p).read_text().splitlines()[0] for p in (paths.demand_main, paths.demand_mirror, paths.state_mu_1, paths.state_mu_2, paths.long_state_mu2)}
    ingest, calls = new_ingest()
    summary = run(paths, reports_17_to_20(), ingest)

    assert summary["status"] == "ok", summary["errors"]
    assert summary["dates_added"] == ["2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20"]
    assert dates_in(paths.demand_main) == ["2026-03-01", "2026-11-15", "2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20"]
    assert dates_in(paths.demand_mirror)[-1] == "2026-11-20"
    assert "2026-11-20" in dates_in(paths.state_mu_1, "date")
    assert "2026-11-20" in dates_in(paths.state_mu_2, "date")
    assert "2026-11-20" in dates_in(paths.long_state_mu2, "date")
    for p, header in headers.items():
        assert Path(p).read_text().splitlines()[0] == header
    assert calls == [(dt.date(2026, 11, 20), 4)]
    assert summary["ledger_reconciled"] == 7
    assert summary["last_demand_date"] == "2026-11-20"
    assert paths.log_file.exists()


def test_second_run_is_a_no_op(paths):
    run(paths, reports_17_to_20())
    watched = [paths.demand_main, paths.demand_mirror, paths.state_mu_1, paths.state_mu_2,
               paths.long_state_mu2, paths.zenodo]
    before = {p: sha(p) for p in watched}
    ingest, calls = new_ingest()
    session = reports_17_to_20()
    summary = run(paths, session, ingest)

    assert summary["dates_added"] == [] and summary["dates_revised"] == []
    assert summary["dates_still_missing"] == []
    assert {p: sha(p) for p in watched} == before
    assert calls == []                                    # nothing new -> ledger untouched
    assert not [c for c in session.calls if "webcdn" in c[1]]   # nothing re-downloaded
    # Recent (provisional-window) weather is re-checked against the archive only; nothing changed.
    assert all("archive-api" in c[1] for c in session.weather_calls())
    assert summary["weather_dates_added"]["upgraded_to_archive"] == []


def test_missing_and_404_dates_are_reported_never_interpolated(paths):
    d17, d18 = dt.date(2026, 11, 17), dt.date(2026, 11, 18)
    session = FakeSession(listing=listing_for(psp_item(d17, 49001), psp_item(d18, 49002)),
                          broken_files=["18.11.26"])
    summary = run(paths, session)

    assert summary["dates_added"] == ["2026-11-17"]
    reasons = {m["date"]: m["reason"] for m in summary["dates_still_missing"]}
    assert "404" in reasons["2026-11-18"]
    assert reasons["2026-11-19"] == daily_update.REASON_NOT_PUBLISHED
    assert reasons["2026-11-20"] == daily_update.REASON_NOT_PUBLISHED
    stored = dates_in(paths.demand_main)
    assert not {"2026-11-18", "2026-11-19", "2026-11-20"} & set(stored)
    assert not {"2026-11-18", "2026-11-19", "2026-11-20"} & set(dates_in(paths.demand_mirror))


def test_old_date_without_listing_is_reported_as_not_listed(paths):
    summary = run(paths, FakeSession(listing=listing_for()), catchup=2)   # window 19..20 only
    assert all(m["reason"] == daily_update.REASON_NOT_PUBLISHED for m in summary["dates_still_missing"])
    summary = run(paths, FakeSession(listing=listing_for()), end=dt.date(2026, 11, 17), catchup=1)
    assert summary["dates_still_missing"] == [{"date": "2026-11-17", "reason": daily_update.REASON_NOT_LISTED}]


def test_unparseable_report_is_not_added(paths):
    UNPARSEABLE.add("2026-11-17")
    summary = run(paths, reports_17_to_20())
    assert "2026-11-17" not in summary["dates_added"]
    assert {"date": "2026-11-17", "reason": "report could not be parsed"} in summary["dates_still_missing"]
    assert "2026-11-17" not in dates_in(paths.demand_main)


def test_newer_revision_replaces_only_the_changed_row(paths):
    run(paths, reports_17_to_20())
    before = pd.read_csv(paths.demand_main).set_index("Date")
    # Grid-India publishes a revised report (higher Id) for 2026-11-18 only.
    items = [psp_item(dt.date(2026, 11, d), 49000 + i) for i, d in enumerate((17, 18, 19, 20))]
    items.append(psp_item(dt.date(2026, 11, 18), 49900))
    ingest, calls = new_ingest()
    summary = run(paths, FakeSession(listing=listing_for(*items)), ingest)

    assert summary["dates_added"] == []
    assert summary["dates_revised"] == ["2026-11-18"]
    after = pd.read_csv(paths.demand_main).set_index("Date")
    assert before.loc["2026-11-18", "S0"] == pytest.approx(149.001)
    assert after.loc["2026-11-18", "S0"] == pytest.approx(149.9)          # value of the revised report
    unchanged = [d for d in before.index if d != "2026-11-18"]
    pd.testing.assert_frame_equal(after.loc[unchanged], before.loc[unchanged])
    assert len(after) == len(before)                      # replaced, not duplicated
    assert calls == [(dt.date(2026, 11, 18), 1)]
    # ... and it settles: the next run revises nothing.
    again = run(paths, FakeSession(listing=listing_for(*items)))
    assert again["dates_revised"] == []


def test_stored_rows_before_first_psp_date_are_never_revised(paths):
    write_csv(paths.demand_main, DEMAND_HEADER, [demand_row("2024-03-01", 80.0)])
    snapshot = sha(paths.demand_main)
    session = FakeSession(listing={"2023-24/03": [psp_item(dt.date(2024, 3, 1), 99999)]})
    summary = run_daily_update(end_date=dt.date(2024, 3, 1), catchup_days=1, paths=paths, session=session,
                               ingest_fn=new_ingest()[0], now=NOW)
    assert summary["dates_revised"] == [] and sha(paths.demand_main) == snapshot
    assert not [c for c in session.calls if "webcdn" in c[1]]


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
def zenodo_rows(paths, day):
    df = pd.read_csv(paths.zenodo)
    return df[df["date"] == day]


def test_weather_fills_gap_up_to_end_for_every_state_and_file(paths):
    summary = run(paths, reports_17_to_20())

    assert summary["weather_dates_added"]["archive"] == ["2026-11-19", "2026-11-20"]
    assert summary["weather_dates_added"]["provisional"] == []
    assert summary["last_weather_date"] == "2026-11-20"
    for day in ("2026-11-19", "2026-11-20"):
        assert sorted(zenodo_rows(paths, day)["state"]) == sorted(ALL_STATES)
    kerala = pd.read_csv(paths.state_weather_dir / STATE_FILE_MAP["Kerala"])
    assert kerala["date"].tolist() == ["2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20"]
    assert list(kerala.columns) == WEATHER_COLS
    # Stored rows inside the provisional window were refreshed from the archive (no duplicates).
    assert zenodo_rows(paths, "2026-11-17")["2m_temperature_mean"].eq(ARCHIVE_TEMP_C + 273.15).all()
    assert len(zenodo_rows(paths, "2026-11-17")) == len(ALL_STATES)


def test_weather_unit_conversion_matches_era5_schema(paths):
    run(paths, reports_17_to_20())
    row = zenodo_rows(paths, "2026-11-20").iloc[0]
    assert row["2m_temperature_mean"] == pytest.approx(ARCHIVE_TEMP_C + 273.15)          # Kelvin
    assert row["2m_dewpoint_temperature_max"] == pytest.approx(ARCHIVE_TEMP_C - 5 + 273.15)
    assert row["utci_mean"] == pytest.approx(ARCHIVE_TEMP_C + 2 + 273.15)
    assert row["surface_solar_radiation_downwards_mean"] == pytest.approx(500.0 * 3600)  # J/m^2
    assert row["total_cloud_cover_mean"] == pytest.approx(0.5)                            # 0-1 fraction
    assert row["10m_u_component_of_wind_mean"] == pytest.approx(-10.0)                    # 36 km/h from the east
    assert row["10m_v_component_of_wind_mean"] == pytest.approx(0.0, abs=1e-9)


def test_recent_days_use_provisional_forecast_then_get_replaced_by_archive(paths):
    # Archive only reaches 2026-11-19, so 2026-11-20 falls back to the forecast API.
    first = run(paths, reports_17_to_20(archive_last_date=dt.date(2026, 11, 19)))
    assert first["weather_dates_added"]["archive"] == ["2026-11-19"]
    assert first["weather_dates_added"]["provisional"] == ["2026-11-20"]
    # Every date inside the provisional window is pending, archive-sourced or not.
    assert first["weather_provisional_pending"] == ["2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20"]
    prov = weather_update.load_provisional(paths.provisional)
    assert prov["2026-11-20"] == set(ALL_STATES)
    assert zenodo_rows(paths, "2026-11-20")["2m_temperature_mean"].eq(FORECAST_TEMP_C + 273.15).all()

    # Archive catches up: provisional rows are replaced in place (no duplicates) but stay
    # provisional while inside the window.
    second = run(paths, reports_17_to_20(archive_last_date=END))
    assert second["weather_dates_added"]["upgraded_to_archive"] == ["2026-11-20"]
    assert second["weather_provisional_pending"] == ["2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20"]
    rows = zenodo_rows(paths, "2026-11-20")
    assert len(rows) == len(ALL_STATES)
    assert rows["2m_temperature_mean"].eq(ARCHIVE_TEMP_C + 273.15).all()
    assert weather_update.load_provisional(paths.provisional)["2026-11-20"] == set(ALL_STATES)
    kerala = pd.read_csv(paths.state_weather_dir / STATE_FILE_MAP["Kerala"])
    assert kerala["date"].tolist().count("2026-11-20") == 1
    assert kerala.loc[kerala["date"] == "2026-11-20", "2m_temperature_mean"].iloc[0] == pytest.approx(ARCHIVE_TEMP_C + 273.15)


def test_provisional_rows_stay_until_archive_is_complete(paths):
    run(paths, reports_17_to_20(archive_last_date=dt.date(2026, 11, 19)))
    run(paths, reports_17_to_20(archive_last_date=dt.date(2026, 11, 19)))   # archive still behind
    assert weather_update.load_provisional(paths.provisional)["2026-11-20"] == set(ALL_STATES)
    assert zenodo_rows(paths, "2026-11-20")["2m_temperature_mean"].eq(FORECAST_TEMP_C + 273.15).all()


def test_day_with_a_missing_hour_is_not_accepted_from_archive(paths):
    summary = run(paths, reports_17_to_20(partial_day="2026-11-19"))
    assert "2026-11-19" not in summary["weather_dates_added"]["archive"]
    assert "2026-11-19" in summary["weather_dates_added"]["provisional"]   # complete from the forecast API instead


def test_weather_request_batches_locations(paths):
    session = reports_17_to_20()
    run(paths, session)
    archive = [c for c in session.weather_calls() if "archive-api" in c[1]]
    assert len(archive) == -(-len(ALL_STATES) // weather_update.LOCATIONS_PER_REQUEST)


# ---------------------------------------------------------------------------
# Dry run, lock, text preservation
# ---------------------------------------------------------------------------
def test_missing_data_directory_is_an_error_not_a_silent_success(tmp_path):
    empty = DataPaths.under(tmp_path / "nothing")
    summary = run(empty, reports_17_to_20())
    assert summary["status"] == "error"
    assert summary["dates_added"] == []
    assert {"stage": "demand", "code": "data_missing"} in [
        {k: e[k] for k in ("stage", "code")} for e in summary["errors"]]


def test_server_starts_and_stops_the_scheduler_only_when_enabled(monkeypatch):
    import server
    assert server._start_update_scheduler() is None                        # opt-in: off by default
    monkeypatch.setenv("LOADLY_AUTO_UPDATE", "1")
    monkeypatch.setenv("LOADLY_UPDATE_STARTUP_DELAY_S", "3600")
    scheduler = server._start_update_scheduler()
    assert scheduler is not None and scheduler._thread.is_alive()
    scheduler.stop()
    assert not scheduler._thread.is_alive()


def session_calls(summary):
    return summary["_session"].calls


def test_dry_run_writes_nothing(paths):
    watched = [paths.demand_main, paths.zenodo, paths.state_mu_1]
    before = {p: sha(p) for p in watched}
    session = reports_17_to_20()
    summary = run(paths, session, dry_run=True)
    summary["_session"] = session
    assert summary["dry_run"] is True
    assert summary["dates_would_fetch"] == ["2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20"]
    assert {p: sha(p) for p in watched} == before
    assert not paths.status_file.exists() and not paths.lock_file.exists() and not paths.psp_dir.exists()
    assert not paths.log_file.exists()                    # dry run logs to the console only
    assert not [c for c in session_calls(summary) if "webcdn" in c[1] or "open-meteo" in c[1]]


def test_concurrent_run_is_skipped_while_lock_is_held(paths):
    lock = FileLock(paths.lock_file, daily_update.LOCK_STALE_SECONDS)
    assert lock.acquire()
    before = sha(paths.demand_main)
    summary = run(paths, reports_17_to_20())
    assert summary["status"] == "skipped"
    assert sha(paths.demand_main) == before
    assert paths.lock_file.exists()                       # the holder's lock is not released by the loser
    lock.release()
    assert run(paths, reports_17_to_20())["status"] == "ok"
    assert not paths.lock_file.exists()                   # released after a normal run


def test_stale_lock_is_reclaimed(tmp_path):
    path = tmp_path / "x.lock"
    path.write_text("999 old")
    old = time.time() - 10_000
    os.utime(path, (old, old))
    lock = FileLock(path, stale_seconds=3600)
    assert lock.acquire()
    assert not FileLock(path, stale_seconds=3600).acquire()
    lock.release()


def test_lock_released_when_a_step_raises(paths, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("listing exploded")
    monkeypatch.setattr(daily_update, "collect_items", boom)
    before = sha(paths.demand_main)
    summary = run(paths, reports_17_to_20())
    assert summary["status"] == "error"
    assert [(e["stage"], e["code"]) for e in summary["errors"]] == [("demand", "internal_error")]
    assert "listing exploded" not in json.dumps(summary)                       # detail: log only
    assert "listing exploded" in paths.log_file.read_text(encoding="utf-8")
    assert not paths.lock_file.exists()
    assert sha(paths.demand_main) == before
    assert summary["weather_dates_added"]["archive"]      # weather still ran


def test_upsert_preserves_existing_text_and_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "t.csv"
    path.write_text('date,v,name\n2026-01-01,298.77493,"RES (Wind, Solar)"\n2026-01-03,289.04999999999995,b\n')
    added, replaced = upsert_csv(path, pd.DataFrame({"date": ["2026-01-02"], "v": [1.5], "name": ["a"]}), ["date"])
    assert (added, replaced) == (1, 0)
    lines = path.read_text().splitlines()
    assert lines[1] == '2026-01-01,298.77493,"RES (Wind, Solar)"'
    assert lines[2] == "2026-01-02,1.5,a" and lines[3] == "2026-01-03,289.04999999999995,b"

    snapshot = path.read_bytes()
    monkeypatch.setattr(os, "replace", lambda *_a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        upsert_csv(path, pd.DataFrame({"date": ["2026-01-04"], "v": [2.0], "name": ["c"]}), ["date"])
    assert path.read_bytes() == snapshot
    assert not list(tmp_path.glob("*.tmp*"))


def test_upsert_keeps_the_files_crlf_line_endings(tmp_path):
    path = tmp_path / "t.csv"
    path.write_bytes(b"date,v\r\n2026-01-01,1.0\r\n")
    upsert_csv(path, pd.DataFrame({"date": ["2026-01-02"], "v": [2.0]}), ["date"])
    assert path.read_bytes() == b"date,v\r\n2026-01-01,1.0\r\n2026-01-02,2.0\r\n"


def test_existing_key_is_ignored_unless_replace_requested(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("date,v\n2026-01-01,1.0\n")
    new = pd.DataFrame({"date": ["2026-01-01"], "v": [9.0]})
    assert upsert_csv(path, new, ["date"]) == (0, 0)
    assert upsert_csv(path, new, ["date"], replace_keys={("2026-01-01",)}) == (1, 1)
    assert path.read_text() == "date,v\n2026-01-01,9.0\n"


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("now,expected", [
    (dt.datetime(2026, 11, 21, 9, 0, tzinfo=IST), dt.datetime(2026, 11, 21, 13, 0, tzinfo=IST)),
    (dt.datetime(2026, 11, 21, 12, 59, 59, tzinfo=IST), dt.datetime(2026, 11, 21, 13, 0, tzinfo=IST)),
    (dt.datetime(2026, 11, 21, 13, 0, tzinfo=IST), dt.datetime(2026, 11, 22, 13, 0, tzinfo=IST)),
    (dt.datetime(2026, 11, 21, 18, 30, tzinfo=IST), dt.datetime(2026, 11, 22, 13, 0, tzinfo=IST)),
    (dt.datetime(2026, 12, 31, 23, 59, tzinfo=IST), dt.datetime(2027, 1, 1, 13, 0, tzinfo=IST)),
])
def test_next_run_is_1pm_ist(now, expected):
    assert update_scheduler.next_scheduled_run(now, "13:00") == expected


def test_next_run_converts_other_timezones_to_ist():
    utc_now = dt.datetime(2026, 11, 21, 7, 29, tzinfo=dt.timezone.utc)      # 12:59 IST
    assert update_scheduler.next_scheduled_run(utc_now).isoformat() == "2026-11-21T13:00:00+05:30"
    utc_now = dt.datetime(2026, 11, 21, 7, 31, tzinfo=dt.timezone.utc)      # 13:01 IST
    assert update_scheduler.next_scheduled_run(utc_now).isoformat() == "2026-11-22T13:00:00+05:30"


def test_scheduler_settings_from_env(monkeypatch):
    assert update_scheduler.scheduler_settings() == (False, "13:00")
    monkeypatch.setenv("LOADLY_AUTO_UPDATE", "1")
    monkeypatch.setenv("LOADLY_UPDATE_TIME", "06:45")
    assert update_scheduler.scheduler_settings() == (True, "06:45")
    monkeypatch.setenv("LOADLY_UPDATE_TIME", "25:99")
    assert update_scheduler.scheduler_settings() == (True, "13:00")
    monkeypatch.setenv("LOADLY_UPDATE_TIME", "noon")
    assert update_scheduler.scheduler_settings() == (True, "13:00")


def test_scheduler_runs_catchup_survives_errors_and_stops_cleanly():
    ran = threading.Event()
    calls = []

    def flaky():
        calls.append(1)
        ran.set()
        raise RuntimeError("network down")      # must not kill the thread

    scheduler = update_scheduler.DailyUpdateScheduler(flaky, "13:00", startup_delay_s=0)
    scheduler.start()
    assert ran.wait(5)
    assert scheduler._thread.is_alive()
    scheduler.stop()
    assert not scheduler._thread.is_alive()
    assert len(calls) == 1                      # next run is a day away


def test_scheduler_stops_during_startup_delay_without_running():
    calls = []
    scheduler = update_scheduler.DailyUpdateScheduler(lambda: calls.append(1), startup_delay_s=60)
    scheduler.start()
    scheduler.stop()
    assert not scheduler._thread.is_alive() and calls == []


def test_scheduler_fires_at_due_time(monkeypatch):
    monkeypatch.setattr(update_scheduler, "CACHE_CHECK_INTERVAL_S", 0.01)
    clock = {"now": dt.datetime(2026, 11, 21, 12, 59, 50, tzinfo=IST)}
    runs = []
    fired = threading.Event()

    def run_fn():
        runs.append(clock["now"])
        if len(runs) == 2:
            fired.set()

    def tick():
        clock["now"] += dt.timedelta(seconds=1)
        return clock["now"]

    scheduler = update_scheduler.DailyUpdateScheduler(
        run_fn, "13:00", startup_delay_s=0, cache_refresh_fn=lambda: None, clock=tick)
    scheduler.start()
    assert fired.wait(10)
    scheduler.stop()
    assert runs[1] >= dt.datetime(2026, 11, 21, 13, 0, tzinfo=IST)


# ---------------------------------------------------------------------------
# Status endpoint and cache clearing
# ---------------------------------------------------------------------------
client = TestClient(app)


def test_status_endpoint_is_public_and_reports_dates(paths, monkeypatch):
    monkeypatch.setattr(daily_update, "default_paths", lambda: paths)
    run(paths, reports_17_to_20())
    monkeypatch.setenv("LOADLY_AUTO_UPDATE", "1")
    res = client.get("/api/data/update-status")                    # no admin key
    assert res.status_code == 200
    body = res.json()
    assert body["running"] is False and body["last_skipped_at"] is None
    assert body["last_demand_date"] == "2026-11-20"
    assert body["last_weather_date"] == "2026-11-20"
    assert body["last_run_status"] == "ok"
    assert body["last_run_summary"]["dates_added"][-1] == "2026-11-20"
    assert body["scheduler_enabled"] is True and body["next_scheduled_run"].endswith("+05:30")


def test_status_endpoint_before_any_run(paths, monkeypatch):
    monkeypatch.setattr(daily_update, "default_paths", lambda: paths)
    body = client.get("/api/data/update-status").json()
    assert body["last_run_at"] is None and body["scheduler_enabled"] is False
    assert body["next_scheduled_run"] is None
    assert body["last_demand_date"] == "2026-11-16"


def test_caches_cleared_after_update_and_on_external_change(paths, monkeypatch):
    cleared = []
    import data_loading
    monkeypatch.setattr(data_loading, "clear_data_caches", lambda: cleared.append(1))
    run(paths, reports_17_to_20())
    assert cleared == [1]
    assert daily_update.refresh_caches_if_data_changed(paths) is False     # already in sync
    paths.demand_main.write_text(paths.demand_main.read_text() + ",".join(map(str, demand_row("2026-11-21"))) + "\n")
    assert daily_update.refresh_caches_if_data_changed(paths) is True      # another worker changed the file
    assert cleared == [1, 1]


def test_data_loading_cache_helper_empties_lru_caches():
    import data_loading
    data_loading.get_state_historical_averages("Kerala")      # read-only; may legitimately return {}
    data_loading.clear_data_caches()
    assert data_loading.get_state_historical_averages.cache_info().currsize == 0
