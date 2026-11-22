"""
Tests for the Continuous Data Ingestion & Forecast Verification Worker.

Actuals come only from the local historical demand CSV. Dates without real data
must yield nothing: no fabricated values, no ledger reconciliation, no cache write.
"""

from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from server import app
from config import ALL_STATES
from utils import ingestion_worker
from utils.ingestion_worker import (
    normalize_state_name,
    fetch_actual_grid_demand,
    run_ingestion_pipeline,
)
from utils.prediction_ledger import record_forecast_batch, query_ledger

REAL_DATE = date(2026, 8, 20)
MISSING_DATE = date(2026, 8, 21)
# Real CSV rows use "Delhi" and "Andaman and Nicobar Islands" and lack Lakshadweep.
CSV_COLUMN_FOR = {"NCT of Delhi": "Delhi", "Andaman and Nicobar": "Andaman and Nicobar Islands"}
ABSENT_STATES = {"Lakshadweep"}


def _csv_row(day: date, **extra) -> dict:
    row = {"Date": day.strftime("%Y-%m-%d")}
    for state in ALL_STATES:
        if state not in ABSENT_STATES:
            row[CSV_COLUMN_FOR.get(state, state)] = 100.0
    row["Maharashtra"] = 400.0
    row.update(extra)
    return row


@pytest.fixture
def demand_csv(tmp_path, monkeypatch):
    """Temp demand CSV with one real day (with national Total) at REAL_DATE."""
    path = tmp_path / "daily_energy_met_MU.csv"
    pd.DataFrame([_csv_row(REAL_DATE, Total=5000.0)]).to_csv(path, index=False)
    monkeypatch.setattr(ingestion_worker, "DEMAND_CACHE_PATH", path)
    return path


def test_normalize_state_name():
    """Verify that colloquial DISCOM / state synonyms map to canonical standard."""
    assert normalize_state_name("Delhi") == "NCT of Delhi"
    assert normalize_state_name("J&K") == "Jammu and Kashmir"
    assert normalize_state_name("Maharashtra") == "Maharashtra"
    assert normalize_state_name("Pondy") == "Puducherry"


def test_fetch_actual_grid_demand_reads_real_csv_row(demand_csv):
    actuals = fetch_actual_grid_demand(REAL_DATE)
    by_state = {r["state"]: r for r in actuals}

    assert by_state["Maharashtra"]["actual_demand_mu"] == 400.0
    assert by_state["NCT of Delhi"]["actual_demand_mu"] == 100.0  # aliased "Delhi" column
    assert "Lakshadweep" not in by_state
    assert all(r["source"] in ("historical_csv", "national_total_csv") for r in actuals)
    assert by_state["All-India"]["actual_demand_mu"] == 5000.0  # CSV Total, not a partial sum


def test_all_india_omitted_when_csv_has_no_total(tmp_path, monkeypatch):
    path = tmp_path / "d.csv"
    pd.DataFrame([_csv_row(REAL_DATE)]).to_csv(path, index=False)
    monkeypatch.setattr(ingestion_worker, "DEMAND_CACHE_PATH", path)

    actuals = fetch_actual_grid_demand(REAL_DATE)
    assert len(actuals) == len(ALL_STATES) - len(ABSENT_STATES)
    assert all(r["state"] != "All-India" for r in actuals)


def test_missing_date_returns_empty_and_warns(demand_csv, caplog):
    with caplog.at_level("WARNING"):
        assert fetch_actual_grid_demand(MISSING_DATE) == []
    assert "2026-08-21" in caplog.text


def test_missing_csv_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestion_worker, "DEMAND_CACHE_PATH", tmp_path / "nope.csv")
    assert fetch_actual_grid_demand(REAL_DATE) == []


def test_sparse_row_returns_empty(tmp_path, monkeypatch):
    path = tmp_path / "d.csv"
    pd.DataFrame([{"Date": "2026-08-20", "Maharashtra": 400.0, "Kerala": 90.0}]).to_csv(path, index=False)
    monkeypatch.setattr(ingestion_worker, "DEMAND_CACHE_PATH", path)
    assert fetch_actual_grid_demand(REAL_DATE) == []


def test_legacy_synthetic_row_is_not_served_as_actual(tmp_path, monkeypatch):
    """Rows the old worker wrote (marked by an All-India value) are fabricated."""
    path = tmp_path / "d.csv"
    pd.DataFrame([_csv_row(REAL_DATE, **{"All-India": 5400.0})]).to_csv(path, index=False)
    monkeypatch.setattr(ingestion_worker, "DEMAND_CACHE_PATH", path)
    assert fetch_actual_grid_demand(REAL_DATE) == []


def _log_pending_forecast(day: date) -> None:
    timeline = [{
        "date": day.strftime("%Y-%m-%d"),
        "forecasted_demand_mu": 390.0, "lower_bound_mu": 370.0, "upper_bound_mu": 410.0,
        "temperature_mean_c": 30.0, "cdd": 5.0, "extreme_heat": 0,
    }]
    assert record_forecast_batch(timeline, state="Maharashtra", auto_reconcile=False) is not None


def test_pipeline_reconciles_real_date_and_reports_missing(demand_csv):
    _log_pending_forecast(REAL_DATE)
    _log_pending_forecast(MISSING_DATE)
    before = demand_csv.read_text()

    summary = run_ingestion_pipeline(target_date=MISSING_DATE, catchup_days=2)

    assert summary["status"] == "success"
    assert summary["dates_missing"] == [MISSING_DATE.strftime("%Y-%m-%d")]
    assert summary["reconciled_forecasts"] >= 1
    assert summary["total_actuals_ingested"] == len(ALL_STATES) - len(ABSENT_STATES) + 1
    assert demand_csv.read_text() == before  # nothing cached/written

    rows = {r["target_date"]: r for r in query_ledger(state="Maharashtra", limit=10)}
    real, missing = rows[REAL_DATE.strftime("%Y-%m-%d")], rows[MISSING_DATE.strftime("%Y-%m-%d")]
    assert real["actual_demand_mu"] == 400.0
    assert real["error_mu"] is not None and real["within_ci"] is not None
    assert missing["actual_demand_mu"] is None


def test_pipeline_with_no_data_records_nothing(demand_csv):
    _log_pending_forecast(MISSING_DATE)
    summary = run_ingestion_pipeline(target_date=MISSING_DATE, catchup_days=1)

    assert summary["dates_missing"] == [MISSING_DATE.strftime("%Y-%m-%d")]
    assert summary["total_actuals_ingested"] == 0
    assert summary["reconciled_forecasts"] == 0
    rows = query_ledger(state="Maharashtra", limit=5)
    assert all(r["actual_demand_mu"] is None for r in rows)


def test_dead_update_demand_cache_is_gone():
    assert not hasattr(ingestion_worker, "update_demand_cache")


def test_api_ingestion_run_endpoint_reports_missing(demand_csv, admin_headers):
    client = TestClient(app)
    res = client.post(
        "/api/ingestion/run",
        json={"target_date": MISSING_DATE.strftime("%Y-%m-%d"), "catchup_days": 1},
        headers=admin_headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["dates_missing"] == [MISSING_DATE.strftime("%Y-%m-%d")]
    assert data["total_actuals_ingested"] == 0
    assert "verification_metrics" in data
