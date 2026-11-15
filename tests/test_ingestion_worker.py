"""
Unit & Integration Tests for Continuous Data Ingestion & Forecast Verification Worker.
"""

import pytest
from datetime import date, timedelta
from fastapi.testclient import TestClient

from server import app
from utils.ingestion_worker import (
    normalize_state_name,
    fetch_realized_weather,
    fetch_actual_grid_demand,
    update_demand_cache,
    run_ingestion_pipeline
)
from utils.prediction_ledger import record_forecast_batch, query_ledger


def test_normalize_state_name():
    """Verify that colloquial DISCOM / state synonyms map to canonical standard."""
    assert normalize_state_name("Delhi") == "NCT of Delhi"
    assert normalize_state_name("J&K") == "Jammu and Kashmir"
    assert normalize_state_name("Maharashtra") == "Maharashtra"
    assert normalize_state_name("Pondy") == "Puducherry"


def test_fetch_realized_weather():
    """Verify that realized weather returns mean, max, and CDD values."""
    target_dt = date.today() - timedelta(days=2)
    weather = fetch_realized_weather("Maharashtra", target_dt)
    assert weather is not None
    assert "temperature_mean_c" in weather
    assert "cdd" in weather
    assert weather["temperature_mean_c"] > -10.0


def test_fetch_actual_grid_demand():
    """Verify that grid demand retrieval returns records for states with positive MU."""
    target_dt = date.today() - timedelta(days=2)
    actuals = fetch_actual_grid_demand(target_dt)
    assert len(actuals) >= 30
    
    first = actuals[0]
    assert "state" in first
    assert "target_date" in first
    assert "actual_demand_mu" in first
    assert first["actual_demand_mu"] > 0.0


def test_end_to_end_ingestion_reconciliation(tmp_path):
    """
    Verify full closed-loop:
    1. Log a forecast for a target date
    2. Run ingestion pipeline for that target date
    3. Verify that the forecast record in the ledger flips to reconciled with calculated error!
    """
    target_dt = date.today() - timedelta(days=1)
    target_dt_str = target_dt.strftime("%Y-%m-%d")

    # 1. Log a synthetic pending forecast for Maharashtra
    timeline = [
        {
            "date": target_dt_str,
            "forecasted_demand_mu": 400.0,
            "lower_bound_mu": 380.0,
            "upper_bound_mu": 420.0,
            "temperature_mean_c": 30.0,
            "cdd": 5.0,
            "extreme_heat": 0
        }
    ]
    forecast_id = record_forecast_batch(timeline, state="Maharashtra", auto_reconcile=False)
    assert forecast_id is not None

    # Verify initial status is unreconciled
    unreconciled = query_ledger(state="Maharashtra", limit=5)
    matching = [r for r in unreconciled if r["target_date"] == target_dt_str]
    assert len(matching) >= 1
    assert matching[0]["actual_demand_mu"] is None

    # 2. Run ingestion pipeline for target_dt
    summary = run_ingestion_pipeline(target_date=target_dt, catchup_days=1)
    assert summary["status"] == "success"
    assert summary["reconciled_forecasts"] >= 1

    # 3. Verify ledger is now reconciled
    reconciled = query_ledger(state="Maharashtra", limit=5)
    matching_after = [r for r in reconciled if r["target_date"] == target_dt_str]
    assert len(matching_after) >= 1
    rec = matching_after[0]
    assert rec["actual_demand_mu"] is not None
    assert rec["error_mu"] is not None
    assert rec["pct_error"] is not None
    assert rec["within_ci"] is not None


def test_api_ingestion_run_endpoint():
    """Verify that POST /api/ingestion/run triggers ingestion through REST API."""
    client = TestClient(app)
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    res = client.post("/api/ingestion/run", json={"target_date": yesterday, "catchup_days": 1})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "dates_processed" in data
    assert data["total_actuals_ingested"] >= 30
    assert "verification_metrics" in data
