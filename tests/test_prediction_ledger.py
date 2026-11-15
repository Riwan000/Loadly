"""
Unit & Integration Tests for Prediction Ledger and Forecast Verification Engine.
"""

import pytest
import sqlite3
from pathlib import Path
from fastapi.testclient import TestClient

from server import app
from utils.prediction_ledger import (
    init_ledger_db,
    record_forecast_batch,
    record_actuals,
    get_verification_metrics,
    query_ledger
)


@pytest.fixture
def temp_db_path(tmp_path):
    """Provides an isolated SQLite database path for testing."""
    return tmp_path / "test_prediction_ledger.db"


def test_init_ledger_db(temp_db_path):
    """Verify that ledger DB initialization creates table and indexes."""
    init_ledger_db(temp_db_path)
    assert temp_db_path.exists()

    conn = sqlite3.connect(str(temp_db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='forecast_ledger';")
    assert cursor.fetchone() is not None

    cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
    index_names = [row[0] for row in cursor.fetchall()]
    assert "idx_ledger_state_target" in index_names
    assert "idx_ledger_forecast_id" in index_names
    conn.close()


def test_record_forecast_batch(temp_db_path):
    """Verify recording a batch of timeline forecasts into the ledger."""
    timeline = [
        {
            "date": "2026-11-20",
            "forecasted_demand_mu": 350.5,
            "lower_bound_mu": 330.0,
            "upper_bound_mu": 371.0,
            "temperature_mean_c": 28.5,
            "cdd": 4.5,
            "extreme_heat": 0
        },
        {
            "date": "2026-11-21",
            "forecasted_demand_mu": 360.0,
            "lower_bound_mu": 340.0,
            "upper_bound_mu": 380.0,
            "temperature_mean_c": 33.2,
            "cdd": 9.2,
            "extreme_heat": 1
        }
    ]

    forecast_id = record_forecast_batch(
        timeline,
        state="Maharashtra",
        model_name="Reduced_LR",
        db_path=temp_db_path,
        auto_reconcile=False
    )
    assert forecast_id is not None
    assert len(forecast_id) > 10

    records = query_ledger(state="Maharashtra", db_path=temp_db_path)
    assert len(records) == 2
    assert records[0]["state"] == "Maharashtra"
    assert records[0]["forecast_id"] == forecast_id


def test_record_actuals_and_reconciliation(temp_db_path):
    """Verify that recording actual ground-truth demand reconciles errors and CI flags."""
    timeline = [
        {
            "date": "2026-11-20",
            "forecasted_demand_mu": 100.0,
            "lower_bound_mu": 90.0,
            "upper_bound_mu": 110.0,
            "temperature_mean_c": 25.0,
            "cdd": 1.0,
            "extreme_heat": 0
        },
        {
            "date": "2026-11-21",
            "forecasted_demand_mu": 200.0,
            "lower_bound_mu": 190.0,
            "upper_bound_mu": 210.0,
            "temperature_mean_c": 26.0,
            "cdd": 2.0,
            "extreme_heat": 0
        }
    ]
    record_forecast_batch(timeline, state="Kerala", model_name="Reduced_LR", db_path=temp_db_path, auto_reconcile=False)

    # Ingest actuals:
    # 2026-11-20 actual is 105.0 (within CI: 90-110, error = +5.0)
    # 2026-11-21 actual is 220.0 (outside CI: 190-210, error = +20.0)
    actuals = [
        {"state": "Kerala", "target_date": "2026-11-20", "actual_demand_mu": 105.0},
        {"state": "Kerala", "target_date": "2026-11-21", "actual_demand_mu": 220.0}
    ]
    reconciled_count = record_actuals(actuals, db_path=temp_db_path)
    assert reconciled_count == 2

    records = query_ledger(state="Kerala", reconciled_only=True, db_path=temp_db_path)
    assert len(records) == 2

    rec1 = next(r for r in records if r["target_date"] == "2026-11-20")
    assert rec1["actual_demand_mu"] == 105.0
    assert rec1["error_mu"] == 5.0
    assert rec1["abs_error_mu"] == 5.0
    assert rec1["within_ci"] is True
    assert pytest.approx(rec1["pct_error"], 0.01) == 4.76  # 5 / 105 * 100

    rec2 = next(r for r in records if r["target_date"] == "2026-11-21")
    assert rec2["actual_demand_mu"] == 220.0
    assert rec2["error_mu"] == 20.0
    assert rec2["within_ci"] is False


def test_verification_metrics_calculation(temp_db_path):
    """Verify accurate computation of MAE, RMSE, MAPE, CI coverage rate, and Bias."""
    timeline = [
        {"date": "2026-11-20", "forecasted_demand_mu": 100.0, "lower_bound_mu": 95.0, "upper_bound_mu": 105.0},
        {"date": "2026-11-21", "forecasted_demand_mu": 100.0, "lower_bound_mu": 95.0, "upper_bound_mu": 105.0}
    ]
    record_forecast_batch(timeline, state="Punjab", db_path=temp_db_path, auto_reconcile=False)

    # Actuals: 102.0 (error=+2, within CI), 96.0 (error=-4, within CI)
    # MAE = (|2| + |-4|) / 2 = 3.0
    # MSE = (4 + 16) / 2 = 10.0 -> RMSE = sqrt(10) ≈ 3.16
    # Bias = (+2 - 4) / 2 = -1.0
    # CI Coverage = 2/2 = 100%
    actuals = [
        {"state": "Punjab", "target_date": "2026-11-20", "actual_demand_mu": 102.0},
        {"state": "Punjab", "target_date": "2026-11-21", "actual_demand_mu": 96.0}
    ]
    record_actuals(actuals, db_path=temp_db_path)

    metrics = get_verification_metrics(state="Punjab", days_window=30, db_path=temp_db_path)
    assert metrics["reconciled_count"] == 2
    assert metrics["mae_mu"] == 3.0
    assert pytest.approx(metrics["rmse_mu"], 0.05) == 3.16
    assert metrics["mean_bias_mu"] == -1.0
    assert metrics["ci_coverage_pct"] == 100.0


def test_api_ledger_endpoints():
    """Verify the FastAPI endpoints for ledger records, metrics, and actuals ingestion."""
    client = TestClient(app)

    # 1. Trigger a forecast endpoint - should automatically log to the ledger
    forecast_payload = {
        "state": "Kerala",
        "horizon_days": 7,
        "use_weather_api": False
    }
    f_res = client.post("/api/forecast", json=forecast_payload)
    assert f_res.status_code == 200

    # 2. Query ledger records
    records_res = client.get("/api/ledger/records?state=Kerala&limit=10")
    assert records_res.status_code == 200
    data = records_res.json()
    assert "records" in data
    assert data["total_count"] >= 7
    first_rec = data["records"][0]
    assert first_rec["state"] == "Kerala"
    assert "forecast_id" in first_rec

    # 3. Ingest an actual demand
    target_dt = first_rec["target_date"]
    actuals_payload = {
        "records": [
            {
                "state": "Kerala",
                "target_date": target_dt,
                "actual_demand_mu": 95.5
            }
        ]
    }
    ingest_res = client.post("/api/ledger/actuals", json=actuals_payload)
    assert ingest_res.status_code == 200
    assert ingest_res.json()["status"] == "success"
    assert ingest_res.json()["reconciled_count"] >= 1

    # 4. Query verification metrics
    metrics_res = client.get("/api/ledger/metrics?state=Kerala")
    assert metrics_res.status_code == 200
    m_data = metrics_res.json()
    assert m_data["reconciled_count"] >= 1
    assert m_data["mae_mu"] is not None
