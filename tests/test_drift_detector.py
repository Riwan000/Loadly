"""
Unit and integration tests for Model Drift Detection & Adaptive Retraining Engine.
"""

import os
import sqlite3
import tempfile
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import app
from utils.drift_detector import calculate_psi, compute_drift_metrics
from utils.model_retrainer import (
    init_retraining_table,
    get_retraining_history,
    retrain_model,
    rollback_model,
    MODEL_PATH,
    MODEL_PREV_PATH
)
from utils.prediction_ledger import init_ledger_db, get_db_connection


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_path = Path(path)
    init_ledger_db(db_path)
    init_retraining_table(db_path)
    yield db_path
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass


def test_psi_identical_distribution():
    np.random.seed(42)
    dist1 = np.random.normal(loc=30.0, scale=5.0, size=500)
    dist2 = np.random.normal(loc=30.0, scale=5.0, size=500)
    psi = calculate_psi(dist1, dist2)
    assert psi < 0.10, f"Expected PSI < 0.10 for identical distributions, got {psi}"


def test_psi_shifted_distribution():
    np.random.seed(42)
    dist1 = np.random.normal(loc=30.0, scale=3.0, size=500)
    dist2 = np.random.normal(loc=45.0, scale=3.0, size=500)
    psi = calculate_psi(dist1, dist2)
    assert psi > 0.25, f"Expected severe PSI >= 0.25 for shifted distributions, got {psi}"


def test_compute_drift_metrics_empty(temp_db):
    metrics = compute_drift_metrics(state="Maharashtra", db_path=temp_db)
    assert metrics["status"] == "INSUFFICIENT_DATA"
    assert metrics["sample_count"] == 0
    assert metrics["recommended_action"] == "INGEST_ACTUALS"


def test_compute_drift_metrics_healthy(temp_db):
    conn = get_db_connection(temp_db)
    # Insert 10 well-calibrated reconciled records (MAPE ~1.5%, low bias, in-band)
    with conn:
        for i in range(10):
            conn.execute("""
                INSERT INTO forecast_ledger (
                    forecast_id, target_date, state, horizon_step, model_name,
                    predicted_demand_mu, lower_bound_mu, upper_bound_mu,
                    temperature_mean_c, cdd, actual_demand_mu, error_mu, abs_error_mu, pct_error, within_ci
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"fc-{i}", f"2026-11-{10+i:02d}", "Maharashtra", 1, "Reduced_LR",
                500.0, 480.0, 520.0,
                30.0, 5.0, 502.0, 2.0, 2.0, 0.4, 1
            ))
    conn.close()

    metrics = compute_drift_metrics(state="Maharashtra", db_path=temp_db)
    assert metrics["status"] == "HEALTHY"
    assert metrics["sample_count"] == 10
    assert metrics["metrics"]["mape_pct"] < 3.5
    assert metrics["metrics"]["ci_coverage_pct"] == 100.0
    assert metrics["recommended_action"] == "NONE"


def test_compute_drift_metrics_critical_drift(temp_db):
    conn = get_db_connection(temp_db)
    # Insert 10 records with large systematic under-prediction error (+50 MU error on 500 MU = 10% MAPE)
    with conn:
        for i in range(10):
            conn.execute("""
                INSERT INTO forecast_ledger (
                    forecast_id, target_date, state, horizon_step, model_name,
                    predicted_demand_mu, lower_bound_mu, upper_bound_mu,
                    temperature_mean_c, cdd, actual_demand_mu, error_mu, abs_error_mu, pct_error, within_ci
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"fc-{i}", f"2026-11-{10+i:02d}", "Maharashtra", 1, "Reduced_LR",
                500.0, 480.0, 520.0,
                30.0, 5.0, 560.0, 60.0, 60.0, 10.7, 0
            ))
    conn.close()

    metrics = compute_drift_metrics(state="Maharashtra", db_path=temp_db)
    assert metrics["status"] == "CRITICAL"
    assert metrics["sample_count"] == 10
    assert metrics["metrics"]["mape_pct"] > 5.0
    assert metrics["metrics"]["ci_coverage_pct"] == 0.0
    assert metrics["recommended_action"] == "RETRAIN_RECOMMENDED"


def test_retraining_and_rollback_flow(temp_db):
    # Test retraining execution
    result = retrain_model(trigger_reason="pytest_test", force=False, db_path=temp_db)
    assert "decision" in result
    assert result["decision"] in ("PROMOTED", "REJECTED")
    assert "champion" in result
    assert "challenger" in result
    assert result["challenger"]["r2"] is not None

    # Check history logged
    history = get_retraining_history(limit=5, db_path=temp_db)
    assert len(history) >= 1
    assert history[0]["trigger_reason"] == "pytest_test"

    # Test rollback
    if MODEL_PREV_PATH.exists():
        rb = rollback_model(db_path=temp_db)
        assert rb["status"] == "SUCCESS"
        history_after_rb = get_retraining_history(limit=5, db_path=temp_db)
        assert history_after_rb[0]["decision"] == "ROLLED_BACK"


def test_mlops_api_endpoints():
    client = TestClient(app)

    # 1. Test Drift Status Endpoint
    res_drift = client.get("/api/mlops/drift-status?state=All-India&window_days=14")
    assert res_drift.status_code == 200
    data_drift = res_drift.json()
    assert "status" in data_drift
    assert "metrics" in data_drift
    assert "diagnostics" in data_drift

    # 2. Test Retraining History Endpoint
    res_hist = client.get("/api/mlops/history?limit=10")
    assert res_hist.status_code == 200
    data_hist = res_hist.json()
    assert "history" in data_hist
    assert "total_count" in data_hist

    # 3. Test Retrain Endpoint
    res_retrain = client.post("/api/mlops/retrain", json={"trigger_reason": "api_test", "force": False})
    assert res_retrain.status_code == 200
    data_retrain = res_retrain.json()
    assert data_retrain["decision"] in ("PROMOTED", "REJECTED")
    assert "challenger" in data_retrain
