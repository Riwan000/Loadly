"""
Unit and integration tests for Model Drift Detection & Adaptive Retraining Engine.
"""

import os
from datetime import date, timedelta
import sqlite3
import tempfile
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import app
from utils.drift_detector import calculate_psi, compute_drift_metrics
from utils import model_retrainer
from utils.model_retrainer import (
    init_retraining_table,
    get_retraining_history,
    retrain_model,
    rollback_model,
)
from utils.prediction_ledger import init_ledger_db, get_db_connection


def _recent_date(days_ago: int) -> str:
    """Target date inside compute_drift_metrics' trailing window (default 14 days), relative to today."""
    return (date.today() - timedelta(days=days_ago + 1)).isoformat()


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
                f"fc-{i}", _recent_date(i), "Maharashtra", 1, "Reduced_LR",
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
                f"fc-{i}", _recent_date(i), "Maharashtra", 1, "Reduced_LR",
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

    # Test rollback (paths are monkeypatched to tmp by conftest, so read them from the module)
    if model_retrainer.MODEL_PREV_PATH.exists():
        rb = rollback_model(db_path=temp_db)
        assert rb["status"] == "SUCCESS"
        history_after_rb = get_retraining_history(limit=5, db_path=temp_db)
        assert history_after_rb[0]["decision"] == "ROLLED_BACK"


def test_mlops_api_endpoints(admin_headers):
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
    res_retrain = client.post(
        "/api/mlops/retrain", json={"trigger_reason": "api_test", "force": False}, headers=admin_headers
    )
    assert res_retrain.status_code == 200
    data_retrain = res_retrain.json()
    assert data_retrain["decision"] in ("PROMOTED", "REJECTED")
    assert "challenger" in data_retrain


# ---------------------------------------------------------------------------
# Retrainer behaviours (isolated: paths are monkeypatched to tmp by conftest)
# ---------------------------------------------------------------------------
import json
import pandas as pd


def _write_tiny_dataset(path, drop_col=None):
    """Small dataset with Total rows plus duplicate non-Total rows carrying garbage targets."""
    rng = np.random.default_rng(0)
    rows = []
    # Span > 12 months so the rolling split leaves a non-empty train set.
    for day in pd.date_range("2022-01-01", "2024-03-01", freq="D"):
        for state in ("Kerala", "Goa"):
            feats = {f: float(rng.normal(10, 1)) for f in model_retrainer.FEATURE_NAMES}
            target = 5.0 * feats["demand_roll_mean_7"] + 3.0
            rows.append({"date": day, "states": state, "source": "Total",
                         "energy met (mu)": target, **feats})
            rows.append({"date": day, "states": state, "source": "Coal",
                         "energy met (mu)": 99999.0, **feats})
    df = pd.DataFrame(rows)
    if drop_col:
        df = df.drop(columns=[drop_col])
    df.to_csv(path, index=False)


def test_retrain_raises_on_missing_feature_columns(tmp_path, monkeypatch, temp_db):
    ds = tmp_path / "ds.csv"
    _write_tiny_dataset(ds, drop_col="CDD")
    monkeypatch.setattr(model_retrainer, "DATASET_PATH", ds)
    with pytest.raises(ValueError, match="CDD"):
        retrain_model(trigger_reason="t", db_path=temp_db)


def test_retrain_uses_total_rows_and_preserves_unowned_metadata(tmp_path, monkeypatch, temp_db):
    ds = tmp_path / "ds.csv"
    _write_tiny_dataset(ds)
    monkeypatch.setattr(model_retrainer, "DATASET_PATH", ds)
    meta_path = model_retrainer.METADATA_PATH
    meta_path.write_text(json.dumps({
        "cdd_threshold_kelvin": 297.15,
        "extreme_heat_threshold_kelvin": 305.0,
        "horizon_rmse": {"1": 2.0},
        "rmse": 123.0,
    }))

    result = retrain_model(trigger_reason="t", force=True, db_path=temp_db)

    assert result["decision"] == "PROMOTED"
    # Duplicate "Coal" rows (target 99999) would wreck RMSE if they were not filtered out.
    assert result["challenger"]["rmse"] < 1.0
    meta = json.loads(meta_path.read_text())
    assert meta["cdd_threshold_kelvin"] == 297.15
    assert meta["extreme_heat_threshold_kelvin"] == 305.0
    # horizon_rmse is recomputed by the shared backtest on promotion (not left stale)
    assert set(meta["horizon_rmse"]) == {str(h) for h in range(1, 31)}
    assert "state_horizon_rmse" in meta and "baseline_horizon_rmse" in meta
    # rolling split: test = last 12 months of the data (2024-03-01 minus 12 months)
    assert meta["train_cutoff_date"] == "2023-03-02"
    assert meta["rmse"] < 1.0  # owned key was updated
    assert not list(meta_path.parent.glob("*.tmp"))


def test_retrain_and_rollback_reject_concurrent_calls(temp_db, admin_headers):
    assert model_retrainer._RETRAIN_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(model_retrainer.RetrainInProgressError):
            retrain_model(trigger_reason="t", db_path=temp_db)
        with pytest.raises(model_retrainer.RetrainInProgressError):
            rollback_model(db_path=temp_db)
        client = TestClient(app)
        assert client.post(
            "/api/mlops/retrain", json={"trigger_reason": "t", "force": False}, headers=admin_headers
        ).status_code == 409
        assert client.post("/api/mlops/rollback", headers=admin_headers).status_code == 409
    finally:
        model_retrainer._RETRAIN_LOCK.release()


def test_rollback_restores_previous_model_in_tmp(isolated_model_artifacts, temp_db):
    model_retrainer.MODEL_PATH.write_bytes(b"current")
    model_retrainer.MODEL_PREV_PATH.write_bytes(b"previous")
    result = rollback_model(db_path=temp_db)
    assert result["status"] == "SUCCESS"
    assert model_retrainer.MODEL_PATH.read_bytes() == b"previous"
    assert (isolated_model_artifacts / "best_model_rolled_back.pkl").read_bytes() == b"current"
