"""Retrainer safety: backtest-before-swap, metadata rollback, cross-process lock."""

import json
import os
import time

import pytest

from tests.test_drift_detector import _write_tiny_dataset, temp_db  # noqa: F401
from utils import model_retrainer
from utils.model_retrainer import retrain_model, rollback_model

OLD_META = {
    "rmse": 123.0,
    "state_rmse": {"Kerala": 9.0},
    "horizon_rmse": {"1": 2.0},
    "state_horizon_rmse": {"Kerala": {"1": 2.0}},
    "national_horizon_rmse": {"1": 5.0},
    "baseline_horizon_rmse": {"persistence": {"1": 3.0}},
    "baselines": {"persistence": 1.0},
    "evaluation": "old text",
    "trained_on": "2013-01-01 to 2020-01-01",
    "candidate_mean_horizon_rmse": {"x": 1.0},
    "cdd_threshold_kelvin": 297.15,
}


@pytest.fixture
def tiny_dataset(tmp_path, monkeypatch):
    ds = tmp_path / "ds.csv"
    _write_tiny_dataset(ds)
    monkeypatch.setattr(model_retrainer, "DATASET_PATH", ds)
    model_retrainer.METADATA_PATH.write_text(json.dumps(OLD_META))
    return ds


def _boom(*args, **kwargs):
    raise RuntimeError("backtest exploded")


def test_backtest_failure_rejects_promotion_and_leaves_files_untouched(tiny_dataset, monkeypatch, temp_db):  # noqa: F811
    monkeypatch.setattr(model_retrainer, "run_horizon_backtest", _boom)
    model_before = model_retrainer.MODEL_PATH.read_bytes()

    result = retrain_model(trigger_reason="t", force=False, db_path=temp_db)

    # The gate may reject first; either way nothing is promoted and nothing changes.
    assert result["decision"] == "REJECTED"
    assert model_retrainer.MODEL_PATH.read_bytes() == model_before
    assert json.loads(model_retrainer.METADATA_PATH.read_text()) == OLD_META
    assert not model_retrainer.MODEL_PREV_PATH.exists()


def test_backtest_failure_with_gate_passing_is_rejected_with_note(tiny_dataset, monkeypatch, temp_db):  # noqa: F811
    monkeypatch.setattr(model_retrainer, "run_horizon_backtest", _boom)
    monkeypatch.setattr(model_retrainer, "RMSE_TOLERANCE", 1e9)
    monkeypatch.setattr(model_retrainer, "MIN_CHALLENGER_R2", -1e9)
    monkeypatch.setattr(model_retrainer, "MIN_CHALLENGER_COVERAGE_PCT", 0.0)

    result = retrain_model(trigger_reason="t", force=False, db_path=temp_db)

    assert result["decision"] == "REJECTED"
    assert "backtest failed" in result["notes"]
    assert json.loads(model_retrainer.METADATA_PATH.read_text()) == OLD_META


def test_forced_promotion_with_backtest_failure_drops_horizon_and_stale_keys(tiny_dataset, monkeypatch, temp_db):  # noqa: F811
    monkeypatch.setattr(model_retrainer, "run_horizon_backtest", _boom)

    result = retrain_model(trigger_reason="t", force=True, db_path=temp_db)

    assert result["decision"] == "PROMOTED"
    assert any("backtest failed" in w for w in result["warnings"])
    meta = json.loads(model_retrainer.METADATA_PATH.read_text())
    for key in model_retrainer.HORIZON_KEYS + model_retrainer.STALE_PIPELINE_KEYS:
        assert key not in meta
    assert meta["cdd_threshold_kelvin"] == 297.15  # unowned key kept
    assert meta["trained_on"].startswith("2022-01-01")  # refreshed, not the stale 2013 range
    assert meta["rmse"] != 123.0


def test_promotion_backs_up_metadata_and_rollback_restores_it(tiny_dataset, temp_db):  # noqa: F811
    result = retrain_model(trigger_reason="t", force=True, db_path=temp_db)
    assert result["decision"] == "PROMOTED"
    assert json.loads(model_retrainer.METADATA_PREV_PATH.read_text()) == OLD_META
    assert json.loads(model_retrainer.METADATA_PATH.read_text())["rmse"] != 123.0

    rb = rollback_model(db_path=temp_db)

    assert rb["status"] == "SUCCESS" and rb["warning"] is None
    assert json.loads(model_retrainer.METADATA_PATH.read_text()) == OLD_META
    assert not list(model_retrainer.METADATA_PATH.parent.glob("*.tmp"))


def test_rollback_without_metadata_backup_keeps_metadata_and_warns(temp_db):  # noqa: F811
    model_retrainer.MODEL_PATH.write_bytes(b"current")
    model_retrainer.MODEL_PREV_PATH.write_bytes(b"previous")
    model_retrainer.METADATA_PATH.write_text(json.dumps({"rmse": 1.0}))

    rb = rollback_model(db_path=temp_db)

    assert model_retrainer.MODEL_PATH.read_bytes() == b"previous"
    assert json.loads(model_retrainer.METADATA_PATH.read_text()) == {"rmse": 1.0}
    assert "metadata" in rb["warning"]


# ---------------------------------------------------------------------------
# Cross-process lock file
# ---------------------------------------------------------------------------
def test_lock_file_blocks_other_process_and_is_released():
    lock = model_retrainer.RETRAIN_LOCK_PATH
    assert model_retrainer._acquire_retrain_lock()
    try:
        assert lock.exists()
        # Simulate a second process: the in-process lock is free but the lock file exists.
        model_retrainer._RETRAIN_LOCK.release()
        try:
            assert not model_retrainer._acquire_retrain_lock()
        finally:
            assert model_retrainer._RETRAIN_LOCK.acquire(blocking=False)
    finally:
        model_retrainer._release_retrain_lock()
    assert not lock.exists()
    assert model_retrainer._acquire_retrain_lock()
    model_retrainer._release_retrain_lock()


def test_foreign_lock_file_gives_409_from_api(admin_headers):
    from fastapi.testclient import TestClient
    from server import app

    model_retrainer.RETRAIN_LOCK_PATH.write_text("99999 other-process")
    client = TestClient(app)
    assert client.post("/api/mlops/retrain", json={"trigger_reason": "t", "force": False},
                       headers=admin_headers).status_code == 409
    assert client.post("/api/mlops/rollback", headers=admin_headers).status_code == 409
    # A rejected caller must not delete the other process's lock.
    assert model_retrainer.RETRAIN_LOCK_PATH.exists()
    assert model_retrainer._RETRAIN_LOCK.acquire(blocking=False)  # thread lock was released
    model_retrainer._RETRAIN_LOCK.release()


def test_stale_lock_file_is_reclaimed(monkeypatch):
    lock = model_retrainer.RETRAIN_LOCK_PATH
    lock.write_text("99999 crashed")
    old = time.time() - model_retrainer.RETRAIN_LOCK_STALE_SECONDS - 60
    os.utime(lock, (old, old))

    assert model_retrainer._acquire_retrain_lock()
    try:
        assert lock.read_text().startswith(str(os.getpid()))
        assert not list(lock.parent.glob("retrain.lock.stale.*"))
    finally:
        model_retrainer._release_retrain_lock()


def test_lock_released_after_retrain_error(temp_db, monkeypatch):  # noqa: F811
    monkeypatch.setattr(model_retrainer, "DATASET_PATH", model_retrainer.ROOT / "does_not_exist.csv")
    with pytest.raises(FileNotFoundError):
        retrain_model(trigger_reason="t", db_path=temp_db)
    assert not model_retrainer.RETRAIN_LOCK_PATH.exists()
