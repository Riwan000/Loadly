"""
Model Retraining & Champion-Challenger Validation Gate for Loadly.
Provides safe, automated, zero-downtime model retraining with:
1. Champion vs Challenger shadow evaluation
2. Multi-metric promotion guardrails (MAE, RMSE, R2, CI Coverage)
3. Atomic model file swap
4. In-memory hot reload
5. Full audit logging in prediction_ledger.db
6. Instant 1-click rollback capability
"""

import os
import json
import shutil
import pickle
import logging
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.file_lock import FileLock
from utils.prediction_ledger import get_db_connection
from models import reload_model_cache
from features import DEMAND_LAG_FEATURE_NAMES, PRODUCTION_FEATURE_NAMES
from utils.backtest import (
    TEST_WINDOW_MONTHS,
    evaluate_promotion_gate,
    rolling_split_cutoff,
    run_horizon_backtest,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
DATA_DIR = ROOT / "data"

MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"
MODEL_PREV_PATH = ARTIFACTS_DIR / "best_model_prev.pkl"
MODEL_CANDIDATE_PATH = ARTIFACTS_DIR / "best_model_candidate.pkl"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
# Backup of the champion's metadata, rolled back together with best_model_prev.pkl.
METADATA_PREV_PATH = ARTIFACTS_DIR / "model_metadata_prev.json"
RETRAIN_LOCK_PATH = ARTIFACTS_DIR / "retrain.lock"
DATASET_PATH = DATA_DIR / "processed" / "feature_engineered_dataset.csv"

# Champion-challenger gate: challenger RMSE may be at most 2% worse than the champion.
RMSE_TOLERANCE = 1.02
MIN_CHALLENGER_R2 = 0.985
MIN_CHALLENGER_COVERAGE_PCT = 85.0
# Rows in the feature dataset are duplicated once per generation source; only the
# "Total" rows are the demand target (matches utils/train_pipeline.py).
TOTAL_SOURCE = "Total"

# Metadata keys that describe the model's multi-step error profile; replaced by the
# candidate's backtest on promotion, or dropped (CIs fall back to flat state_rmse).
HORIZON_KEYS = ("horizon_rmse", "state_horizon_rmse", "national_horizon_rmse",
                "baseline_horizon_rmse", "backtest_config")
# train_pipeline-only keys the retrainer cannot recompute; dropped on promotion.
STALE_PIPELINE_KEYS = ("baselines", "evaluation", "candidate_mean_horizon_rmse",
                       "reference_horizon_rmse", "ridge_alpha", "promotion_gate")
# The retrainer always refits an ordinary least-squares model on the production feature list.
RETRAINED_MODEL_NAME = "LR_lags"

# A retrain holds the lock for minutes; a lock file older than this is from a crashed process.
RETRAIN_LOCK_STALE_SECONDS = 3600

# In-process lock + cross-process lock file (see _acquire_retrain_lock): serializes
# retrain/rollback across threads and across uvicorn worker processes.
_RETRAIN_LOCK = threading.Lock()
_retrain_file_lock: Optional[FileLock] = None


def _acquire_retrain_lock() -> bool:
    """
    Non-blocking acquire of the thread lock and the cross-process lock file.

    The lock file is a utils.file_lock.FileLock: a lock left by a process that is no
    longer running (same host) is reclaimed immediately, any other lock once it is older
    than RETRAIN_LOCK_STALE_SECONDS.
    """
    global _retrain_file_lock
    if not _RETRAIN_LOCK.acquire(blocking=False):
        return False
    try:
        file_lock = FileLock(Path(RETRAIN_LOCK_PATH), RETRAIN_LOCK_STALE_SECONDS)
        if file_lock.acquire():
            _retrain_file_lock = file_lock
            return True
    except BaseException:
        _RETRAIN_LOCK.release()
        raise
    _RETRAIN_LOCK.release()
    return False


def _lock_busy_message() -> str:
    """Explain a held lock: its age, when it auto-expires, and how to clear a crashed one."""
    base = "A model retrain or rollback is already in progress."
    try:
        age_seconds = max(0.0, datetime.now().timestamp() - Path(RETRAIN_LOCK_PATH).stat().st_mtime)
    except FileNotFoundError:
        return base  # only the in-process thread lock is held
    age_min = int(age_seconds // 60)
    expires_min = max(0, int((RETRAIN_LOCK_STALE_SECONDS - age_seconds) // 60))
    return (f"{base} The lock has been held for {age_min} min. It is reclaimed automatically as soon "
            f"as the holding process is no longer running, and in {expires_min} min at the latest. "
            f"If no retrain is actually running and the lock persists, delete "
            f"artifacts/{Path(RETRAIN_LOCK_PATH).name} to clear it.")


def _release_retrain_lock() -> None:
    global _retrain_file_lock
    try:
        if _retrain_file_lock is not None:
            _retrain_file_lock.release()
    finally:
        _retrain_file_lock = None
        _RETRAIN_LOCK.release()


class RetrainInProgressError(RuntimeError):
    """Raised when a retrain or rollback is requested while another is already running."""


# One shared feature list with utils/train_pipeline.py and serving (features.PRODUCTION_FEATURE_NAMES).
FEATURE_NAMES = list(PRODUCTION_FEATURE_NAMES)


def init_retraining_table(db_path: Optional[Path] = None) -> None:
    """Initialize the retraining audit history table."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS retraining_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    retrained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    trigger_reason TEXT NOT NULL,
                    champion_model TEXT,
                    challenger_model TEXT,
                    champion_mae REAL,
                    challenger_mae REAL,
                    champion_rmse REAL,
                    challenger_rmse REAL,
                    champion_r2 REAL,
                    challenger_r2 REAL,
                    champion_coverage REAL,
                    challenger_coverage REAL,
                    decision TEXT NOT NULL,
                    notes TEXT
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_retrain_time ON retraining_history(retrained_at);")
    finally:
        conn.close()


def get_retraining_history(limit: int = 15, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve past model retraining and evaluation records."""
    init_retraining_table(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute("""
            SELECT 
                id, retrained_at, trigger_reason, champion_model, challenger_model,
                champion_mae, challenger_mae, champion_rmse, challenger_rmse,
                champion_r2, challenger_r2, champion_coverage, challenger_coverage,
                decision, notes
            FROM retraining_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON to a temp file in the same directory, then os.replace it into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy src over dst via a temp file in dst's directory + os.replace (readers never see a partial file)."""
    dst = Path(dst)
    fd, tmp_name = tempfile.mkstemp(dir=dst.parent, prefix=dst.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        shutil.copy2(src, tmp_name)
        os.replace(tmp_name, dst)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _load_existing_metadata() -> Dict[str, Any]:
    """Current metadata (so keys the retrainer does not own are preserved); {} if absent/corrupt."""
    if not Path(METADATA_PATH).exists():
        return {}
    try:
        with open(METADATA_PATH, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Could not read existing metadata at {METADATA_PATH}: {e}")
        return {}


def _build_promoted_metadata(
    challenger_stats,
    df_test_eval: pd.DataFrame,
    trigger_reason: str,
    backtest: Optional[Dict[str, Any]],
    fit_dates: pd.Series,
    train_cutoff,
) -> Dict[str, Any]:
    """
    Metadata for a promoted challenger, built from the current metadata.

    Keys the retrainer does not own (cdd/extreme-heat thresholds, target_scale, ...) survive.
    Keys describing the previous model that the retrainer cannot vouch for are handled so
    no stale value ships: horizon-specific keys are replaced by the candidate's backtest, or
    dropped when there is none (CIs then fall back to flat state_rmse); `baselines`,
    `evaluation` and `candidate_mean_horizon_rmse` are train_pipeline-only and always
    dropped; `trained_on` is refreshed from the rows the DEPLOYED model was fit on (all data).
    rmse/state_rmse/horizon keys are held-out estimates of the train-only fit (see
    `error_estimates_from`).
    """
    chall_rmse, chall_mae, chall_r2 = challenger_stats
    state_rmse = {
        name: float(np.sqrt(mean_squared_error(g["actual"], g["pred"])))
        for name, g in df_test_eval.groupby("states") if len(g) > 0
    }
    rmse_values = list(state_rmse.values())

    metadata = _load_existing_metadata()
    for key in HORIZON_KEYS + STALE_PIPELINE_KEYS:
        metadata.pop(key, None)
    metadata.update({
        "model_name": RETRAINED_MODEL_NAME,
        "last_retrained_at": datetime.now().isoformat(),
        "trigger_reason": trigger_reason,
        "rmse": chall_rmse,
        "mae": chall_mae,
        "r2": chall_r2,
        "confidence_level": 0.90,
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "state_rmse": state_rmse,
        "state_rmse_summary": {
            "total_states": len(state_rmse),
            "successful": len(state_rmse),
            "mean_rmse": float(np.mean(rmse_values)) if rmse_values else chall_rmse,
            "min_rmse": float(np.min(rmse_values)) if rmse_values else chall_rmse,
            "max_rmse": float(np.max(rmse_values)) if rmse_values else chall_rmse,
        },
        "trained_on": f"{fit_dates.min().date()} to {fit_dates.max().date()}",
        "train_cutoff_date": str(train_cutoff.date()),
        "error_estimates_from": (f"held-out backtest, train-only fit (trained before {train_cutoff.date()}); "
                                 f"deployed model refit on all data through {fit_dates.max().date()}"),
        "test_window_months": TEST_WINDOW_MONTHS,
    })
    if backtest is not None:
        for key in HORIZON_KEYS:
            metadata[key] = backtest[key]
        metadata["promotion_gate"] = {"passed": evaluate_promotion_gate(backtest)[0]}
    return metadata


def retrain_model(
    trigger_reason: str = "manual",
    force: bool = False,
    db_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Execute Champion vs Challenger retraining pipeline (serialized by a module lock).

    Raises RetrainInProgressError if another retrain/rollback is running,
    FileNotFoundError if the dataset is missing, ValueError if required feature
    columns are absent.
    """
    if not _acquire_retrain_lock():
        raise RetrainInProgressError(_lock_busy_message())
    try:
        return _retrain_model_locked(trigger_reason, force, db_path)
    finally:
        _release_retrain_lock()


def _retrain_model_locked(
    trigger_reason: str,
    force: bool,
    db_path: Optional[Path],
) -> Dict[str, Any]:
    """
    Execute Champion vs Challenger retraining pipeline.
    
    1. Loads dataset.
    2. Evaluates current Champion model.
    3. Fits new Challenger model on the train period only.
    4. Evaluates Challenger under identical test split (one-step gate + horizon backtest gate).
    5. Validates promotion guardrails.
    6. If promoted: REFITS the challenger on all rows (train + test window), then atomically
       deploys and hot-reloads memory. Error estimates stay those of the train-only fit.
    7. Logs full telemetry to retraining_history.
    """
    init_retraining_table(db_path)
    
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Feature dataset not found at {DATASET_PATH}. Cannot retrain.")

    df = pd.read_csv(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # Each (date, state) row is duplicated per generation source; keep only the Total rows.
    if "source" not in df.columns:
        raise ValueError(f"Dataset {DATASET_PATH} has no 'source' column; cannot select '{TOTAL_SOURCE}' rows.")
    df = df[df["source"] == TOTAL_SOURCE]

    missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset {DATASET_PATH} is missing required feature columns: {missing_features}")

    df_clean = df.dropna(subset=["energy met (mu)"]).copy()
    lag_cols = [f for f in FEATURE_NAMES if f in DEMAND_LAG_FEATURE_NAMES]
    # Rows with an undefined demand lag (series warm-up, holes) are never fit on or scored one-step;
    # the backtest builds lags from its own buffer and uses every row.
    fit_rows = df_clean[lag_cols].notna().all(axis=1)
    other_features = [f for f in FEATURE_NAMES if f not in lag_cols]
    df_clean[other_features] = df_clean[other_features].replace([np.inf, -np.inf], 0).fillna(0)

    # Rolling temporal split: test = last TEST_WINDOW_MONTHS of available data (same
    # rule as utils/train_pipeline.py), train = everything before, so retraining learns
    # from newly ingested data instead of a frozen 2024-01-01 cutoff.
    train_cutoff = rolling_split_cutoff(df_clean["date"])
    train_idx = (df_clean["date"] < train_cutoff) & fit_rows
    test_idx = (df_clean["date"] >= train_cutoff) & fit_rows

    X_train = df_clean.loc[train_idx, FEATURE_NAMES]
    y_train = df_clean.loc[train_idx, "energy met (mu)"]
    X_test = df_clean.loc[test_idx, FEATURE_NAMES]
    y_test = df_clean.loc[test_idx, "energy met (mu)"]

    # 1. Load & Evaluate Current Champion
    champion_model = None
    champ_mae = 999.0
    champ_rmse = 999.0
    champ_r2 = 0.0
    champ_coverage = 0.0
    champ_model_name = "Reduced_LR (incumbent)"

    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                champion_model = pickle.load(f)
            # Evaluate the incumbent on ITS OWN features (a legacy 15-feature model has no lags).
            champ_features = list(getattr(champion_model, "feature_names_in_", FEATURE_NAMES))
            champ_pred = champion_model.predict(df_clean.loc[test_idx, champ_features])
            champ_mae = float(mean_absolute_error(y_test, champ_pred))
            champ_rmse = float(np.sqrt(mean_squared_error(y_test, champ_pred)))
            champ_r2 = float(r2_score(y_test, champ_pred))
            # Approximate 90% CI coverage
            champ_ci_lower = champ_pred - 1.645 * champ_rmse
            champ_ci_upper = champ_pred + 1.645 * champ_rmse
            champ_coverage = float(np.mean((y_test >= champ_ci_lower) & (y_test <= champ_ci_upper)) * 100.0)
        except Exception as e:
            logger.warning(f"Could not evaluate current champion model: {e}")

    # 2. Train Challenger (Reduced Linear Regression with refreshed data)
    challenger = LinearRegression()
    challenger.fit(X_train, y_train)

    chall_pred = challenger.predict(X_test)
    chall_mae = float(mean_absolute_error(y_test, chall_pred))
    chall_rmse = float(np.sqrt(mean_squared_error(y_test, chall_pred)))
    chall_r2 = float(r2_score(y_test, chall_pred))

    chall_ci_lower = chall_pred - 1.645 * chall_rmse
    chall_ci_upper = chall_pred + 1.645 * chall_rmse
    chall_coverage = float(np.mean((y_test >= chall_ci_lower) & (y_test <= chall_ci_upper)) * 100.0)
    chall_model_name = "Reduced_LR (challenger)"

    # 3. Champion-Challenger Validation Gate
    # Promotion conditions:
    # A. Challenger RMSE no worse than RMSE_TOLERANCE x Champion RMSE (2%), or champion is missing
    # B. Challenger R2 >= MIN_CHALLENGER_R2
    # C. Challenger Coverage >= MIN_CHALLENGER_COVERAGE_PCT
    gate_passed = True
    gate_reasons = []

    if champion_model is not None:
        if chall_rmse > champ_rmse * RMSE_TOLERANCE:
            gate_passed = False
            gate_reasons.append(f"Challenger RMSE ({chall_rmse:.3f} MU) exceeded Champion RMSE ({champ_rmse:.3f} MU) by >2%.")
        
        if chall_r2 < MIN_CHALLENGER_R2:
            gate_passed = False
            gate_reasons.append(f"Challenger R² ({chall_r2:.4f}) below minimum {MIN_CHALLENGER_R2} threshold.")

        if chall_coverage < MIN_CHALLENGER_COVERAGE_PCT:
            gate_passed = False
            gate_reasons.append(f"Challenger CI coverage ({chall_coverage:.1f}%) below {MIN_CHALLENGER_COVERAGE_PCT:.0f}% minimum.")

    decision = "PROMOTED" if (gate_passed or force) else "REJECTED"
    notes = "Gate criteria met." if decision == "PROMOTED" else "; ".join(gate_reasons)
    if force and not gate_passed:
        notes = f"Forced promotion overriding gate failure: {'; '.join(gate_reasons)}"

    # 4. If Promoted: backtest the candidate, then atomic file swap & cache reload
    warnings: List[str] = []
    backtest = None
    if decision == "PROMOTED":
        # The horizon backtest runs on the candidate BEFORE the swap so a failure can
        # never ship the new model with the previous model's horizon_rmse.
        try:
            backtest = run_horizon_backtest(challenger, df_clean, FEATURE_NAMES, train_cutoff)
        except Exception as e:
            logger.error(f"Horizon backtest failed on candidate: {e}")
            if not force:
                decision = "REJECTED"
                notes = f"Candidate not promoted: horizon backtest failed ({e})."
            else:
                warnings.append(
                    f"Horizon backtest failed ({e}); promoted by force without horizon-specific "
                    "error profile, so confidence intervals use flat per-state RMSE."
                )
        if backtest is not None:
            # Baseline gate on the held-out multi-horizon backtest (shared with train_pipeline).
            baseline_ok, baseline_reasons = evaluate_promotion_gate(backtest)
            if not baseline_ok and not force:
                decision = "REJECTED"
                gate_passed = False
                notes = f"Candidate not promoted: baseline gate failed ({'; '.join(baseline_reasons)})"
            elif not baseline_ok:
                warnings.append(f"Baseline gate failed but promoted by force: {'; '.join(baseline_reasons)}")

    if decision == "PROMOTED":
        metadata = _build_promoted_metadata(
            challenger_stats=(chall_rmse, chall_mae, chall_r2),
            df_test_eval=df_clean.loc[test_idx].assign(pred=chall_pred, actual=y_test),
            trigger_reason=trigger_reason,
            backtest=backtest,
            fit_dates=df_clean.loc[fit_rows, "date"],
            train_cutoff=train_cutoff,
        )
        # Refit on ALL rows only now that evaluation is finished; the held-out numbers above
        # describe the train-only fit.
        deployed = clone(challenger).fit(df_clean.loc[fit_rows, FEATURE_NAMES],
                                         df_clean.loc[fit_rows, "energy met (mu)"])

        # Step A: Back up the current champion AND its metadata (rolled back together)
        if MODEL_PATH.exists():
            shutil.copy2(MODEL_PATH, MODEL_PREV_PATH)
        if Path(METADATA_PATH).exists():
            shutil.copy2(METADATA_PATH, METADATA_PREV_PATH)
        elif Path(METADATA_PREV_PATH).exists():
            Path(METADATA_PREV_PATH).unlink()  # never pair a backup model with an unrelated old metadata

        # Step B: Write candidate model, Step C: atomic replace
        with open(MODEL_CANDIDATE_PATH, "wb") as f:
            pickle.dump(deployed, f)
        os.replace(MODEL_CANDIDATE_PATH, MODEL_PATH)

        # Step D: metadata (atomic) matching the new model
        _write_json_atomic(METADATA_PATH, metadata)

        # Step E: Zero-downtime hot reload
        reload_model_cache()
        logger.info("Successfully deployed new model with zero downtime.")

    if warnings:
        notes = f"{notes} WARNING: {' '.join(warnings)}"

    # 5. Record event in Retraining History Table
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute("""
                INSERT INTO retraining_history (
                    trigger_reason, champion_model, challenger_model,
                    champion_mae, challenger_mae, champion_rmse, challenger_rmse,
                    champion_r2, challenger_r2, champion_coverage, challenger_coverage,
                    decision, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trigger_reason, champ_model_name, chall_model_name,
                champ_mae if champion_model else None, chall_mae,
                champ_rmse if champion_model else None, chall_rmse,
                champ_r2 if champion_model else None, chall_r2,
                champ_coverage if champion_model else None, chall_coverage,
                decision, notes
            ))
    finally:
        conn.close()

    return {
        "decision": decision,
        "trigger_reason": trigger_reason,
        "champion": {
            "model_name": champ_model_name,
            "mae": round(champ_mae, 4) if champion_model else None,
            "rmse": round(champ_rmse, 4) if champion_model else None,
            "r2": round(champ_r2, 6) if champion_model else None,
            "ci_coverage_pct": round(champ_coverage, 2) if champion_model else None
        },
        "challenger": {
            "model_name": chall_model_name,
            "mae": round(chall_mae, 4),
            "rmse": round(chall_rmse, 4),
            "r2": round(chall_r2, 6),
            "ci_coverage_pct": round(chall_coverage, 2)
        },
        "gate_passed": gate_passed,
        "notes": notes,
        "rollback_available": MODEL_PREV_PATH.exists(),
        "warnings": warnings,
    }


def rollback_model(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Roll back to the previous champion model if a backup exists (serialized with retrain).

    Raises RetrainInProgressError if another retrain/rollback is running.
    """
    if not _acquire_retrain_lock():
        raise RetrainInProgressError(_lock_busy_message())
    try:
        return _rollback_model_locked(db_path)
    finally:
        _release_retrain_lock()


def _rollback_model_locked(db_path: Optional[Path]) -> Dict[str, Any]:
    init_retraining_table(db_path)

    if not MODEL_PREV_PATH.exists():
        raise FileNotFoundError("No previous model backup (best_model_prev.pkl) found to roll back to.")

    # Swap model: keep the rolled-back-from model, then replace atomically
    temp_backup = Path(MODEL_PATH).parent / "best_model_rolled_back.pkl"
    shutil.copy2(MODEL_PATH, temp_backup)
    _atomic_copy(MODEL_PREV_PATH, MODEL_PATH)

    # Restore the metadata that describes the restored model (state_rmse, horizon_rmse, ...)
    warning = None
    if Path(METADATA_PREV_PATH).exists():
        if Path(METADATA_PATH).exists():
            shutil.copy2(METADATA_PATH, Path(METADATA_PATH).parent / "model_metadata_rolled_back.json")
        _atomic_copy(METADATA_PREV_PATH, METADATA_PATH)
    else:
        warning = (
            f"No metadata backup ({Path(METADATA_PREV_PATH).name}) found; model restored but "
            "metadata (RMSE / confidence intervals) still describes the newer model."
        )
        logger.warning(warning)

    # Hot reload memory
    reload_model_cache()

    # Log rollback
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute("""
                INSERT INTO retraining_history (
                    trigger_reason, champion_model, challenger_model,
                    decision, notes
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                "manual_rollback", "Current Champion", "best_model_prev.pkl",
                "ROLLED_BACK", "Reverted model weights to previous champion version."
            ))
    finally:
        conn.close()

    return {
        "status": "SUCCESS",
        "message": "Successfully rolled back model to previous champion version.",
        "reloaded": True,
        "warning": warning,
    }
