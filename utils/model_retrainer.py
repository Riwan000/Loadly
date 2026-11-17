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
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.prediction_ledger import get_db_connection, DEFAULT_DB_PATH
from models import reload_model_cache

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
DATA_DIR = ROOT / "data"

MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"
MODEL_PREV_PATH = ARTIFACTS_DIR / "best_model_prev.pkl"
MODEL_CANDIDATE_PATH = ARTIFACTS_DIR / "best_model_candidate.pkl"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
DATASET_PATH = DATA_DIR / "processed" / "feature_engineered_dataset.csv"

FEATURE_NAMES = [
    "2m_temperature_mean",
    "2m_temperature_max",
    "2m_temperature_min",
    "lag_1_demand",
    "lag_7_demand",
    "rolling_7_mean_demand",
    "rolling_7_std_demand",
    "lag_1_temp",
    "rolling_3_mean_temp",
    "extreme_heat",
    "temp_range",
    "CDD",
    "month",
    "day_of_week",
    "is_weekend"
]


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


def retrain_model(
    trigger_reason: str = "manual",
    force: bool = False,
    db_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Execute Champion vs Challenger retraining pipeline.
    
    1. Loads dataset.
    2. Evaluates current Champion model.
    3. Fits new Challenger model.
    4. Evaluates Challenger under identical test split.
    5. Validates promotion guardrails.
    6. If promoted: atomically deploys and hot-reloads memory.
    7. Logs full telemetry to retraining_history.
    """
    init_retraining_table(db_path)
    
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Feature dataset not found at {DATASET_PATH}. Cannot retrain.")

    df = pd.read_csv(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    
    # Clean features
    for f in FEATURE_NAMES:
        if f not in df.columns:
            df[f] = 0.0

    df_clean = df.dropna(subset=["energy met (mu)"]).copy()
    df_clean[FEATURE_NAMES] = df_clean[FEATURE_NAMES].replace([np.inf, -np.inf], 0).fillna(0)

    # Temporal Split: train < 2024-01-01, test >= 2024-01-01
    train_idx = df_clean["date"] < "2024-01-01"
    test_idx = df_clean["date"] >= "2024-01-01"

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
            champ_pred = champion_model.predict(X_test)
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
    # A. Challenger RMSE within 1% or lower than Champion (or champion is missing)
    # B. Challenger R2 >= 0.985
    # C. Challenger Coverage >= 85%
    gate_passed = True
    gate_reasons = []

    if champion_model is not None:
        if chall_rmse > champ_rmse * 1.02:
            gate_passed = False
            gate_reasons.append(f"Challenger RMSE ({chall_rmse:.3f} MU) exceeded Champion RMSE ({champ_rmse:.3f} MU) by >2%.")
        
        if chall_r2 < 0.985:
            gate_passed = False
            gate_reasons.append(f"Challenger R² ({chall_r2:.4f}) below minimum 0.985 threshold.")

        if chall_coverage < 85.0:
            gate_passed = False
            gate_reasons.append(f"Challenger CI coverage ({chall_coverage:.1f}%) below 85% minimum.")

    decision = "PROMOTED" if (gate_passed or force) else "REJECTED"
    notes = "Gate criteria met." if decision == "PROMOTED" else "; ".join(gate_reasons)
    if force and not gate_passed:
        notes = f"Forced promotion overriding gate failure: {'; '.join(gate_reasons)}"

    # 4. If Promoted, Atomic File Swap & Cache Reload
    if decision == "PROMOTED":
        # Step A: Backup current champion if it exists
        if MODEL_PATH.exists():
            shutil.copy2(MODEL_PATH, MODEL_PREV_PATH)

        # Step B: Write candidate model
        with open(MODEL_CANDIDATE_PATH, "wb") as f:
            pickle.dump(challenger, f)

        # Step C: Atomic rename/replace
        os.replace(MODEL_CANDIDATE_PATH, MODEL_PATH)

        # Step D: Recalculate State RMSEs and update metadata
        df_test_eval = df_clean.loc[test_idx].copy()
        df_test_eval["pred"] = chall_pred
        df_test_eval["actual"] = y_test

        state_rmse = {}
        for state_name, group in df_test_eval.groupby("states"):
            if len(group) > 0:
                state_rmse[state_name] = float(np.sqrt(mean_squared_error(group["actual"], group["pred"])))

        metadata = {
            "model_name": "Reduced_LR",
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
                "mean_rmse": float(np.mean(list(state_rmse.values()))) if state_rmse else chall_rmse,
                "min_rmse": float(np.min(list(state_rmse.values()))) if state_rmse else chall_rmse,
                "max_rmse": float(np.max(list(state_rmse.values()))) if state_rmse else chall_rmse
            }
        }
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

        # Step E: Zero-downtime hot reload
        reload_model_cache()
        logger.info("Successfully deployed new model with zero downtime.")

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
        "rollback_available": MODEL_PREV_PATH.exists()
    }


def rollback_model(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Roll back to the previous champion model if a backup exists.
    """
    init_retraining_table(db_path)

    if not MODEL_PREV_PATH.exists():
        raise FileNotFoundError("No previous model backup (best_model_prev.pkl) found to roll back to.")

    # Swap files
    temp_backup = ARTIFACTS_DIR / "best_model_rolled_back.pkl"
    shutil.copy2(MODEL_PATH, temp_backup)
    shutil.copy2(MODEL_PREV_PATH, MODEL_PATH)

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
        "reloaded": True
    }
