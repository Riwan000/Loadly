"""
Prediction Ledger & Verification Engine for Loadly.
Tracks every generated forecast and reconciles against ground-truth actuals.
"""

import os
import sqlite3
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "prediction_ledger.db"


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Create and return a SQLite connection with foreign keys and Row factory enabled."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_ledger_db(db_path: Optional[Path] = None) -> None:
    """Initialize the forecast ledger table and associated indexes."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forecast_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    forecast_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    state TEXT NOT NULL,
                    target_date TEXT NOT NULL,
                    horizon_step INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    predicted_demand_mu REAL NOT NULL,
                    lower_bound_mu REAL,
                    upper_bound_mu REAL,
                    temperature_mean_c REAL,
                    cdd REAL,
                    extreme_heat INTEGER DEFAULT 0,
                    actual_demand_mu REAL,
                    reconciled_at TIMESTAMP,
                    error_mu REAL,
                    abs_error_mu REAL,
                    pct_error REAL,
                    within_ci INTEGER
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ledger_state_target ON forecast_ledger(state, target_date);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ledger_forecast_id ON forecast_ledger(forecast_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ledger_created_at ON forecast_ledger(created_at);")
    finally:
        conn.close()


def record_forecast_batch(
    timeline_items: List[Dict[str, Any]],
    state: str,
    model_name: str = "Reduced_LR",
    db_path: Optional[Path] = None,
    auto_reconcile: bool = True,
    horizon_offset: int = 0
) -> str:
    """
    Record a batch of forecasted timeline items (e.g. 7, 14, or 30 days) into the ledger.
    Returns the unique forecast_id (UUID4).

    horizon_step is the true lead time of each row: its 1-based position plus
    horizon_offset (the number of unobserved days the forecast was bridged over).
    """
    init_ledger_db(db_path)
    forecast_id = str(uuid.uuid4())
    conn = get_db_connection(db_path)

    try:
        # Pre-fetch historical demand if auto_reconcile is requested
        historical_actuals = {}
        if auto_reconcile:
            try:
                from data_loading import load_historical_demand
                hist_df = load_historical_demand(state)
                if hist_df is not None and not hist_df.empty:
                    # Map date string 'YYYY-MM-DD' to actual_demand_MU
                    hist_df['dt_str'] = hist_df['date'].dt.strftime('%Y-%m-%d')
                    historical_actuals = dict(zip(hist_df['dt_str'], hist_df['actual_demand_MU']))
            except Exception as e:
                logger.debug(f"Could not load historical demand for auto-reconciliation: {e}")

        rows_to_insert = []
        now_iso = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        for idx, item in enumerate(timeline_items):
            dt_str = item.get("date")
            pred_mu = float(item.get("forecasted_demand_mu", 0.0))
            lower_mu = float(item["lower_bound_mu"]) if item.get("lower_bound_mu") is not None else None
            upper_mu = float(item["upper_bound_mu"]) if item.get("upper_bound_mu") is not None else None
            temp_c = float(item["temperature_mean_c"]) if item.get("temperature_mean_c") is not None else None
            cdd_val = float(item["cdd"]) if item.get("cdd") is not None else None
            ext_heat = int(item.get("extreme_heat", 0))

            actual_val = historical_actuals.get(dt_str)
            reconciled_at = None
            error_mu = None
            abs_err = None
            pct_err = None
            within_ci = None

            if actual_val is not None:
                actual_val = float(actual_val)
                reconciled_at = now_iso
                error_mu = round(actual_val - pred_mu, 4)
                abs_err = round(abs(error_mu), 4)
                pct_err = round((abs_err / actual_val) * 100.0, 2) if actual_val > 0 else 0.0
                
                # Check 90% confidence interval coverage
                if lower_mu is not None and upper_mu is not None:
                    within_ci = 1 if (lower_mu <= actual_val <= upper_mu) else 0
                elif lower_mu is not None:
                    within_ci = 1 if (lower_mu <= actual_val) else 0
                elif upper_mu is not None:
                    within_ci = 1 if (actual_val <= upper_mu) else 0

            rows_to_insert.append((
                forecast_id,
                state,
                dt_str,
                idx + 1 + horizon_offset,
                model_name,
                pred_mu,
                lower_mu,
                upper_mu,
                temp_c,
                cdd_val,
                ext_heat,
                actual_val,
                reconciled_at,
                error_mu,
                abs_err,
                pct_err,
                within_ci
            ))

        with conn:
            conn.executemany("""
                INSERT INTO forecast_ledger (
                    forecast_id, state, target_date, horizon_step, model_name,
                    predicted_demand_mu, lower_bound_mu, upper_bound_mu,
                    temperature_mean_c, cdd, extreme_heat,
                    actual_demand_mu, reconciled_at, error_mu, abs_error_mu, pct_error, within_ci
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows_to_insert)

        return forecast_id
    finally:
        conn.close()


def record_actuals(records: List[Dict[str, Any]], db_path: Optional[Path] = None) -> int:
    """
    Ingest realized actual demand records and reconcile existing forecasts for matching (state, target_date).
    Records format: [{'state': 'Maharashtra', 'target_date': '2026-11-10', 'actual_demand_mu': 412.5}, ...]
    Returns the count of reconciled forecast entries updated.
    """
    init_ledger_db(db_path)
    conn = get_db_connection(db_path)
    now_iso = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    reconciled_count = 0

    try:
        with conn:
            for rec in records:
                state = rec.get("state")
                target_date = rec.get("target_date")
                actual_mu = rec.get("actual_demand_mu")

                if not state or not target_date or actual_mu is None:
                    continue

                actual_val = float(actual_mu)

                # Fetch matching forecasts that haven't been reconciled or need updating
                cursor = conn.execute("""
                    SELECT id, predicted_demand_mu, lower_bound_mu, upper_bound_mu
                    FROM forecast_ledger
                    WHERE state = ? AND target_date = ?
                """, (state, target_date))
                matching_rows = cursor.fetchall()

                for row in matching_rows:
                    row_id = row["id"]
                    pred_mu = float(row["predicted_demand_mu"])
                    lower_mu = float(row["lower_bound_mu"]) if row["lower_bound_mu"] is not None else None
                    upper_mu = float(row["upper_bound_mu"]) if row["upper_bound_mu"] is not None else None

                    error_mu = round(actual_val - pred_mu, 4)
                    abs_err = round(abs(error_mu), 4)
                    pct_err = round((abs_err / actual_val) * 100.0, 2) if actual_val > 0 else 0.0

                    within_ci = None
                    if lower_mu is not None and upper_mu is not None:
                        within_ci = 1 if (lower_mu <= actual_val <= upper_mu) else 0
                    elif lower_mu is not None:
                        within_ci = 1 if (lower_mu <= actual_val) else 0
                    elif upper_mu is not None:
                        within_ci = 1 if (actual_val <= upper_mu) else 0

                    conn.execute("""
                        UPDATE forecast_ledger
                        SET actual_demand_mu = ?,
                            reconciled_at = ?,
                            error_mu = ?,
                            abs_error_mu = ?,
                            pct_error = ?,
                            within_ci = ?
                        WHERE id = ?
                    """, (actual_val, now_iso, error_mu, abs_err, pct_err, within_ci, row_id))
                    reconciled_count += 1

        return reconciled_count
    finally:
        conn.close()


def get_verification_metrics(
    state: Optional[str] = None,
    days_window: int = 30,
    db_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Compute aggregate verification and accuracy metrics over reconciled forecasts.
    Returns MAE, RMSE, MAPE (%), 90% CI Empirical Coverage (%), and Mean Bias.
    """
    init_ledger_db(db_path)
    conn = get_db_connection(db_path)

    try:
        # Build query filters
        where_clauses = ["actual_demand_mu IS NOT NULL"]
        params: List[Any] = []

        if state:
            where_clauses.append("state = ?")
            params.append(state)

        if days_window > 0:
            where_clauses.append("created_at >= datetime('now', ?)")
            params.append(f"-{days_window} days")

        where_str = " AND ".join(where_clauses)

        # Count total logged regardless of reconciliation
        total_logged_cursor = conn.execute(
            "SELECT COUNT(*) as cnt FROM forecast_ledger WHERE 1=1" +
            (" AND state = ?" if state else ""),
            [state] if state else []
        )
        total_logged = total_logged_cursor.fetchone()["cnt"]

        # Aggregate metrics over reconciled rows
        stats_query = f"""
            SELECT 
                COUNT(*) as reconciled_count,
                AVG(abs_error_mu) as mae,
                AVG(error_mu * error_mu) as mse,
                AVG(pct_error) as mape,
                AVG(error_mu) as bias,
                SUM(CASE WHEN within_ci = 1 THEN 1.0 ELSE 0.0 END) as ci_hits,
                COUNT(within_ci) as ci_total
            FROM forecast_ledger
            WHERE {where_str}
        """
        row = conn.execute(stats_query, params).fetchone()

        reconciled_count = row["reconciled_count"] or 0
        if reconciled_count == 0:
            return {
                "state": state,
                "days_window": days_window,
                "total_predictions_logged": total_logged,
                "reconciled_count": 0,
                "mae_mu": None,
                "rmse_mu": None,
                "mape_pct": None,
                "mean_bias_mu": None,
                "ci_coverage_pct": None
            }

        mae = round(float(row["mae"]), 2) if row["mae"] is not None else 0.0
        mse = float(row["mse"]) if row["mse"] is not None else 0.0
        rmse = round(float(mse ** 0.5), 2)
        mape = round(float(row["mape"]), 2) if row["mape"] is not None else 0.0
        bias = round(float(row["bias"]), 2) if row["bias"] is not None else 0.0

        ci_total = row["ci_total"] or 0
        ci_hits = row["ci_hits"] or 0
        ci_coverage = round(float((ci_hits / ci_total) * 100.0), 1) if ci_total > 0 else None

        return {
            "state": state,
            "days_window": days_window,
            "total_predictions_logged": total_logged,
            "reconciled_count": reconciled_count,
            "mae_mu": mae,
            "rmse_mu": rmse,
            "mape_pct": mape,
            "mean_bias_mu": bias,
            "ci_coverage_pct": ci_coverage
        }
    finally:
        conn.close()


def query_ledger(
    state: Optional[str] = None,
    limit: int = 100,
    reconciled_only: bool = False,
    db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve historical forecast ledger records with optional state filtering.
    """
    init_ledger_db(db_path)
    conn = get_db_connection(db_path)

    try:
        where_clauses = []
        params: List[Any] = []

        if state:
            where_clauses.append("state = ?")
            params.append(state)

        if reconciled_only:
            where_clauses.append("actual_demand_mu IS NOT NULL")

        where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        query = f"""
            SELECT 
                id, forecast_id, created_at, state, target_date, horizon_step, model_name,
                predicted_demand_mu, lower_bound_mu, upper_bound_mu,
                temperature_mean_c, cdd, extreme_heat,
                actual_demand_mu, reconciled_at, error_mu, abs_error_mu, pct_error, within_ci
            FROM forecast_ledger
            {where_str}
            ORDER BY id DESC
            LIMIT ?
        """
        params.append(limit)

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for r in rows:
            results.append({
                "id": r["id"],
                "forecast_id": r["forecast_id"],
                "created_at": r["created_at"],
                "state": r["state"],
                "target_date": r["target_date"],
                "horizon_step": r["horizon_step"],
                "model_name": r["model_name"],
                "predicted_demand_mu": r["predicted_demand_mu"],
                "lower_bound_mu": r["lower_bound_mu"],
                "upper_bound_mu": r["upper_bound_mu"],
                "temperature_mean_c": r["temperature_mean_c"],
                "cdd": r["cdd"],
                "extreme_heat": r["extreme_heat"],
                "actual_demand_mu": r["actual_demand_mu"],
                "reconciled_at": r["reconciled_at"],
                "error_mu": r["error_mu"],
                "abs_error_mu": r["abs_error_mu"],
                "pct_error": r["pct_error"],
                "within_ci": bool(r["within_ci"]) if r["within_ci"] is not None else None
            })
        return results
    finally:
        conn.close()
