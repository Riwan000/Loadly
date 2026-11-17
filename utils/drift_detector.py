"""
Model Drift Detection Engine for Loadly.
Monitors operational forecast performance against ground-truth actuals stored in the prediction ledger.
Evaluates 4 core dimensions:
1. Trailing Window Performance Decay (MAPE)
2. Directional Bias Skew (Mean Bias Error & Z-score)
3. Confidence Interval Hit Rate Degradation (Empirical Coverage)
4. Covariate / Feature Shift (Population Stability Index)
"""

import math
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

from utils.prediction_ledger import get_db_connection, init_ledger_db, DEFAULT_DB_PATH
from models import load_model_metadata, get_state_rmse

logger = logging.getLogger(__name__)

# Default Operational Drift Thresholds
HEALTHY_MAPE_THRESHOLD = 3.5      # % error
WARNING_MAPE_THRESHOLD = 5.0      # % error
SIGNIFICANT_BIAS_Z = 1.96         # 95% confidence interval for zero-mean bias
CRITICAL_BIAS_Z = 3.00            # 99.7% confidence interval
MIN_NOMINAL_CI_COVERAGE = 80.0    # % of actuals within 90% confidence band
CRITICAL_CI_COVERAGE = 70.0       # Severe coverage degradation


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """
    Calculate the Population Stability Index (PSI) between baseline and production distributions.
    
    PSI < 0.10: No significant shift
    0.10 <= PSI < 0.25: Moderate shift
    PSI >= 0.25: Severe distribution drift
    """
    exp_clean = expected[~np.isnan(expected)]
    act_clean = actual[~np.isnan(actual)]
    
    if len(exp_clean) < 10 or len(act_clean) < 10:
        return 0.0

    if float(np.std(exp_clean)) < 1e-3 or len(np.unique(exp_clean)) < 3:
        return 0.0

    # Determine quantile bins based on the baseline/expected distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.unique(np.percentile(exp_clean, percentiles))
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Bucket counts
    exp_counts, _ = np.histogram(exp_clean, bins=bin_edges)
    act_counts, _ = np.histogram(act_clean, bins=bin_edges)

    # Convert to fractions with Laplace smoothing to avoid division by zero or log(0)
    k = len(exp_counts)
    eps = 1e-4
    exp_pct = (exp_counts + eps) / (len(exp_clean) + eps * k)
    act_pct = (act_counts + eps) / (len(act_clean) + eps * k)

    # Compute PSI
    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(max(0.0, psi_val))


def compute_drift_metrics(
    state: Optional[str] = None,
    window_days: int = 14,
    db_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Compute real-time drift metrics from verified prediction ledger records.
    
    Parameters:
        state: Optional state name ('Maharashtra', 'All-India', None for all).
        window_days: Trailing window of days to analyze (default 14 days).
        db_path: Custom SQLite database path.
        
    Returns:
        Structured drift assessment dictionary including metrics, status, and recommendations.
    """
    init_ledger_db(db_path)
    conn = get_db_connection(db_path)
    
    metadata = load_model_metadata() or {}
    baseline_mape = float(metadata.get("mae", 3.21) / 200.0 * 100.0) if metadata else 1.68
    if baseline_mape > 3.0:
        baseline_mape = 1.68

    baseline_rmse = float(metadata.get("rmse", 5.36)) if metadata else 5.36
    if state and state.strip().lower() in ("all-india", "national"):
        baseline_rmse = 32.16  # National aggregate portfolio uncertainty (~sqrt(36) * state_rmse)
        baseline_mape = 1.68
    elif state and state.strip().lower() not in ("all", "all grid"):
        s_rmse = get_state_rmse(state.strip(), metadata)
        if s_rmse:
            baseline_rmse = s_rmse

    try:
        where_clauses = ["actual_demand_mu IS NOT NULL"]
        params: List[Any] = []

        if state:
            clean_state = state.strip()
            if clean_state.lower() in ("all-india", "national"):
                where_clauses.append("state IN ('All-India', 'National')")
            elif clean_state.lower() not in ("all", "all grid"):
                where_clauses.append("state = ?")
                params.append(clean_state)

        # Apply trailing date window if applicable
        cutoff_date = (datetime.now() - timedelta(days=window_days)).strftime("%Y-%m-%d")
        where_clauses.append("target_date >= ?")
        params.append(cutoff_date)

        where_sql = " AND ".join(where_clauses)
        query = f"""
            SELECT 
                target_date, state, predicted_demand_mu, lower_bound_mu, upper_bound_mu,
                actual_demand_mu, error_mu, abs_error_mu, pct_error, within_ci,
                temperature_mean_c, cdd
            FROM forecast_ledger
            WHERE {where_sql}
            ORDER BY target_date ASC
        """
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()

        # If not enough rows in recent window, query all reconciled without date cutoff
        if len(rows) < 5:
            fallback_where = ["actual_demand_mu IS NOT NULL"]
            fallback_params: List[Any] = []
            if state:
                clean_state = state.strip()
                if clean_state.lower() in ("all-india", "national"):
                    fallback_where.append("state IN ('All-India', 'National')")
                elif clean_state.lower() not in ("all", "all grid"):
                    fallback_where.append("state = ?")
                    fallback_params.append(clean_state)
            
            fb_sql = " AND ".join(fallback_where)
            cursor = conn.execute(f"""
                SELECT 
                    target_date, state, predicted_demand_mu, lower_bound_mu, upper_bound_mu,
                    actual_demand_mu, error_mu, abs_error_mu, pct_error, within_ci,
                    temperature_mean_c, cdd
                FROM forecast_ledger
                WHERE {fb_sql}
                ORDER BY target_date ASC
            """, fallback_params)
            rows = cursor.fetchall()

        total_samples = len(rows)
        
        if total_samples == 0:
            return {
                "scope": state or "All Grid",
                "status": "INSUFFICIENT_DATA",
                "sample_count": 0,
                "window_days": window_days,
                "metrics": {
                    "mape_pct": None,
                    "mae_mu": None,
                    "rmse_mu": None,
                    "mean_bias_mu": None,
                    "bias_z_score": 0.0,
                    "ci_coverage_pct": None,
                    "baseline_mape_pct": baseline_mape,
                    "baseline_rmse_mu": baseline_rmse
                },
                "feature_drift": {
                    "temperature_psi": 0.0,
                    "cdd_psi": 0.0
                },
                "diagnostics": ["No reconciled records found yet. Trigger actuals ingestion to verify forecasts."],
                "recommended_action": "INGEST_ACTUALS"
            }

        # Calculate metrics from rows
        actuals = np.array([float(r["actual_demand_mu"]) for r in rows])
        preds = np.array([float(r["predicted_demand_mu"]) for r in rows])
        residuals = actuals - preds  # Positive = under-predicting, Negative = over-predicting
        abs_errors = np.abs(residuals)
        pct_errors = np.abs(residuals / np.maximum(actuals, 1e-4)) * 100.0

        within_ci_list = []
        for r in rows:
            act = float(r["actual_demand_mu"])
            lb = float(r["lower_bound_mu"]) if r["lower_bound_mu"] is not None else -np.inf
            ub = float(r["upper_bound_mu"]) if r["upper_bound_mu"] is not None else np.inf
            within_ci_list.append(1 if lb <= act <= ub else 0)

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        mape = float(np.mean(pct_errors))
        bias = float(np.mean(residuals))
        std_residuals = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

        # Bias Z-score
        if total_samples > 1 and std_residuals > 1e-5:
            bias_z_score = float(bias / (std_residuals / math.sqrt(total_samples)))
        else:
            bias_z_score = 0.0

        ci_coverage = float(np.mean(within_ci_list) * 100.0)

        # PSI computation on temperature and CDD if records have them
        temps = [float(r["temperature_mean_c"]) for r in rows if r["temperature_mean_c"] is not None]
        temp_psi = 0.0
        cdd_psi = 0.0
        if len(temps) >= 14:
            mid = len(temps) // 2
            temp_psi = calculate_psi(np.array(temps[:mid]), np.array(temps[mid:]))
        
        cdds = [float(r["cdd"]) for r in rows if r["cdd"] is not None]
        if len(cdds) >= 14:
            mid = len(cdds) // 2
            cdd_psi = calculate_psi(np.array(cdds[:mid]), np.array(cdds[mid:]))

        # Drift Status Classification
        diagnostics = []
        is_critical = False
        is_warning = False

        if total_samples < 5:
            status = "INSUFFICIENT_DATA"
            diagnostics.append(f"Only {total_samples} verified samples available. At least 5 required for high-confidence statistical drift test.")
            recommended_action = "MONITOR"
        else:
            # Check MAPE decay
            if mape >= WARNING_MAPE_THRESHOLD:
                is_critical = True
                diagnostics.append(f"Severe MAPE degradation ({mape:.2f}% vs baseline {baseline_mape:.2f}%).")
            elif mape >= HEALTHY_MAPE_THRESHOLD:
                is_warning = True
                diagnostics.append(f"Moderate MAPE degradation ({mape:.2f}% vs baseline {baseline_mape:.2f}%).")

            # Check Bias skew
            if abs(bias_z_score) >= CRITICAL_BIAS_Z:
                is_critical = True
                direction = "under-predicting (shortfall risk)" if bias > 0 else "over-predicting (curtailment risk)"
                diagnostics.append(f"Critical systematic directional bias: z={bias_z_score:.2f}, mean bias={bias:+.2f} MU ({direction}).")
            elif abs(bias_z_score) >= SIGNIFICANT_BIAS_Z:
                is_warning = True
                direction = "under-predicting" if bias > 0 else "over-predicting"
                diagnostics.append(f"Statistically significant bias detected: z={bias_z_score:.2f}, mean bias={bias:+.2f} MU ({direction}).")

            # Check CI coverage
            if ci_coverage < CRITICAL_CI_COVERAGE:
                is_critical = True
                diagnostics.append(f"Critical confidence interval coverage decay: {ci_coverage:.1f}% (expected >= 90%).")
            elif ci_coverage < MIN_NOMINAL_CI_COVERAGE:
                is_warning = True
                diagnostics.append(f"Uncertainty coverage degradation: {ci_coverage:.1f}% (expected >= 90%).")

            # Check Feature PSI
            if temp_psi >= 0.25:
                is_warning = True
                diagnostics.append(f"Significant temperature distribution shift detected (PSI={temp_psi:.3f}).")

            if is_critical:
                status = "CRITICAL"
                recommended_action = "RETRAIN_RECOMMENDED"
            elif is_warning:
                status = "WARNING"
                recommended_action = "CALIBRATE_RECOMMENDED"
            else:
                status = "HEALTHY"
                recommended_action = "NONE"
                diagnostics.append(f"Model operating nominally. Trailing MAPE is {mape:.2f}% (baseline {baseline_mape:.2f}%), CI coverage {ci_coverage:.1f}%.")

        return {
            "scope": state or "All Grid",
            "status": status,
            "sample_count": total_samples,
            "window_days": window_days,
            "metrics": {
                "mape_pct": round(mape, 2),
                "mae_mu": round(mae, 2),
                "rmse_mu": round(rmse, 2),
                "mean_bias_mu": round(bias, 2),
                "bias_z_score": round(bias_z_score, 2),
                "ci_coverage_pct": round(ci_coverage, 1),
                "baseline_mape_pct": round(baseline_mape, 2),
                "baseline_rmse_mu": round(baseline_rmse, 2)
            },
            "feature_drift": {
                "temperature_psi": round(temp_psi, 3),
                "cdd_psi": round(cdd_psi, 3)
            },
            "diagnostics": diagnostics,
            "recommended_action": recommended_action
        }
    finally:
        conn.close()
