"""
End-to-End Prediction Verification Lifecycle Demo.
Demonstrates the full closed loop:
1. Forecast Issued (Pending Status)
2. Ground-Truth Data Ingested
3. Forecast Verified & Audited (MAPE, Error MU, 90% CI Pass/Fail)

Run with:
    python verify_prediction_lifecycle.py
"""

from datetime import date, timedelta
from utils.prediction_ledger import record_forecast_batch, query_ledger, get_verification_metrics
from utils.ingestion_worker import run_ingestion_pipeline

def run_lifecycle_demo():
    print("\n" + "=" * 75)
    print("⚡ LOADLY — CLOSED-LOOP PREDICTION VERIFICATION DEMO")
    print("=" * 75)

    yesterday = date.today() - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")

    # -------------------------------------------------------------
    # Step 1: Simulate Forecast Generation on Day t
    # -------------------------------------------------------------
    print(f"\n[STEP 1] Generating Forward Forecast for Target Date: {yesterday_str}...")
    sample_timeline = [
        {
            "date": yesterday_str,
            "forecasted_demand_mu": 412.0,
            "lower_bound_mu": 390.0,
            "upper_bound_mu": 434.0,
            "temperature_mean_c": 31.5,
            "cdd": 7.35,
            "extreme_heat": 1
        }
    ]
    
    forecast_id = record_forecast_batch(
        sample_timeline, 
        state="Maharashtra", 
        model_name="Reduced_LR", 
        auto_reconcile=False
    )
    print(f"  • Forecast ID:        {forecast_id}")
    print(f"  • State:              Maharashtra")
    print(f"  • Predicted Demand:   412.00 MU")
    print(f"  • 90% Confidence CI:  [390.00 MU — 434.00 MU]")
    
    # Check ledger status before verification
    pre_audit = query_ledger(state="Maharashtra", limit=1)[0]
    print(f"  • Status in Ledger:   ⏳ PENDING ACTUAL (Actual: {pre_audit['actual_demand_mu']}, Error: {pre_audit['error_mu']})")

    # -------------------------------------------------------------
    # Step 2: The Next Morning — Ingestion Worker Runs
    # -------------------------------------------------------------
    print(f"\n[STEP 2] Running Ingestion Worker for {yesterday_str}...")
    print("  • Fetching realized grid consumption & weather reality...")
    summary = run_ingestion_pipeline(target_date=yesterday, catchup_days=1)
    print(f"  • Total Actuals Ingested across India: {summary['total_actuals_ingested']}")
    print(f"  • Forecasts Reconciled in Ledger:      {summary['reconciled_forecasts']}")

    # -------------------------------------------------------------
    # Step 3: Verification & Audit in Prediction Ledger
    # -------------------------------------------------------------
    print(f"\n[STEP 3] Audited Ledger Verification Result:")
    post_audit = query_ledger(state="Maharashtra", limit=1)[0]
    
    pred_val = post_audit["predicted_demand_mu"]
    actual_val = post_audit["actual_demand_mu"]
    error_val = post_audit["error_mu"]
    mape_val = post_audit["pct_error"]
    ci_status = "✅ PASS (Within 90% CI)" if post_audit["within_ci"] else "❌ DIVERGENT (Outside CI)"

    print(f"  • State:               {post_audit['state']}")
    print(f"  • Target Date:         {post_audit['target_date']}")
    print(f"  • Model Prediction:    {pred_val:.2f} MU")
    print(f"  • Realized Actual:     {actual_val:.2f} MU")
    print(f"  • Realized Error:      {error_val:+.2f} MU")
    print(f"  • Realized MAPE (%):   {mape_val:.2f}%")
    print(f"  • 90% CI Audit Status: {ci_status}")
    print(f"  • Reconciled At:       {post_audit['reconciled_at']}")

    # -------------------------------------------------------------
    # Step 4: Overall Verification Health
    # -------------------------------------------------------------
    print(f"\n[STEP 4] Updated Platform Verification Metrics:")
    metrics = get_verification_metrics(days_window=0)
    print(f"  • Total Predictions Logged: {metrics['total_predictions_logged']}")
    print(f"  • Verified & Audited:       {metrics['reconciled_count']}")
    print(f"  • Overall Realized MAPE:    {metrics['mape_pct']}%")
    print(f"  • Overall Realized RMSE:    {metrics['rmse_mu']} MU")
    print(f"  • 90% CI Empirical Hit Rate:{metrics['ci_coverage_pct']}%")
    print(f"  • Mean Forecast Bias:       {metrics['mean_bias_mu']:+.2f} MU")

    print("\n" + "=" * 75)
    print("✅ PREDICTION VERIFICATION COMPLETE: CLOSED-LOOP AUDIT VERIFIED")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    run_lifecycle_demo()
