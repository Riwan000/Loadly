"""
Continuous Data Ingestion & Forecast Verification Worker for Loadly.
Reads ground-truth actual electricity demand from the local historical grid CSV to
verify past predictions in the ledger. Dates with no real data are skipped and reported;
actuals are never estimated or synthesized.

Usage:
    python utils/ingestion_worker.py                  # Ingest yesterday's actuals
    python utils/ingestion_worker.py --catchup 7       # Reconcile last 7 days
    python utils/ingestion_worker.py --date 2026-11-14 # Specific date
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import ALL_STATES
from utils.prediction_ledger import record_actuals, get_verification_metrics, init_ledger_db

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEMAND_CACHE_PATH = ROOT_DIR / "data" / "daily_energy_met_MU.csv"
CLIMATOLOGY_PATH = ROOT_DIR / "artifacts" / "climatology_reference.json"

# A CSV row with fewer states than this is considered too sparse to be a real day of data.
MIN_STATES_FOR_ACTUALS = 20

STATE_SYNONYMS = {
    "Delhi": "NCT of Delhi",
    "J&K": "Jammu and Kashmir",
    "Jammu & Kashmir": "Jammu and Kashmir",
    "J&K(UT)": "Jammu and Kashmir",
    "Pondicherry": "Puducherry",
    "Pondy": "Puducherry",
    "Andaman and Nicobar Islands": "Andaman and Nicobar",
    "DNH": "Dadra and Nagar Haveli",
    "D&D": "Daman and Diu",
    "DNHDDPDCL": "Dadra and Nagar Haveli"
}


def normalize_state_name(raw_name: str) -> str:
    """Normalize raw state name from external reports to canonical ALL_STATES standard."""
    cleaned = raw_name.strip()
    if cleaned in ALL_STATES:
        return cleaned
    if cleaned in STATE_SYNONYMS:
        return STATE_SYNONYMS[cleaned]
    for standard_state in ALL_STATES:
        if standard_state.lower() == cleaned.lower():
            return standard_state
    return cleaned


# Column used by earlier versions of this worker to persist synthetic national
# aggregates into the demand CSV. Real source rows never carry it, so a row with a
# value here is treated as fabricated and is never served as an actual.
_SYNTHETIC_ROW_MARKER_COL = "All-India"
# Genuine national total column shipped with the historical source data.
_NATIONAL_TOTAL_COL = "Total"

_CSV_STATE_ALIASES = {
    canonical: [raw for raw, canon in STATE_SYNONYMS.items() if canon == canonical]
    for canonical in set(STATE_SYNONYMS.values())
}


def _csv_state_value(row: "pd.Series", state: str) -> Optional[float]:
    """First non-null value for `state` under its canonical name or known CSV aliases."""
    for col in [state, *_CSV_STATE_ALIASES.get(state, [])]:
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
    return None


def fetch_actual_grid_demand(target_date: date) -> List[Dict[str, Any]]:
    """
    Return realized electricity consumption (MU) for target_date from the local
    historical grid CSV. This is the only source of actuals: if the date is not
    there (or the row is untrustworthy) an empty list is returned and a warning is
    logged. Demand is never estimated or synthesized.

    The All-India record uses the CSV's genuine "Total" column. It is never a sum of
    whatever states happen to be present, because real rows routinely lack some
    states (e.g. Lakshadweep), which would understate the national figure.
    """
    date_str = target_date.strftime("%Y-%m-%d")

    if not DEMAND_CACHE_PATH.exists():
        logger.warning(f"No actual grid demand for {date_str}: {DEMAND_CACHE_PATH} not found.")
        return []

    try:
        df = pd.read_csv(DEMAND_CACHE_PATH)
    except Exception as e:
        logger.warning(f"No actual grid demand for {date_str}: could not read demand CSV ({e}).")
        return []

    date_col = "Date" if "Date" in df.columns else "date"
    if date_col not in df.columns:
        logger.warning(f"No actual grid demand for {date_str}: demand CSV has no date column.")
        return []

    dt_str = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    matching = df[dt_str == date_str]
    if matching.empty:
        logger.warning(f"No actual grid demand available for {date_str}; skipping (nothing recorded).")
        return []

    row = matching.iloc[0]
    if _SYNTHETIC_ROW_MARKER_COL in row.index and pd.notna(row[_SYNTHETIC_ROW_MARKER_COL]):
        logger.warning(
            f"Demand CSV row for {date_str} was written by the legacy synthetic ingestion path; "
            "ignoring it as an actual."
        )
        return []

    results: List[Dict[str, Any]] = []
    for state in ALL_STATES:
        val = _csv_state_value(row, state)
        if val is not None:
            results.append({
                "state": state,
                "target_date": date_str,
                "actual_demand_mu": round(val, 2),
                "source": "historical_csv",
            })

    if len(results) < MIN_STATES_FOR_ACTUALS:
        logger.warning(
            f"Only {len(results)} states in CSV for {date_str} (< {MIN_STATES_FOR_ACTUALS}); "
            "treating the date as having no actuals."
        )
        return []

    total = row[_NATIONAL_TOTAL_COL] if _NATIONAL_TOTAL_COL in row.index else None
    if total is not None and pd.notna(total):
        results.append({
            "state": "All-India",
            "target_date": date_str,
            "actual_demand_mu": round(float(total), 2),
            "source": "national_total_csv",
        })
    else:
        logger.warning(f"No national total for {date_str}; All-India actual not recorded.")

    return results


def run_ingestion_pipeline(
    target_date: Optional[date] = None,
    catchup_days: int = 1
) -> Dict[str, Any]:
    """
    Execute end-to-end continuous ingestion:
    1. For each target date, fetch state electricity actuals.
    2. Reconcile matching rows in prediction_ledger.db.
    3. Dates with no real actuals are skipped and reported in `dates_missing`.
    4. Compute updated verification accuracy metrics (MAPE %, CI Hit Rate).
    """
    init_ledger_db()
    
    start_dt = target_date if target_date else (date.today() - timedelta(days=1))
    dates_to_process = [start_dt - timedelta(days=i) for i in range(catchup_days)]
    
    total_reconciled = 0
    all_ingested_records = []
    dates_missing: List[str] = []

    for d in reversed(dates_to_process):
        date_str = d.strftime("%Y-%m-%d")
        logger.info(f"Ingesting grid verification actuals for {date_str}...")

        actuals = fetch_actual_grid_demand(d)
        if not actuals:
            logger.warning(f" -> No real actuals for {date_str}; nothing reconciled or cached.")
            dates_missing.append(date_str)
            continue
        all_ingested_records.extend(actuals)

        # Reconcile Prediction Ledger
        reconciled = record_actuals(actuals)
        total_reconciled += reconciled
        logger.info(f" -> Reconciled {reconciled} forecast records for {date_str}")

        # The demand CSV is the only actuals source; ingestion never writes to it.

    # Compute latest overall verification metrics
    metrics = get_verification_metrics(days_window=0)

    summary = {
        "status": "success",
        "dates_processed": [d.strftime("%Y-%m-%d") for d in dates_to_process],
        "states_per_day": len(ALL_STATES),
        "dates_missing": dates_missing,
        "total_actuals_ingested": len(all_ingested_records),
        "reconciled_forecasts": total_reconciled,
        "verification_metrics": metrics,
        "executed_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Loadly Continuous Data Ingestion & Verification Worker")
    parser.add_argument("--date", type=str, help="Specific date to ingest (YYYY-MM-DD)")
    parser.add_argument("--catchup", type=int, default=1, help="Number of past days to catch up on (default: 1)")
    parser.add_argument("--state", type=str, help="Optional state filter for summary display")

    args = parser.parse_args()

    target_dt = None
    if args.date:
        try:
            target_dt = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"❌ Invalid date format: {args.date}. Use YYYY-MM-DD.")
            sys.exit(1)

    print("\n" + "=" * 65)
    print("⚡ Loadly - Continuous Data Ingestion & Verification Worker")
    print("=" * 65)

    summary = run_ingestion_pipeline(target_date=target_dt, catchup_days=args.catchup)
    metrics = summary["verification_metrics"]

    print("\n✅ Ingestion & Reconciliation Complete!")
    print(f"• Dates Processed:        {', '.join(summary['dates_processed'])}")
    if summary["dates_missing"]:
        print(f"• Dates With No Data:     {', '.join(summary['dates_missing'])}")
    print(f"• Total Actuals Ingested:  {summary['total_actuals_ingested']}")
    print(f"• Forecasts Reconciled:   {summary['reconciled_forecasts']}")
    print("-" * 65)
    print("📊 Latest Verified Ledger Performance:")
    print(f"• Total Predictions Logged: {metrics.get('total_predictions_logged')}")
    print(f"• Reconciled Actuals:       {metrics.get('reconciled_count')}")
    print(f"• Realized MAPE (%):        {metrics.get('mape_pct')}%")
    print(f"• Realized RMSE:            {metrics.get('rmse_mu')} MU")
    print(f"• 90% CI Hit Rate:          {metrics.get('ci_coverage_pct')}%")
    print(f"• Mean Bias:                {metrics.get('mean_bias_mu')} MU")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
