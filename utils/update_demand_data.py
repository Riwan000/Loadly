"""
Utility pipeline to ingest parsed Grid-India PSP records and update
data/daily_energy_met_MU.csv, data/14983362/daily_energy_met_MU.csv,
data/State_mu/state_mu_1.csv, data/State_mu/state_mu_2.csv, and data/processed/long_state_mu2.csv.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.parse_psp_reports import parse_all_psp_files

ROOT_DIR = Path(__file__).resolve().parent.parent
DEMAND_CSV_1 = ROOT_DIR / "data" / "daily_energy_met_MU.csv"
DEMAND_CSV_2 = ROOT_DIR / "data" / "14983362" / "daily_energy_met_MU.csv"
STATE_MU_1_PATH = ROOT_DIR / "data" / "State_mu" / "state_mu_1.csv"
STATE_MU_2_PATH = ROOT_DIR / "data" / "State_mu" / "state_mu_2.csv"
LONG_STATE_MU2_PATH = ROOT_DIR / "data" / "processed" / "long_state_mu2.csv"


def update_demand_csvs(df_wide_new: pd.DataFrame, df_long_demand_new: pd.DataFrame = None, df_long_gen_new: pd.DataFrame = None):
    """
    Merge newly parsed wide demand DataFrame into daily_energy_met_MU.csv files.
    """
    if df_wide_new is None or df_wide_new.empty:
        print("⚠️ No new demand records to merge.")
        return

    # 1. Update daily_energy_met_MU.csv (Wide format)
    if DEMAND_CSV_1.exists():
        df_orig = pd.read_csv(DEMAND_CSV_1)
        orig_cols = list(df_orig.columns)
        print(f"Original {DEMAND_CSV_1.name}: {len(df_orig)} rows ({df_orig['Date'].min()} to {df_orig['Date'].max()})")

        df_new = df_wide_new.copy()
        df_new['Date'] = pd.to_datetime(df_new['Date']).dt.strftime('%Y-%m-%d')

        parsed_total = df_new['Total'] if 'Total' in df_new.columns else pd.Series(np.nan, index=df_new.index)
        for col in orig_cols:
            if col not in df_new.columns:
                df_new[col] = np.nan

        # Prefer the report's own All-India total; only fall back to summing columns when absent.
        state_cols = [c for c in orig_cols if c not in ['Date', 'Total']]
        summed = df_new[state_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
        df_new['Total'] = pd.to_numeric(parsed_total, errors='coerce').fillna(summed)

        df_orig_filtered = df_orig[~df_orig['Date'].isin(df_new['Date'])]
        combined_wide = pd.concat([df_orig_filtered, df_new[orig_cols]], ignore_index=True)
        combined_wide['Date'] = pd.to_datetime(combined_wide['Date'])
        combined_wide = combined_wide.sort_values('Date').drop_duplicates(subset=['Date'])
        combined_wide['Date'] = combined_wide['Date'].dt.strftime('%Y-%m-%d')

        combined_wide.to_csv(DEMAND_CSV_1, index=False)
        print(f"✓ Updated {DEMAND_CSV_1}: {len(combined_wide):,} rows ({combined_wide['Date'].min()} to {combined_wide['Date'].max()})")

        if DEMAND_CSV_2.parent.exists():
            combined_wide.to_csv(DEMAND_CSV_2, index=False)
            print(f"✓ Updated {DEMAND_CSV_2}: {len(combined_wide):,} rows")

    # 2. Update State_mu_1.csv (Long format state demand)
    if df_long_demand_new is not None and not df_long_demand_new.empty and STATE_MU_1_PATH.exists():
        df_sm1 = pd.read_csv(STATE_MU_1_PATH)
        df_sm1['date'] = pd.to_datetime(df_sm1['date']).dt.strftime('%Y-%m-%d')
        
        df_l_new = df_long_demand_new.copy()
        df_l_new['date'] = pd.to_datetime(df_l_new['date']).dt.strftime('%Y-%m-%d')

        sm1_combined = pd.concat([df_sm1, df_l_new], ignore_index=True)
        sm1_combined = sm1_combined.drop_duplicates(subset=['date', 'States'], keep='last')
        sm1_combined = sm1_combined.sort_values(['date', 'States'])
        sm1_combined.to_csv(STATE_MU_1_PATH, index=False)
        print(f"✓ Updated {STATE_MU_1_PATH}: {len(sm1_combined):,} rows ({sm1_combined['date'].min()} to {sm1_combined['date'].max()})")

    # 3. Update State_mu_2.csv and long_state_mu2.csv (Regional generation by source)
    if df_long_gen_new is not None and not df_long_gen_new.empty and STATE_MU_2_PATH.exists():
        df_sm2 = pd.read_csv(STATE_MU_2_PATH)
        df_sm2['date'] = pd.to_datetime(df_sm2['date']).dt.strftime('%Y-%m-%d')

        df_g_new = df_long_gen_new.copy()
        df_g_new['date'] = pd.to_datetime(df_g_new['date']).dt.strftime('%Y-%m-%d')

        # Align columns
        cols_sm2 = list(df_sm2.columns)
        for c in cols_sm2:
            if c not in df_g_new.columns:
                df_g_new[c] = np.nan

        sm2_combined = pd.concat([df_sm2, df_g_new[cols_sm2]], ignore_index=True)
        sm2_combined = sm2_combined.drop_duplicates(subset=['date', 'source'], keep='last')
        sm2_combined = sm2_combined.sort_values(['date', 'source'])
        sm2_combined.to_csv(STATE_MU_2_PATH, index=False)
        print(f"✓ Updated {STATE_MU_2_PATH}: {len(sm2_combined):,} rows ({sm2_combined['date'].min()} to {sm2_combined['date'].max()})")

        # Update long_state_mu2.csv
        if LONG_STATE_MU2_PATH.parent.exists():
            df_sm2_long = sm2_combined.melt(
                id_vars=["source", "date"],
                value_vars=["NR", "WR", "SR", "ER", "NER", "All India"],
                var_name="region",
                value_name="regional_generation_mu"
            )
            df_sm2_long['region'] = df_sm2_long['region'].str.lower()
            df_sm2_long.to_csv(LONG_STATE_MU2_PATH, index=False)
            print(f"✓ Updated {LONG_STATE_MU2_PATH}: {len(df_sm2_long):,} rows")


def run_pipeline():
    """Execute complete demand & generation ingestion pipeline from data/psp/."""
    print("=== Running Electricity Demand & Generation Ingestion Pipeline ===")
    df_wide, df_long_demand, df_long_gen = parse_all_psp_files()
    update_demand_csvs(df_wide, df_long_demand, df_long_gen)
    print("\n✓ Ingestion Pipeline Completed.")


if __name__ == "__main__":
    run_pipeline()
