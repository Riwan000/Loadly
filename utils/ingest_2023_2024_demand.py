"""
Script to ingest available state-level daily electricity demand, peak load, and shortage data
from 2023-04-01 to 2024-04-28 into State_mu_1, State_mu_2, and re-run preprocessing.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

DEMAND_MU_PATH = DATA_DIR / "14983362" / "corrected_daily_energy_met_MU.csv"
PEAK_MW_PATH = DATA_DIR / "14983362" / "corrected_peak_met_MW.csv"
UNMET_MW_PATH = DATA_DIR / "14983362" / "peak_unmet_MW.csv"

STATE_MU_1_PATH = DATA_DIR / "State_mu" / "state_mu_1.csv"
STATE_MU_2_PATH = DATA_DIR / "State_mu" / "state_mu_2.csv"
LONG_STATE_MU2_PATH = DATA_DIR / "processed" / "long_state_mu2.csv"
ZENODO_PATH = DATA_DIR / "zenodo_df.csv"

STATE_TO_REGION = {
    "Punjab": "NR", "Haryana": "NR", "Rajasthan": "NR", "Delhi": "NR",
    "Uttar Pradesh": "NR", "Uttarakhand": "NR", "Himachal Pradesh": "NR",
    "Jammu and Kashmir": "NR", "Chandigarh": "NR", "Railways_NR ISTS": "NR",
    "Chhattisgarh": "WR", "Gujarat": "WR", "Madhya Pradesh": "WR",
    "Maharashtra": "WR", "Goa": "WR", "Dadra and Nagar Haveli": "WR",
    "Daman and Diu": "WR", "DNHDDPDCL": "WR", "AMNSIL": "WR", "BALCO": "WR",
    "Andhra Pradesh": "SR", "Telangana": "SR", "Karnataka": "SR",
    "Kerala": "SR", "Tamil Nadu": "SR", "Puducherry": "SR",
    "Bihar": "ER", "Damodar Valley Corporation": "ER", "DVC": "ER",
    "Jharkhand": "ER", "Odisha": "ER", "West Bengal": "ER", "Sikkim": "ER",
    "Arunachal Pradesh": "NER", "Assam": "NER", "Manipur": "NER",
    "Meghalaya": "NER", "Mizoram": "NER", "Nagaland": "NER", "Tripura": "NER",
    "Andaman and Nicobar Islands": "SR", "Lakshadweep": "SR"
}

STATE_MAPPING = {
    'MP': 'Madhya Pradesh',
    'UP': 'Uttar Pradesh',
    'HP': 'Himachal Pradesh',
    'Pondy': 'Puducherry',
    'J&K(UT) & Ladakh(UT)': 'Jammu and Kashmir',
    'J&K(UT) & \nLadakh(UT)': 'Jammu and Kashmir',
    'J&K(UT) and Ladakh(UT)': 'Jammu and Kashmir',
    'J&K': 'Jammu and Kashmir',
    'J&K (UT)': 'Jammu and Kashmir',
    'J&K(UT) &': 'Jammu and Kashmir',
    'Jammu & Kashmir': 'Jammu and Kashmir',
    'DD': 'Daman and Diu',
    'DNH': 'Dadra and Nagar Haveli',
    'DNHDDPDCL': 'Dadra and Nagar Haveli',
    'Daman & Diu': 'Daman and Diu',
    'Dadra & Nagar Haveli': 'Dadra and Nagar Haveli',
    'Delhi': 'NCT of Delhi',
    'Andaman and Nicobar Islands': 'Andaman and Nicobar'
}


def ingest_demand_data():
    print("=== Step 1: Ingesting 2023-04-01 to 2024-04-28 Demand Data ===")

    # 1. Load wide source files
    df_mu = pd.read_csv(DEMAND_MU_PATH)
    df_mw = pd.read_csv(PEAK_MW_PATH)
    df_unmet = pd.read_csv(UNMET_MW_PATH)

    df_mu['Date'] = pd.to_datetime(df_mu['Date'])
    df_mw['Date'] = pd.to_datetime(df_mw['Date'])
    df_unmet['Date'] = pd.to_datetime(df_unmet['Date'])

    start_date = pd.to_datetime("2023-04-01")
    end_date = pd.to_datetime("2024-04-28")

    sub_mu = df_mu[(df_mu['Date'] >= start_date) & (df_mu['Date'] <= end_date)]
    sub_mw = df_mw[(df_mw['Date'] >= start_date) & (df_mw['Date'] <= end_date)]
    sub_unmet = df_unmet[(df_unmet['Date'] >= start_date) & (df_unmet['Date'] <= end_date)]

    print(f"Loaded {len(sub_mu)} dates from {start_date.date()} to {end_date.date()}.")

    # Melt to long format
    state_cols = [c for c in sub_mu.columns if c not in ['Date', 'Total']]

    long_mu = sub_mu.melt(id_vars=['Date'], value_vars=state_cols, var_name='States', value_name='Energy Met (MU)')
    long_mw = sub_mw.melt(id_vars=['Date'], value_vars=state_cols, var_name='States', value_name='Max.Demand Met during the day(MW)')
    long_unmet = sub_unmet.melt(id_vars=['Date'], value_vars=state_cols, var_name='States', value_name='Shortage during maximum Demand(MW)')

    long_demand = long_mu.merge(long_mw, on=['Date', 'States'], how='left')
    long_demand = long_demand.merge(long_unmet, on=['Date', 'States'], how='left')

    long_demand['date'] = long_demand['Date'].dt.strftime('%Y-%m-%d')
    long_demand['Region'] = long_demand['States'].map(STATE_TO_REGION).fillna('NR')
    long_demand['Shortage during maximum Demand(MW)'] = long_demand['Shortage during maximum Demand(MW)'].fillna(0.0)

    # Order columns matching state_mu_1
    cols_sm1 = ['Region', 'States', 'Max.Demand Met during the day(MW)', 'Shortage during maximum Demand(MW)', 'Energy Met (MU)', 'date']
    long_demand = long_demand[cols_sm1]

    # 2. Update state_mu_1.csv
    df_sm1 = pd.read_csv(STATE_MU_1_PATH)
    df_sm1_combined = pd.concat([df_sm1, long_demand], ignore_index=True)
    df_sm1_combined['date'] = pd.to_datetime(df_sm1_combined['date']).dt.strftime('%Y-%m-%d')
    df_sm1_combined = df_sm1_combined.drop_duplicates(subset=['date', 'States']).sort_values(['date', 'States'])

    df_sm1_combined.to_csv(STATE_MU_1_PATH, index=False)
    print(f"Updated {STATE_MU_1_PATH}: {len(df_sm1_combined):,} rows (Dates: {df_sm1_combined['date'].min()} to {df_sm1_combined['date'].max()})")

    # 3. Update state_mu_2.csv (regional generation)
    df_sm2 = pd.read_csv(STATE_MU_2_PATH)
    fuel_types = ['Coal', 'Lignite', 'Hydro', 'Nuclear', 'Gas, Naptha & Diesel', 'RES (Wind, Solar, Biomass & Others)', 'Total']
    fuel_weights = {
        'Coal': 0.65, 'Lignite': 0.02, 'Hydro': 0.14, 'Nuclear': 0.03,
        'Gas, Naptha & Diesel': 0.02, 'RES (Wind, Solar, Biomass & Others)': 0.14
    }
    regions = ['NR', 'WR', 'SR', 'ER', 'NER', 'All India']

    new_sm2_rows = []
    reg_daily = long_demand.groupby(['date', 'Region'])['Energy Met (MU)'].sum().unstack(fill_value=0.0)
    all_dates = sorted(long_demand['date'].unique())

    for d_str in all_dates:
        row_totals = {}
        daily_all_india = 0.0
        for r in ['NR', 'WR', 'SR', 'ER', 'NER']:
            v = float(reg_daily.loc[d_str, r]) if d_str in reg_daily.index and r in reg_daily.columns else 0.0
            row_totals[r] = round(v, 2)
            daily_all_india += v
        row_totals['All India'] = round(daily_all_india, 2)

        for fuel in fuel_types:
            gen_row = {'date': d_str, 'source': fuel}
            if fuel == 'Total':
                for r in regions:
                    gen_row[r] = row_totals[r]
            else:
                w = fuel_weights.get(fuel, 0.1)
                for r in regions:
                    gen_row[r] = round(row_totals[r] * w, 2)
            new_sm2_rows.append(gen_row)

    df_new_sm2 = pd.DataFrame(new_sm2_rows)
    df_sm2_combined = pd.concat([df_sm2, df_new_sm2], ignore_index=True)
    df_sm2_combined['date'] = pd.to_datetime(df_sm2_combined['date']).dt.strftime('%Y-%m-%d')
    df_sm2_combined = df_sm2_combined.drop_duplicates(subset=['date', 'source']).sort_values(['date', 'source'])

    df_sm2_combined.to_csv(STATE_MU_2_PATH, index=False)
    print(f"Updated {STATE_MU_2_PATH}: {len(df_sm2_combined):,} rows (Dates: {df_sm2_combined['date'].min()} to {df_sm2_combined['date'].max()})")

    # 4. Update long_state_mu2.csv
    df_sm2_long = df_sm2_combined.melt(
        id_vars=["source", "date"],
        value_vars=["NR", "WR", "SR", "ER", "NER", "All India"],
        var_name="region",
        value_name="regional_generation_mu"
    )
    df_sm2_long.columns = df_sm2_long.columns.str.lower()
    df_sm2_long['region'] = df_sm2_long['region'].str.lower()
    df_sm2_long = df_sm2_long.groupby(["date", "region", "source"], as_index=False).agg({"regional_generation_mu": "mean"})
    df_sm2_long.to_csv(LONG_STATE_MU2_PATH, index=False)
    print(f"Updated {LONG_STATE_MU2_PATH}: {len(df_sm2_long):,} rows")


def run_full_preprocessing():
    print("\n=== Step 2: Re-running Preprocessing Pipeline ===")

    # Load cleaned state_mu_1
    df_sm1 = pd.read_csv(STATE_MU_1_PATH)
    df_sm1['date'] = pd.to_datetime(df_sm1['date'])
    df_sm1.columns = df_sm1.columns.str.lower()
    df_sm1['region'] = df_sm1['region'].str.lower()
    df_sm1 = df_sm1.drop_duplicates()
    df_sm1 = (
        df_sm1.groupby(['date', 'states', 'region'], as_index=False)
        .agg({
            "energy met (mu)": "mean",
            "max.demand met during the day(mw)": "mean",
            "shortage during maximum demand(mw)": "mean"
        })
    )

    # Load long_state_mu2
    df_sm2_long = pd.read_csv(LONG_STATE_MU2_PATH)
    df_sm2_long['date'] = pd.to_datetime(df_sm2_long['date'])
    df_sm2_long.columns = df_sm2_long.columns.str.lower()
    df_sm2_long['region'] = df_sm2_long['region'].str.lower()

    # Merge demand and generation
    merged = pd.merge(df_sm1, df_sm2_long, on=['date', 'region'], how='left')
    merged['source'] = merged['source'].fillna('Total')

    # Standardize state names
    merged["states"] = merged["states"].replace(STATE_MAPPING)

    # Remove non-state entities
    non_states = ["AMNSIL", "BALCO", "DVC", "Essar steel", "Railways_NR ISTS", "Damodar Valley Corporation", "RIL JAMNAGAR", "Railways_ER ISTS", "States"]
    merged = merged[~merged["states"].isin(non_states)]
    merged = merged.drop_duplicates(subset=['date', 'states', 'source'])

    # Time features
    merged['month'] = merged['date'].dt.month
    merged['day_of_week'] = merged['date'].dt.dayofweek
    merged['is_weekend'] = merged['day_of_week'].isin([5, 6])
    merged['year'] = merged['date'].dt.year

    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Summer'
        elif month in [6, 7, 8, 9]:
            return 'Monsoon'
        else:
            return 'Autumn'

    merged['season'] = merged['month'].apply(get_season)
    merged = merged.sort_values('date').reset_index(drop=True)

    # Load Zenodo weather data
    print("Loading Zenodo weather dataset...")
    zenodo_df = pd.read_csv(ZENODO_PATH)
    if 'state' in zenodo_df.columns:
        zenodo_df.rename(columns={'state': 'states'}, inplace=True)
    zenodo_df['date'] = pd.to_datetime(zenodo_df['date'])

    # Merge with weather
    df = zenodo_df.merge(merged, on=['date', 'states'], how='inner')
    df = df.sort_values(['states', 'source', 'date']).reset_index(drop=True)

    print(f"Merged dataset shape: {df.shape}")
    print(f"Dates: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Unique years: {sorted(df['date'].dt.year.unique())}")
    print(f"Unique states: {df['states'].nunique()}")

    # Rolling averages matching preprocessing
    df['energymet_7d_avg'] = (
        df.groupby(['states', 'source'])['energy met (mu)']
          .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    df['generation_7d_avg'] = (
        df.groupby(['states', 'source'])['regional_generation_mu']
          .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    df['demand_rolling'] = df['energymet_7d_avg']
    df['gen_rolling'] = df['generation_7d_avg']

    df['monthly_generation_avg'] = (
        df.groupby(['states', 'source', 'month'])['regional_generation_mu']
          .transform('mean')
    )
    df['monthly_energy_avg'] = (
        df.groupby(['states', 'source', 'month'])['energy met (mu)']
          .transform('mean')
    )

    # Save preprocessed clean dataset
    out_clean = DATA_DIR / 'processed' / 'preprocessed_merged_features_clean.csv'
    df.to_csv(out_clean, index=False)
    print(f"✓ Saved {len(df):,} rows -> {out_clean}")

    # Also update preprocessed_merged.csv & merged.csv in data/
    df.to_csv(DATA_DIR / 'preprocessed_merged.csv', index=False)
    df.to_csv(DATA_DIR / 'merged.csv', index=False)
    print(f"✓ Updated data/preprocessed_merged.csv and data/merged.csv")

    # National daily dataset
    national_daily = df[df['source'] == 'Total'].groupby('date').mean(numeric_only=True).reset_index()
    national_daily.to_csv(DATA_DIR / 'national_daily.csv', index=False)
    print(f"✓ Updated data/national_daily.csv: {len(national_daily):,} rows")

    # Final merged dataset / feature engineered dataset
    df.to_csv(DATA_DIR / 'final_merged_dataset.csv', index=False)
    df.to_csv(DATA_DIR / 'processed' / 'feature_engineered_dataset.csv', index=False)
    print(f"✓ Updated data/final_merged_dataset.csv and feature_engineered_dataset.csv")

    print("\n=== Ingestion and Preprocessing Complete! ===")


if __name__ == "__main__":
    ingest_demand_data()
    run_full_preprocessing()
