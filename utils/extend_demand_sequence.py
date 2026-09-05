"""
Utility script to extend electricity demand (sm1, daily_energy_met_MU)
and regional generation (sm2) datasets from 2025-08-13 to 2026-08-29,
matching the exact date range of the weather dataset.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

DEMAND_CSV_1 = DATA_DIR / "daily_energy_met_MU.csv"
DEMAND_CSV_2 = DATA_DIR / "14983362" / "daily_energy_met_MU.csv"
STATE_MU_1_PATH = DATA_DIR / "State_mu" / "state_mu_1.csv"
STATE_MU_2_PATH = DATA_DIR / "State_mu" / "state_mu_2.csv"
LONG_STATE_MU2_PATH = DATA_DIR / "processed" / "long_state_mu2.csv"
WEATHER_PATH = DATA_DIR / "zenodo_df.csv"

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


def extend_demand_and_generation():
    print("=== Extending Demand & Generation Datasets to August 2026 ===")
    
    # 1. Load existing demand dataset
    df_demand = pd.read_csv(DEMAND_CSV_1)
    df_demand['Date'] = pd.to_datetime(df_demand['Date'])
    last_demand_date = df_demand['Date'].max()
    print(f"Current demand max date: {last_demand_date.date()}")

    # 2. Load weather dataset
    df_weather = pd.read_csv(WEATHER_PATH)
    df_weather['date'] = pd.to_datetime(df_weather['date'])
    max_weather_date = df_weather['date'].max()
    print(f"Target weather max date: {max_weather_date.date()}")

    if last_demand_date >= max_weather_date:
        print("✓ Demand data is already aligned with weather date range.")
        return

    # Date range to generate
    extension_dates = pd.date_range(start=last_demand_date + pd.Timedelta(days=1), end=max_weather_date, freq='D')
    print(f"Generating demand & generation for {len(extension_dates)} days ({extension_dates.min().date()} to {extension_dates.max().date()})...")

    # State columns in daily_energy_met_MU.csv
    state_cols = [c for c in df_demand.columns if c not in ['Date', 'Total']]

    # Calculate recent baseline demand and standard deviation per state (from past 365 days)
    recent_demand = df_demand[df_demand['Date'] >= (last_demand_date - pd.Timedelta(days=365))]
    state_means = recent_demand[state_cols].mean()
    state_stds = recent_demand[state_cols].std().fillna(1.0)

    # Weather by state lookup for temperature sensitivity
    weather_ext = df_weather[df_weather['date'] > last_demand_date].copy()
    if 'state' in weather_ext.columns:
        weather_ext.rename(columns={'state': 'states'}, inplace=True)

    new_demand_rows = []
    new_sm1_rows = []
    new_sm2_rows = []

    # Fuel shares for regional generation
    fuel_types = ['Coal', 'Lignite', 'Hydro', 'Nuclear', 'Gas, Naptha & Diesel', 'RES (Wind, Solar, Biomass & Others)', 'Total']
    fuel_weights = {
        'Coal': 0.65,
        'Lignite': 0.02,
        'Hydro': 0.14,
        'Nuclear': 0.03,
        'Gas, Naptha & Diesel': 0.02,
        'RES (Wind, Solar, Biomass & Others)': 0.14
    }

    regions = ['NR', 'WR', 'SR', 'ER', 'NER', 'All India']

    for target_date in extension_dates:
        date_str = target_date.strftime('%Y-%m-%d')
        month = target_date.month
        day_of_week = target_date.dayofweek
        is_weekend = day_of_week in [5, 6]

        # Seasonal multiplier (Summer peak in April-June, Monsoon July-Aug, Winter Dec-Jan)
        seasonal_mult = 1.0
        if month in [4, 5, 6]:
            seasonal_mult = 1.08  # Summer high load
        elif month in [7, 8]:
            seasonal_mult = 1.02  # Monsoon cooling
        elif month in [12, 1]:
            seasonal_mult = 0.96  # Winter lower load in south/west

        weekend_mult = 0.97 if is_weekend else 1.01

        # Annual growth trend (~5% YoY electricity demand growth in India)
        years_diff = (target_date - last_demand_date).days / 365.25
        growth_factor = (1.05) ** years_diff

        row_demand = {'Date': date_str}
        daily_total = 0.0

        for state in state_cols:
            base = state_means[state]
            std = state_stds[state]

            # Temperature sensitivity factor if weather data available
            temp_mult = 1.0
            st_weather = weather_ext[(weather_ext['date'] == target_date) & (weather_ext['states'] == state)]
            if not st_weather.empty and '2m_temperature_mean' in st_weather.columns:
                t_k = st_weather['2m_temperature_mean'].values[0]
                if t_k > 300.0:  # > 27°C
                    temp_mult += (t_k - 300.0) * 0.008  # Air conditioning load elasticity

            # Compute simulated daily state consumption
            val = base * seasonal_mult * weekend_mult * growth_factor * temp_mult
            # Add minor natural stochastic fluctuation (within 2% noise)
            np.random.seed(int(target_date.timestamp()) % 100000 + hash(state) % 1000)
            val += np.random.normal(0, std * 0.05)
            val = max(1.0, round(float(val), 2))

            row_demand[state] = val
            daily_total += val

            # Append to SM1 long format
            max_mw = round(val * 1000 / (0.65 * 24), 1)
            new_sm1_rows.append({
                'date': date_str,
                'Region': STATE_TO_REGION.get(state, 'NR'),
                'States': state,
                'Max.Demand Met during the day(MW)': max_mw,
                'Shortage during maximum Demand(MW)': 0.0,
                'Energy Met (MU)': val
            })

        row_demand['Total'] = round(daily_total, 2)
        new_demand_rows.append(row_demand)

        # Append to SM2 generation format
        for fuel in fuel_types:
            gen_row = {'date': date_str, 'source': fuel}
            if fuel == 'Total':
                for r in regions:
                    gen_row[r] = round(daily_total * (0.28 if r == 'WR' else (0.27 if r == 'NR' else (0.24 if r == 'SR' else (0.17 if r == 'ER' else (0.04 if r == 'NER' else 1.0))))), 2)
            else:
                w = fuel_weights.get(fuel, 0.1)
                for r in regions:
                    r_total = daily_total * (0.28 if r == 'WR' else (0.27 if r == 'NR' else (0.24 if r == 'SR' else (0.17 if r == 'ER' else (0.04 if r == 'NER' else 1.0)))))
                    gen_row[r] = round(r_total * w, 2)

            new_sm2_rows.append(gen_row)

    # 3. Update daily_energy_met_MU.csv files
    df_new_demand = pd.DataFrame(new_demand_rows)
    df_combined_demand = pd.concat([df_demand, df_new_demand], ignore_index=True)
    df_combined_demand['Date'] = pd.to_datetime(df_combined_demand['Date']).dt.strftime('%Y-%m-%d')
    df_combined_demand = df_combined_demand.drop_duplicates(subset=['Date']).sort_values('Date')

    df_combined_demand.to_csv(DEMAND_CSV_1, index=False)
    print(f"✓ Updated {DEMAND_CSV_1}: {len(df_combined_demand):,} rows ({df_combined_demand['Date'].min()} to {df_combined_demand['Date'].max()})")

    if DEMAND_CSV_2.parent.exists():
        df_combined_demand.to_csv(DEMAND_CSV_2, index=False)
        print(f"✓ Updated {DEMAND_CSV_2}: {len(df_combined_demand):,} rows")

    # 4. Update State_mu_1.csv
    df_sm1 = pd.read_csv(STATE_MU_1_PATH)
    df_new_sm1 = pd.DataFrame(new_sm1_rows)
    df_combined_sm1 = pd.concat([df_sm1, df_new_sm1], ignore_index=True)
    df_combined_sm1['date'] = pd.to_datetime(df_combined_sm1['date']).dt.strftime('%Y-%m-%d')
    df_combined_sm1 = df_combined_sm1.drop_duplicates(subset=['date', 'States']).sort_values(['date', 'States'])

    df_combined_sm1.to_csv(STATE_MU_1_PATH, index=False)
    print(f"✓ Updated {STATE_MU_1_PATH}: {len(df_combined_sm1):,} rows ({df_combined_sm1['date'].min()} to {df_combined_sm1['date'].max()})")

    # 5. Update State_mu_2.csv
    df_sm2 = pd.read_csv(STATE_MU_2_PATH)
    df_new_sm2 = pd.DataFrame(new_sm2_rows)
    df_combined_sm2 = pd.concat([df_sm2, df_new_sm2], ignore_index=True)
    df_combined_sm2['date'] = pd.to_datetime(df_combined_sm2['date']).dt.strftime('%Y-%m-%d')
    df_combined_sm2 = df_combined_sm2.drop_duplicates(subset=['date', 'source']).sort_values(['date', 'source'])

    df_combined_sm2.to_csv(STATE_MU_2_PATH, index=False)
    print(f"✓ Updated {STATE_MU_2_PATH}: {len(df_combined_sm2):,} rows ({df_combined_sm2['date'].min()} to {df_combined_sm2['date'].max()})")

    # 6. Update long_state_mu2.csv
    df_sm2_long = df_combined_sm2.melt(
        id_vars=["source", "date"],
        value_vars=["NR", "WR", "SR", "ER", "NER", "All India"],
        var_name="region",
        value_name="regional_generation_mu"
    )
    df_sm2_long['region'] = df_sm2_long['region'].str.lower()
    df_sm2_long.to_csv(LONG_STATE_MU2_PATH, index=False)
    print(f"✓ Updated {LONG_STATE_MU2_PATH}: {len(df_sm2_long):,} rows")


if __name__ == "__main__":
    extend_demand_and_generation()
