"""
Utility script to fetch recent weather data (from 2024-04-28 to present)
for all 36 Indian states using the Open-Meteo Archive API,
convert to standard ERA5/Zenodo schema (Kelvin, SSRD, wind vectors),
update individual state CSVs in data/14983362/, and rebuild data/zenodo_df.csv.
"""

import os
import time
import glob
import datetime
import requests
import numpy as np
import pandas as pd
from pathlib import Path

# Fix relative imports when running directly
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import STATE_COORDINATES, ALL_STATES

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "14983362"
ZENODO_PATH = ROOT_DIR / "data" / "zenodo_df.csv"

# Map state display name to filename in data/14983362/
STATE_FILE_MAP = {
    "Andaman and Nicobar": "Andaman_and_Nicobar.csv",
    "Andhra Pradesh": "Andhra_Pradesh.csv",
    "Arunachal Pradesh": "Arunachal_Pradesh.csv",
    "Assam": "Assam.csv",
    "Bihar": "Bihar.csv",
    "Chandigarh": "Chandigarh.csv",
    "Chhattisgarh": "Chhattisgarh.csv",
    "Dadra and Nagar Haveli": "Dadra_and_Nagar_Haveli.csv",
    "Daman and Diu": "Daman_and_Diu.csv",
    "Goa": "Goa.csv",
    "Gujarat": "Gujarat.csv",
    "Haryana": "Haryana.csv",
    "Himachal Pradesh": "Himachal_Pradesh.csv",
    "Jammu and Kashmir": "Jammu_and_Kashmir.csv",
    "Jharkhand": "Jharkhand.csv",
    "Karnataka": "Karnataka.csv",
    "Kerala": "Kerala.csv",
    "Lakshadweep": "Lakshadweep.csv",
    "Madhya Pradesh": "Madhya_Pradesh.csv",
    "Maharashtra": "Maharashtra.csv",
    "Manipur": "Manipur.csv",
    "Meghalaya": "Meghalaya.csv",
    "Mizoram": "Mizoram.csv",
    "NCT of Delhi": "NCT_of_Delhi.csv",
    "Nagaland": "Nagaland.csv",
    "Odisha": "Odisha.csv",
    "Puducherry": "Puducherry.csv",
    "Punjab": "Punjab.csv",
    "Rajasthan": "Rajasthan.csv",
    "Sikkim": "Sikkim.csv",
    "Tamil Nadu": "Tamil_Nadu.csv",
    "Telangana": "Telangana.csv",
    "Tripura": "Tripura.csv",
    "Uttar Pradesh": "Uttar_Pradesh.csv",
    "Uttarakhand": "Uttarakhand.csv",
    "West Bengal": "West_Bengal.csv"
}

EXPECTED_COLUMNS = [
    'date',
    '10m_u_component_of_wind_min',
    '10m_u_component_of_wind_mean',
    '10m_u_component_of_wind_max',
    '10m_v_component_of_wind_min',
    '10m_v_component_of_wind_mean',
    '10m_v_component_of_wind_max',
    '2m_temperature_min',
    '2m_temperature_mean',
    '2m_temperature_max',
    '2m_dewpoint_temperature_min',
    '2m_dewpoint_temperature_mean',
    '2m_dewpoint_temperature_max',
    'surface_solar_radiation_downwards_min',
    'surface_solar_radiation_downwards_mean',
    'surface_solar_radiation_downwards_max',
    'total_cloud_cover_min',
    'total_cloud_cover_mean',
    'total_cloud_cover_max',
    'utci_min',
    'utci_mean',
    'utci_max'
]


def fetch_openmeteo_hourly_archive(lat: float, lon: float, start_date: str, end_date: str, max_retries: int = 4):
    """Fetch hourly weather data from Open-Meteo Archive API with retries."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "dewpoint_2m",
            "apparent_temperature",
            "windspeed_10m",
            "winddirection_10m",
            "shortwave_radiation",
            "cloudcover"
        ],
        "timezone": "Asia/Kolkata"
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if "hourly" in data:
                    return data["hourly"]
            elif response.status_code == 429:
                print(f"    [Rate limit 429] Waiting 5s before retry (attempt {attempt}/{max_retries})...")
                time.sleep(5)
            else:
                print(f"    [HTTP {response.status_code}] {response.text}")
        except Exception as e:
            print(f"    [Error attempt {attempt}] {e}")
            time.sleep(2 * attempt)

    return None


def process_hourly_to_daily(hourly_data: dict) -> pd.DataFrame:
    """Convert Open-Meteo hourly response to daily ERA5-formatted DataFrame."""
    df_h = pd.DataFrame(hourly_data)
    df_h['time'] = pd.to_datetime(df_h['time'])
    df_h['date'] = df_h['time'].dt.strftime('%Y-%m-%d')

    # Compute wind vectors u, v (wind blowing towards direction)
    # Open-Meteo windspeed is in km/h -> convert to m/s
    speed_ms = df_h['windspeed_10m'] / 3.6
    dir_rad = np.radians(df_h['winddirection_10m'])
    df_h['u'] = -speed_ms * np.sin(dir_rad)
    df_h['v'] = -speed_ms * np.cos(dir_rad)

    # Convert Celsius to Kelvin
    df_h['temp_k'] = df_h['temperature_2m'] + 273.15
    df_h['dew_k'] = df_h['dewpoint_2m'] + 273.15
    df_h['utci_k'] = df_h['apparent_temperature'] + 273.15

    # Cloud cover: scale from 0-100 to 0.0-1.0
    df_h['cloud'] = df_h['cloudcover'] / 100.0

    # Solar radiation: W/m^2 -> J/m^2 for hourly accumulation
    df_h['ssrd_j'] = df_h['shortwave_radiation'] * 3600.0

    # Group by date and calculate min, mean, max
    daily_records = []
    for date_str, group in df_h.groupby('date'):
        record = {
            'date': date_str,
            '10m_u_component_of_wind_min': group['u'].min(),
            '10m_u_component_of_wind_mean': group['u'].mean(),
            '10m_u_component_of_wind_max': group['u'].max(),
            '10m_v_component_of_wind_min': group['v'].min(),
            '10m_v_component_of_wind_mean': group['v'].mean(),
            '10m_v_component_of_wind_max': group['v'].max(),
            '2m_temperature_min': group['temp_k'].min(),
            '2m_temperature_mean': group['temp_k'].mean(),
            '2m_temperature_max': group['temp_k'].max(),
            '2m_dewpoint_temperature_min': group['dew_k'].min(),
            '2m_dewpoint_temperature_mean': group['dew_k'].mean(),
            '2m_dewpoint_temperature_max': group['dew_k'].max(),
            'surface_solar_radiation_downwards_min': group['ssrd_j'].min(),
            'surface_solar_radiation_downwards_mean': group['ssrd_j'].mean(),
            'surface_solar_radiation_downwards_max': group['ssrd_j'].max(),
            'total_cloud_cover_min': group['cloud'].min(),
            'total_cloud_cover_mean': group['cloud'].mean(),
            'total_cloud_cover_max': group['cloud'].max(),
            'utci_min': group['utci_k'].min(),
            'utci_mean': group['utci_k'].mean(),
            'utci_max': group['utci_k'].max()
        }
        daily_records.append(record)

    return pd.DataFrame(daily_records)


def update_all_states(start_date: str = "2024-04-28", end_date: str = None):
    """Fetch and update weather data for all 36 states."""
    if end_date is None:
        today = datetime.date.today()
        end_date = (today - datetime.timedelta(days=2)).strftime('%Y-%m-%d')

    print(f"=== Starting Weather Data Ingestion: {start_date} to {end_date} ===")
    os.makedirs(DATA_DIR, exist_ok=True)

    successful_states = 0
    total_new_rows = 0

    for state_name in ALL_STATES:
        filename = STATE_FILE_MAP.get(state_name)
        if not filename:
            print(f"⚠️ No file mapping for state '{state_name}'. Skipping.")
            continue

        file_path = DATA_DIR / filename
        coords = STATE_COORDINATES.get(state_name)
        if not coords:
            print(f"⚠️ No coordinates for state '{state_name}'. Skipping.")
            continue

        lat, lon = coords["lat"], coords["lon"]
        print(f"[{successful_states + 1}/{len(ALL_STATES)}] Processing {state_name} ({lat}, {lon})...")

        # Load existing file if present
        existing_df = None
        if file_path.exists():
            existing_df = pd.read_csv(file_path)
            # Remove any trailing all-null rows (e.g. placeholder row on 2024-04-27)
            existing_df = existing_df.dropna(subset=['2m_temperature_mean'])

        # Fetch from Open-Meteo
        hourly_data = fetch_openmeteo_hourly_archive(lat, lon, start_date, end_date)
        if not hourly_data:
            print(f"❌ Failed to fetch data for {state_name}.")
            continue

        new_daily_df = process_hourly_to_daily(hourly_data)
        if new_daily_df.empty:
            print(f"⚠️ Empty daily DataFrame for {state_name}.")
            continue

        # Combine with existing data
        if existing_df is not None and not existing_df.empty:
            # Filter out dates already in new_daily_df
            existing_df = existing_df[~existing_df['date'].isin(new_daily_df['date'])]
            combined_df = pd.concat([existing_df, new_daily_df], ignore_index=True)
        else:
            combined_df = new_daily_df

        # Sort and clean
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        combined_df = combined_df.sort_values('date').drop_duplicates(subset=['date'])
        combined_df['date'] = combined_df['date'].dt.strftime('%Y-%m-%d')

        # Ensure column ordering matches EXPECTED_COLUMNS
        cols = [c for c in EXPECTED_COLUMNS if c in combined_df.columns]
        combined_df = combined_df[cols]

        combined_df.to_csv(file_path, index=False)
        print(f"  ✓ Updated {filename}: {len(combined_df)} total rows ({combined_df['date'].min()} to {combined_df['date'].max()})")

        successful_states += 1
        total_new_rows += len(new_daily_df)
        time.sleep(0.5)  # Friendly pause between requests

    print(f"\n✓ Successfully updated {successful_states}/{len(ALL_STATES)} states.")
    print(f"✓ Total new weather records ingested: {total_new_rows:,}")

    # Rebuild zenodo_df.csv
    rebuild_zenodo_df()


def rebuild_zenodo_df():
    """Rebuild data/zenodo_df.csv by aggregating all 36 state CSVs."""
    print("\n=== Rebuilding data/zenodo_df.csv ===")
    state_files = glob.glob(str(DATA_DIR / "*.csv"))
    
    # Filter out non-state files
    state_files = [
        f for f in state_files 
        if not f.endswith("MU.csv") and not f.endswith("MW.csv") and not "utci" in f
    ]

    dfs = []
    for file in state_files:
        df = pd.read_csv(file)
        state_name = Path(file).stem.replace("_", " ")
        df['state'] = state_name
        dfs.append(df)

    if not dfs:
        print("❌ No state files found to rebuild zenodo_df.csv.")
        return

    full_zenodo_df = pd.concat(dfs, ignore_index=True)
    full_zenodo_df['date'] = pd.to_datetime(full_zenodo_df['date'])
    full_zenodo_df = full_zenodo_df.sort_values(['date', 'state']).reset_index(drop=True)
    full_zenodo_df['date'] = full_zenodo_df['date'].dt.strftime('%Y-%m-%d')

    full_zenodo_df.to_csv(ZENODO_PATH, index=False)
    print(f"✓ Saved {len(full_zenodo_df):,} rows across {full_zenodo_df['state'].nunique()} states to {ZENODO_PATH}")
    print(f"  Date range: {full_zenodo_df['date'].min()} to {full_zenodo_df['date'].max()}")


if __name__ == "__main__":
    update_all_states()
