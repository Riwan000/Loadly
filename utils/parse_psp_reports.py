"""
Utility to parse Grid-India (formerly POSOCO) Daily Power Supply Position (PSP) reports
from Excel (.xls) and PDF files into structured state-wise demand and generation DataFrames.
"""

import os
import re
import glob
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import xlrd

ROOT_DIR = Path(__file__).resolve().parent.parent
PSP_DIR = ROOT_DIR / "data" / "psp"

# Standard state names matching daily_energy_met_MU.csv columns
STATE_NAME_MAPPING = {
    "UP": "Uttar Pradesh",
    "Uttar Pradesh": "Uttar Pradesh",
    "MP": "Madhya Pradesh",
    "Madhya Pradesh": "Madhya Pradesh",
    "HP": "Himachal Pradesh",
    "Himachal Pradesh": "Himachal Pradesh",
    "J&K(UT) & Ladakh(UT)": "Jammu and Kashmir",
    "J&K(UT) and Ladakh(UT)": "Jammu and Kashmir",
    "J&K": "Jammu and Kashmir",
    "Jammu and Kashmir": "Jammu and Kashmir",
    "Jammu & Kashmir": "Jammu and Kashmir",
    "Punjab": "Punjab",
    "Haryana": "Haryana",
    "Rajasthan": "Rajasthan",
    "Delhi": "Delhi",
    "Uttarakhand": "Uttarakhand",
    "Chandigarh": "Chandigarh",
    "Chhattisgarh": "Chhattisgarh",
    "Gujarat": "Gujarat",
    "Maharashtra": "Maharashtra",
    "Goa": "Goa",
    "DNHDDPDCL": "Dadra and Nagar Haveli",  # Merged UT
    "Dadra and Nagar Haveli": "Dadra and Nagar Haveli",
    "Daman and Diu": "Daman and Diu",
    "AMNSIL": "AMNSIL",
    "Andhra Pradesh": "Andhra Pradesh",
    "Telangana": "Telangana",
    "Karnataka": "Karnataka",
    "Kerala": "Kerala",
    "Tamil Nadu": "Tamil Nadu",
    "Puducherry": "Puducherry",
    "Bihar": "Bihar",
    "DVC": "Damodar Valley Corporation",
    "Damodar Valley Corporation": "Damodar Valley Corporation",
    "Jharkhand": "Jharkhand",
    "Odisha": "Odisha",
    "West Bengal": "West Bengal",
    "Sikkim": "Sikkim",
    "Arunachal Pradesh": "Arunachal Pradesh",
    "Assam": "Assam",
    "Manipur": "Manipur",
    "Meghalaya": "Meghalaya",
    "Mizoram": "Mizoram",
    "Nagaland": "Nagaland",
    "Tripura": "Tripura",
    "Andaman and Nicobar Islands": "Andaman and Nicobar Islands",
    "Lakshadweep": "Lakshadweep"
}

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


def parse_date_from_filename_or_sheet(filepath: str, wb: xlrd.Book = None) -> str:
    """Extract report date (YYYY-MM-DD) from filename or sheet metadata."""
    filename = os.path.basename(filepath)
    # Match pattern DD.MM.YY (e.g. 01.08.25)
    m = re.search(r'(\d{2})\.(\d{2})\.(\d{2,4})', filename)
    if m:
        d, mth, y = m.groups()
        if len(y) == 2:
            y = f"20{y}"
        try:
            dt = datetime.date(int(y), int(mth), int(d))
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            pass

    # Fallback to workbook cell
    if wb and 'MOP_E' in wb.sheet_names():
        sh = wb.sheet_by_name('MOP_E')
        for r in range(min(5, sh.nrows)):
            for c in range(sh.ncols):
                val = str(sh.cell_value(r, c)).strip()
                # e.g. 02-Aug-2025
                try:
                    dt = pd.to_datetime(val).date()
                    # Previous day report
                    prev_dt = dt - datetime.timedelta(days=1)
                    return prev_dt.strftime('%Y-%m-%d')
                except Exception:
                    continue

    return None


def parse_psp_xls(filepath: str) -> dict:
    """
    Parse a single Grid-India NLDC PSP Excel report (.xls).
    
    Returns:
        dict with:
            - 'date': 'YYYY-MM-DD'
            - 'state_demand_wide': dict of state_name -> energy_met_mu
            - 'state_demand_long': list of dicts with state demand details
            - 'generation_long': list of dicts with fuel generation details
    """
    try:
        wb = xlrd.open_workbook(filepath)
    except Exception as e:
        print(f"❌ Error opening {filepath}: {e}")
        return None

    if 'MOP_E' not in wb.sheet_names():
        return None

    date_str = parse_date_from_filename_or_sheet(filepath, wb)
    if not date_str:
        print(f"⚠️ Could not determine date for {filepath}")
        return None

    sh = wb.sheet_by_name('MOP_E')

    # Locate Section C: Power Supply Position in States
    state_start_row = None
    for r in range(sh.nrows):
        val = str(sh.cell_value(r, 0)).strip()
        if "Power Supply Position in States" in val or "Power Supply Position in States" in str(sh.cell_value(r, 1)):
            state_start_row = r
            break

    state_demand_wide = {'Date': date_str}
    state_demand_long = []

    if state_start_row is not None:
        # Table headers are usually at state_start_row + 2
        # Data rows start from state_start_row + 3
        current_region = "NR"
        for r in range(state_start_row + 2, sh.nrows):
            r_val0 = str(sh.cell_value(r, 0)).strip()
            state_raw = str(sh.cell_value(r, 1)).strip()

            if r_val0 in ["NR", "WR", "SR", "ER", "NER"]:
                current_region = r_val0

            if not state_raw or state_raw.startswith("D.") or "Transnational" in state_raw or "Total" in state_raw:
                if state_raw.startswith("D.") or "Transnational" in state_raw:
                    break
                continue

            # Standardize state name
            mapped_state = STATE_NAME_MAPPING.get(state_raw, state_raw)

            try:
                max_demand_mw = float(sh.cell_value(r, 2)) if sh.cell_value(r, 2) != '-' and sh.cell_value(r, 2) != '' else 0.0
            except (ValueError, TypeError):
                max_demand_mw = 0.0

            try:
                shortage_mw = float(sh.cell_value(r, 3)) if sh.cell_value(r, 3) != '-' and sh.cell_value(r, 3) != '' else 0.0
            except (ValueError, TypeError):
                shortage_mw = 0.0

            try:
                energy_met_mu = float(sh.cell_value(r, 4)) if sh.cell_value(r, 4) != '-' and sh.cell_value(r, 4) != '' else 0.0
            except (ValueError, TypeError):
                energy_met_mu = 0.0

            state_demand_wide[mapped_state] = energy_met_mu

            state_demand_long.append({
                'date': date_str,
                'Region': STATE_TO_REGION.get(mapped_state, current_region),
                'States': mapped_state,
                'Max.Demand Met during the day(MW)': max_demand_mw,
                'Shortage during maximum Demand(MW)': shortage_mw,
                'Energy Met (MU)': energy_met_mu
            })

    # Locate Section G: Sourcewise generation (Gross)
    gen_start_row = None
    for r in range(sh.nrows):
        val = str(sh.cell_value(r, 0)).strip()
        if "Sourcewise generation" in val:
            gen_start_row = r
            break

    generation_long = []
    if gen_start_row is not None:
        regions = ['NR', 'WR', 'SR', 'ER', 'NER', 'All India']
        for r in range(gen_start_row + 2, min(gen_start_row + 10, sh.nrows)):
            fuel_source = str(sh.cell_value(r, 0)).strip()
            if not fuel_source or "Total" in fuel_source or "Share" in fuel_source:
                continue

            gen_row = {'date': date_str, 'source': fuel_source}
            # Region columns are typically col 2 to 7
            for idx, reg in enumerate(regions):
                col_idx = idx + 2
                try:
                    val = float(sh.cell_value(r, col_idx)) if sh.cell_value(r, col_idx) not in ['-', ''] else 0.0
                except (ValueError, TypeError):
                    val = 0.0
                gen_row[reg] = val

            generation_long.append(gen_row)

    return {
        'date': date_str,
        'state_demand_wide': state_demand_wide,
        'state_demand_long': state_demand_long,
        'generation_long': generation_long
    }


def parse_all_psp_files(psp_dir: Path = PSP_DIR) -> tuple:
    """
    Parse all available PSP .xls files in data/psp/.
    
    Returns:
        (df_wide_demand, df_long_demand, df_generation)
    """
    xls_files = sorted(glob.glob(str(psp_dir / "*.xls")))
    print(f"=== Found {len(xls_files)} PSP Excel files in {psp_dir} ===")

    all_wide = []
    all_long_demand = []
    all_long_gen = []

    for f in xls_files:
        res = parse_psp_xls(f)
        if res:
            all_wide.append(res['state_demand_wide'])
            all_long_demand.extend(res['state_demand_long'])
            all_long_gen.extend(res['generation_long'])
            print(f"  ✓ Parsed {os.path.basename(f)} for date {res['date']}")

    df_wide = pd.DataFrame(all_wide)
    df_long = pd.DataFrame(all_long_demand)
    df_gen = pd.DataFrame(all_long_gen)

    return df_wide, df_long, df_gen


if __name__ == "__main__":
    df_w, df_l, df_g = parse_all_psp_files()
    print("\nSummary of Parsed Demand (Wide):")
    print(f"Shape: {df_w.shape}")
    print(df_w.head(3))
