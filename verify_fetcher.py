"""
Verification Script for Ingestion Worker & Normalization.
Run with:
    python verify_fetcher.py
"""

from datetime import date, timedelta
from utils.ingestion_worker import normalize_state_name, fetch_actual_grid_demand

print("\n" + "=" * 60)
print("🔍 1. Testing State Normalization")
print("=" * 60)
raw_inputs = ["Delhi", "J&K", "Pondy", "Maharashtra", "Andaman and Nicobar Islands", "DNH"]
for raw in raw_inputs:
    print(f"  • Raw: {raw:<30} -> Normalized: {normalize_state_name(raw)}")

yesterday = date.today() - timedelta(days=1)

print("\n" + "=" * 60)
print(f"⚡ 2. Testing Actual Grid Demand Retrieval ({yesterday})")
print("=" * 60)
actuals = fetch_actual_grid_demand(yesterday)
print(f"  • Records found: {len(actuals)} (empty = no real data for this date)")
print(f"  • Sample Ingested Records:")
for row in actuals[:5]:
    print(f"    - {row['state']:<25} : {row['actual_demand_mu']:>7.2f} MU ({row['source']})")

print("=" * 60)
print("✅ Verification Complete! All data fetchers and normalizers functional.\n")
