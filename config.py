# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# State to coordinates mapping (approximate centers of Indian states)
# These coordinates are used to fetch weather data from APIs
STATE_COORDINATES = {
    "All-India": {"lat": 20.5937, "lon": 78.9629},
    "Kerala": {"lat": 10.8505, "lon": 76.2711},
    "Goa": {"lat": 15.2993, "lon": 74.1240},
    "Maharashtra": {"lat": 19.7515, "lon": 75.7139},
    "Karnataka": {"lat": 15.3173, "lon": 75.7139},
    "Tamil Nadu": {"lat": 11.1271, "lon": 78.6569},
    "Gujarat": {"lat": 23.0225, "lon": 72.5714},
    "Rajasthan": {"lat": 27.0238, "lon": 74.2179},
    "Uttar Pradesh": {"lat": 26.8467, "lon": 80.9462},
    "West Bengal": {"lat": 22.9868, "lon": 87.8550},
    "Andhra Pradesh": {"lat": 15.9129, "lon": 79.7400},
    "Telangana": {"lat": 18.1124, "lon": 79.0193},
    "Punjab": {"lat": 31.1471, "lon": 75.3412},
    "Haryana": {"lat": 29.0588, "lon": 76.0856},
    "Madhya Pradesh": {"lat": 22.9734, "lon": 78.6569},
    "Bihar": {"lat": 25.0961, "lon": 85.3131},
    "Odisha": {"lat": 20.9517, "lon": 85.0985},
    "Assam": {"lat": 26.2006, "lon": 92.9376},
    "Jharkhand": {"lat": 23.6102, "lon": 85.2799},
    "Chhattisgarh": {"lat": 21.2787, "lon": 81.8661},
    "Uttarakhand": {"lat": 30.0668, "lon": 79.0193},
    "Himachal Pradesh": {"lat": 31.1048, "lon": 77.1734},
    "Jammu and Kashmir": {"lat": 34.0837, "lon": 74.7973},
    "Manipur": {"lat": 24.6637, "lon": 93.9063},
    "Meghalaya": {"lat": 25.4670, "lon": 91.3662},
    "Mizoram": {"lat": 23.1645, "lon": 92.9376},
    "Nagaland": {"lat": 26.1584, "lon": 94.5624},
    "Tripura": {"lat": 23.9408, "lon": 91.9882},
    "Arunachal Pradesh": {"lat": 28.2180, "lon": 94.7278},
    "Sikkim": {"lat": 27.5330, "lon": 88.5122},
    "NCT of Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Puducherry": {"lat": 11.9416, "lon": 79.8083},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794},
    "Dadra and Nagar Haveli": {"lat": 20.1809, "lon": 73.0169},
    "Daman and Diu": {"lat": 20.3974, "lon": 72.8328},
    "Lakshadweep": {"lat": 10.5667, "lon": 72.6417},
    "Andaman and Nicobar": {"lat": 11.7401, "lon": 92.6586},
}

# List of all Indian states for dropdown
ALL_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir",
    "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
    "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "NCT of Delhi",
    "Puducherry", "Chandigarh", "Dadra and Nagar Haveli", "Daman and Diu",
    "Lakshadweep", "Andaman and Nicobar"
]

NATIONAL_STATE = "All-India"

# A state whose latest demand observation is more than this many days before the
# forecast start is flagged as stale (the recursive forecast is seeded from it).
STALE_DEMAND_DAYS = 30

# ---------------------------------------------------------------------------
# Canonical state -> grid region map (single source of truth).
# Regions follow the Indian power-grid regions: nr, wr, sr, er, ner.
# Chhattisgarh is Western (wr); Sikkim and Andaman and Nicobar are Eastern (er).
# Everything that needs a region (API summaries, UI warnings, legacy feature
# encoding in features.py) must derive from this map.
# ---------------------------------------------------------------------------
STATE_REGIONS = {
    # Northern Region
    "Punjab": "nr", "Haryana": "nr", "Rajasthan": "nr", "Uttar Pradesh": "nr",
    "Uttarakhand": "nr", "Himachal Pradesh": "nr", "Jammu and Kashmir": "nr",
    "NCT of Delhi": "nr", "Chandigarh": "nr",
    # Western Region
    "Maharashtra": "wr", "Gujarat": "wr", "Madhya Pradesh": "wr", "Chhattisgarh": "wr",
    "Goa": "wr", "Dadra and Nagar Haveli": "wr", "Daman and Diu": "wr",
    # Southern Region
    "Tamil Nadu": "sr", "Karnataka": "sr", "Andhra Pradesh": "sr",
    "Telangana": "sr", "Kerala": "sr", "Puducherry": "sr", "Lakshadweep": "sr",
    # Eastern Region
    "Bihar": "er", "Odisha": "er", "West Bengal": "er", "Jharkhand": "er",
    "Sikkim": "er", "Andaman and Nicobar": "er",
    # North-Eastern Region
    "Assam": "ner", "Manipur": "ner", "Meghalaya": "ner", "Mizoram": "ner",
    "Nagaland": "ner", "Tripura": "ner", "Arunachal Pradesh": "ner",
}

EASTERN_REGION_STATES = frozenset(s for s, r in STATE_REGIONS.items() if r == "er")

# Names used by the PSP / Zenodo ingestion sources that differ from the canonical
# state names above. Only spelling variants; regions always come from STATE_REGIONS.
STATE_NAME_ALIASES = {
    "Delhi": "NCT of Delhi",
    "Andaman and Nicobar Islands": "Andaman and Nicobar",
    "Daman & Diu": "Daman and Diu",
    "Dadra & Nagar Haveli": "Dadra and Nagar Haveli",
    "DNHDDPDCL": "Dadra and Nagar Haveli",
    "Jammu & Kashmir": "Jammu and Kashmir",
}

# Non-state grid entities that appear as rows in PSP reports (not forecastable
# states, so they are not in STATE_REGIONS).
NON_STATE_ENTITY_REGIONS = {
    "Railways_NR ISTS": "nr",
    "AMNSIL": "wr", "BALCO": "wr",
    "Damodar Valley Corporation": "er", "DVC": "er",
}


def grid_region_code(name, default=None):
    """Return the upper-case grid region code (NR/WR/SR/ER/NER) for a state or
    grid-entity name as spelled by the ingestion sources, derived from
    STATE_REGIONS. Returns `default` for unknown names."""
    canonical = STATE_NAME_ALIASES.get(name, name)
    region = STATE_REGIONS.get(canonical) or NON_STATE_ENTITY_REGIONS.get(canonical)
    return region.upper() if region else default

