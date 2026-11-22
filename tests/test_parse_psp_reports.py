"""Tests for the PSP report parser and downloader selection logic (no network, no binaries)."""

import datetime as dt
import math
from pathlib import Path

import pytest

from utils import fetch_psp_reports as fetch
from utils import parse_psp_reports as psp


def _build_grid(extra_state_rows=None):
    """Mimic the layout of a real MOP_E sheet (section A, C and G)."""
    grid = [[""] * 9 for _ in range(3)]
    grid[2][6], grid[2][8] = "Date of Reporting:", "29-Apr-2024"
    grid += [
        ["A. Power Supply Position at All India and Regional level", "", "", "", "", "", "", "", ""],
        ["", "", "NR", "WR", "SR", "ER", "NER", "TOTAL", ""],
        ["Demand Met during Evening Peak hrs(MW)", "", 1, 2, 3, 4, 5, 15, ""],
        ["Energy Met (MU)", "", 1332.0, 1553.3, 1300.55, 637.79, 50.13, 4873.77, ""],
        ["C. Power Supply Position in States", "", "", "", "", "", "", "", ""],
        ["", "", "Max.Demand", "Shortage during", "Energy Met", "Drawal", "OD(+)/UD(-)", "Max OD", "Energy"],
        ["Region", "States", "Met during the day (MW)", "maximum Demand (MW)", "(MU)", "Schedule", "(MU)", "(MW)", "Shortage (MU)"],
        ["", "Punjab", 9296.0, 0.0, 157.85, 50.85, -1.02, 390.0, 0.0],
        ["", "Delhi", 4984.0, 0.0, 100.13, 82.4, -0.44, 83.0, 0.0],
        ["NR", "UP", 24603.0, 0.0, 492.16, 221.42, -0.36, 382.0, 0.0],
        ["", "J&K(UT) & Ladakh(UT)", 2559.0, 0.0, 49.15, 30.25, -0.99, 39.0, 0.36],
        ["", "Railways_NR ISTS", 180.0, 0.0, 3.89, 3.6, 0.29, 62.0, 0.0],
        ["", "DNHDDPDCL", 1237.0, 0.0, 28.5, 27.7, 0.8, 199.0, 0.0],
        ["WR", "Sikkim", "-", "-", "-", 1.27, -0.15, 22.0, 0.0],
    ]
    grid += extra_state_rows or []
    grid += [
        ["", "", "", "", "", "", "", "", ""],
        ["D. Transnational Exchanges (MU)", "", "", "", "", "", "", "", ""],
        ["", "Bhutan", 1.0, 2.0, 3.0, 4.0, "", "", ""],
        ["G. Sourcewise generation (Gross) (MU)", "", "", "", "", "", "", "", ""],
        ["", "", "NR", "WR", "SR", "ER", "NER", "All India", "% Share"],
        ["Coal", "", 839.68, 1621.06, 762.6, 707.63, 15.15, 3946.12, 74.6],
        ["Wind", "", 7.84, "-", 38.36, "-", "-", 46.2, 0.9],
        ["Total", "", 1308.8, 2018.27, 1172.4, 738.28, 48.51, 5286.26, 100.0],
        ["Share of RES in total generation (%)", "", 13.7, 9.8, 17.2, 0.5, 1.9, 11.0, ""],
    ]
    return grid


@pytest.fixture
def parsed():
    return psp.parse_psp_grid(_build_grid(), "2024-04-28")


def test_state_names_map_to_canonical_columns(parsed):
    wide = parsed["state_demand_wide"]
    assert wide["Delhi"] == 100.13
    assert wide["Uttar Pradesh"] == 492.16
    assert wide["Jammu and Kashmir"] == 49.15
    assert wide["Dadra and Nagar Haveli"] == 28.5


def test_all_india_total_comes_from_section_a(parsed):
    assert parsed["state_demand_wide"]["Total"] == 4873.77


def test_dash_values_are_nan_not_zero(parsed):
    assert math.isnan(parsed["state_demand_wide"]["Sikkim"])


def test_stops_at_transnational_section(parsed):
    names = [r["States"] for r in parsed["state_demand_long"]]
    assert "Bhutan" not in names
    assert names[0] == "Punjab" and names[-1] == "Sikkim"


def test_region_taken_from_mapping_then_table_marker(parsed):
    regions = {r["States"]: r["Region"] for r in parsed["state_demand_long"]}
    assert regions["Delhi"] == "NR"
    assert regions["Dadra and Nagar Haveli"] == "WR"


def test_max_demand_and_shortage_columns_are_located(parsed):
    punjab = parsed["state_demand_long"][0]
    assert punjab["Max.Demand Met during the day(MW)"] == 9296.0
    assert punjab["Shortage during maximum Demand(MW)"] == 0.0


def test_generation_table_includes_total_and_nan_for_dash(parsed):
    gen = {r["source"]: r for r in parsed["generation_long"]}
    assert gen["Total"]["All India"] == 5286.26
    assert math.isnan(gen["Wind"]["WR"])
    assert "Share of RES in total generation (%)" not in gen


def test_unknown_label_passes_through_unmapped():
    assert psp.normalise_state_name("  New   Utility ") == "New Utility"
    assert psp.normalise_state_name("delhi") == "Delhi"


def test_column_order_changes_are_tolerated():
    grid = _build_grid()
    # Swap the Energy Met and Max.Demand columns (header and data) in the state table.
    start = next(i for i, r in enumerate(grid) if r[1] == "States")
    for r in range(start - 1, len(grid)):
        if grid[r][1] in ("Bhutan",) or str(grid[r][0]).startswith("D."):
            break
        grid[r][2], grid[r][4] = grid[r][4], grid[r][2]
    res = psp.parse_psp_grid(grid, "2024-04-28")
    assert res["state_demand_wide"]["Punjab"] == 157.85


def test_date_from_filename_is_data_date():
    assert psp.parse_date_from_filename("x/28.04.24_NLDC_PSP_24829.xls") == "2024-04-28"
    assert psp.parse_date_from_filename("07.08.25_NLDC_PSP_H&E%20v1_304.xls") == "2025-08-07"
    assert psp.parse_date_from_filename("nodate.xls") is None


def test_date_falls_back_to_reporting_date_minus_one():
    assert psp._date_from_grid(_build_grid()) == "2024-04-28"


def test_latest_revision_wins_per_date(tmp_path: Path):
    for name in ("01.05.24_NLDC_PSP_100.xls", "01.05.24_NLDC_PSP_250.xls", "01.05.24_NLDC_PSP_9.pdf"):
        (tmp_path / name).write_bytes(b"x")
    chosen = psp.select_files_per_date(tmp_path)
    assert chosen["2024-05-01"].name == "01.05.24_NLDC_PSP_250.xls"


# --------------------------- downloader ---------------------------

def _item(day, ext, item_id):
    return {"Id": item_id, "Field2": day, "FilePath": f"files/grdw/x/{day}.{ext}"}


def test_fetch_prefers_spreadsheet_and_latest_id():
    items = [_item("2024-05-01", "pdf", 5), _item("2024-05-01", "xls", 3), _item("2024-05-01", "xls", 4)]
    chosen = fetch.select_best_files(items, dt.date(2024, 5, 1), dt.date(2024, 5, 31))
    assert chosen[dt.date(2024, 5, 1)]["Id"] == 4


def test_fetch_falls_back_to_pdf_and_respects_range():
    items = [_item("2024-05-02", "pdf", 7), _item("2024-06-01", "xls", 8)]
    chosen = fetch.select_best_files(items, dt.date(2024, 5, 1), dt.date(2024, 5, 31))
    assert list(chosen) == [dt.date(2024, 5, 2)]


def test_month_keys_use_indian_fiscal_year():
    keys = fetch.month_keys(dt.date(2025, 2, 10), dt.date(2025, 5, 3))
    assert keys == [("2024-25", "02"), ("2024-25", "03"), ("2025-26", "04"), ("2025-26", "05")]


def test_rank_candidates_orders_sheets_before_pdfs():
    items = [_item("2024-05-01", "pdf", 9), _item("2024-05-01", "xls", 3), _item("2024-05-01", "xlsx", 4)]
    ranked = fetch.rank_candidates(items, dt.date(2024, 5, 1), dt.date(2024, 5, 1))
    assert [c["Id"] for c in ranked[dt.date(2024, 5, 1)]] == [4, 3, 9]


def test_fetch_item_date_uses_file_name_not_field2():
    # Real API quirk: report titled 29.05.25 carries Field2 = 2025-05-30.
    item = {"Id": 1, "Title_": "29.05.25_NLDC_PSP", "Field2": "2025-05-30",
            "FilePath": "files/x/29.05.25_NLDC_PSP_576.xls"}
    assert fetch.item_date(item) == dt.date(2025, 5, 29)
    assert fetch.item_date({"Field2": "2025-05-30", "FilePath": "x/weird.xls"}) == dt.date(2025, 5, 30)


PDF_TEXT = """11-Nov-2025
NR WR SR ER NER TOTAL
52918 59165 49368 21957 2823 186231
0 0 0 0 0 0
1080 1316 1096 446 53 3991
Max.Demand Shortage during Energy Met Drawal OD(+)/UD(-) Max OD Energy
Region States Met during the
day (MW)
maximum
Demand (MW) (MU) Schedule
(MU) (MU) (MW) Shortage (MU)
Punjab 7277 0 137.3 64.1 -0.7 728 0.00
NR UP 16377 0 307.3 84.0 0.4 269 0.00
J&K(UT) & Ladakh(UT) 2978 0 56.9 50.6 0.3 154 0.00
WR Maharashtra 25105 0 507.4 176.1 -0.3 496 0.00
DNHDDPDCL 1332 0 29.7 29.6 0.1 38 0.00
Bhutan Nepal Bangladesh
12.6 21.2 -24.4
"""


def test_pdf_fallback_parses_state_table_and_total():
    res = psp.parse_psp_pdf_text(PDF_TEXT, "2025-11-10")
    wide = res["state_demand_wide"]
    assert wide["Total"] == 3991.0
    assert wide["Uttar Pradesh"] == 307.3
    assert wide["Jammu and Kashmir"] == 56.9
    assert wide["Dadra and Nagar Haveli"] == 29.7
    regions = {r["States"]: r["Region"] for r in res["state_demand_long"]}
    assert regions["Maharashtra"] == "WR"
    assert res["generation_long"] == []


def test_pdf_only_used_when_no_spreadsheet(tmp_path: Path):
    for name in ("01.05.24_NLDC_PSP_5.xls", "01.05.24_NLDC_PSP_9.pdf", "02.05.24_NLDC_PSP_7.pdf"):
        (tmp_path / name).write_bytes(b"x")
    chosen = psp.select_files_per_date(tmp_path)
    assert chosen["2024-05-01"].suffix == ".xls"
    assert chosen["2024-05-02"].suffix == ".pdf"


def test_region_labels_derive_from_canonical_config():
    from config import STATE_REGIONS, grid_region_code
    for state, region in STATE_REGIONS.items():
        assert grid_region_code(state) == region.upper()
    assert grid_region_code("Andaman and Nicobar Islands") == "ER"
    assert grid_region_code("Delhi") == "NR"
    assert grid_region_code("DVC") == "ER"
    assert grid_region_code("Nowhere", default="XX") == "XX"


def test_no_duplicate_region_maps_in_ingestion_modules():
    from utils import ingest_2023_2024_demand as ing
    assert not hasattr(ing, "STATE_TO_REGION")
    assert not hasattr(psp, "STATE_TO_REGION")
