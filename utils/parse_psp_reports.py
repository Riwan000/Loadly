"""
Utility to parse Grid-India (formerly POSOCO) Daily Power Supply Position (PSP) reports
from Excel (.xls) and PDF files into structured state-wise demand and generation DataFrames.
"""

import datetime
import logging
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import xlrd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import grid_region_code  # noqa: E402

logger = logging.getLogger(__name__)

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
    "NCT of Delhi": "Delhi",
    "Andaman & Nicobar": "Andaman and Nicobar Islands",
    "Andaman and Nicobar": "Andaman and Nicobar Islands",
    "Daman & Diu": "Daman and Diu",
    "Dadra & Nagar Haveli": "Dadra and Nagar Haveli",
    "DNH&DD": "Dadra and Nagar Haveli",
    "DNHDD": "Dadra and Nagar Haveli",
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

REGION_CODES = ("NR", "WR", "SR", "ER", "NER")
SUPPORTED_EXTS = (".xls", ".xlsx")
DEFAULT_STATE_COLS = {"max_demand": 2, "shortage": 3, "energy_met": 4}
FILENAME_DATE_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{2,4})")
FILENAME_ID_RE = re.compile(r"_PSP_.*?(\d+)(?:\.\w+)?$")
SECTION_BREAK_RE = re.compile(r"^[D-Z]\.\s")
_CASEFOLDED_NAMES = {k.casefold(): v for k, v in STATE_NAME_MAPPING.items()}

Grid = List[list]


def to_float(value) -> float:
    """Convert a spreadsheet cell to float; blank / '-' / text become NaN (never 0)."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text in ("", "-", "--", "NA", "N/A"):
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def normalise_state_name(raw: str) -> str:
    """Map a PSP state label to the canonical column name (unknown labels pass through)."""
    cleaned = re.sub(r"\s+", " ", str(raw)).strip()
    return _CASEFOLDED_NAMES.get(cleaned.casefold(), cleaned)


def _text(grid: Grid, r: int, c: int) -> str:
    if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[r]):
        return ""
    return re.sub(r"\s+", " ", str(grid[r][c])).strip()


def _num(grid: Grid, r: int, c: int) -> float:
    if r < 0 or r >= len(grid) or c >= len(grid[r]):
        return float("nan")
    return to_float(grid[r][c])


def parse_date_from_filename(filepath: str) -> Optional[str]:
    """Data date (YYYY-MM-DD) from a ``DD.MM.YY_NLDC_PSP...`` filename.

    Verified against real data: the date in the filename is the day the figures
    describe (the sheet itself is dated one day later as 'Date of Reporting').
    """
    m = FILENAME_DATE_RE.search(os.path.basename(filepath))
    if not m:
        return None
    d, mth, y = m.groups()
    year = int(y) + 2000 if len(y) == 2 else int(y)
    try:
        return datetime.date(year, int(mth), int(d)).isoformat()
    except ValueError:
        return None


def _date_from_grid(grid: Grid) -> Optional[str]:
    """Fallback: 'Date of Reporting' minus one day (reports cover the previous day)."""
    for r in range(min(8, len(grid))):
        for c in range(len(grid[r])):
            if "date of reporting" not in _text(grid, r, c).casefold():
                continue
            for cc in range(c + 1, len(grid[r])):
                cell = _text(grid, r, cc)
                if not cell:
                    continue
                try:
                    reported = pd.to_datetime(cell, dayfirst=True)
                except (ValueError, TypeError):
                    continue
                if pd.notna(reported):
                    return (reported.date() - datetime.timedelta(days=1)).isoformat()
    return None


def parse_date_from_filename_or_sheet(filepath: str, grid: Optional[Grid] = None) -> Optional[str]:
    """Report data date, preferring the filename and falling back to sheet metadata."""
    return parse_date_from_filename(filepath) or (_date_from_grid(grid) if grid else None)


def _load_grid(filepath: str) -> Optional[Grid]:
    """Load the 'MOP_E' sheet (or the first sheet whose name contains 'MOP') as a list of rows."""
    ext = Path(filepath).suffix.lower()
    if ext == ".xls":
        wb = xlrd.open_workbook(filepath)
        names = wb.sheet_names()
        name = next((n for n in names if n.strip().upper() == "MOP_E"), None) \
            or next((n for n in names if "MOP" in n.upper()), None)
        if name is None:
            return None
        sh = wb.sheet_by_name(name)
        return [[sh.cell_value(r, c) for c in range(sh.ncols)] for r in range(sh.nrows)]
    if ext == ".xlsx":
        import openpyxl
        wb = openpyxl.load_workbook(filepath, data_only=True, read_only=True)
        name = next((n for n in wb.sheetnames if "MOP" in n.upper()), None)
        if name is None:
            return None
        return [["" if v is None else v for v in row] for row in wb[name].iter_rows(values_only=True)]
    return None


def _column_map(grid: Grid, header_row: int) -> Dict[str, int]:
    """Locate demand / shortage / energy columns from the two header rows (first match wins)."""
    found: Dict[str, int] = {}
    ncols = max(len(grid[header_row]), len(grid[header_row - 1]) if header_row else 0)
    for c in range(ncols):
        label = (_text(grid, header_row - 1, c) + " " + _text(grid, header_row, c)).casefold()
        if "max.demand" in label or "max demand" in label:
            found.setdefault("max_demand", c)
        elif "shortage during" in label:
            found.setdefault("shortage", c)
        elif "energy met" in label:
            found.setdefault("energy_met", c)
    return {**DEFAULT_STATE_COLS, **found}


def _parse_state_table(grid: Grid, date_str: str) -> List[dict]:
    header_row = next((r for r in range(len(grid)) if _text(grid, r, 1).casefold() == "states"), None)
    if header_row is None:
        return []

    cols = _column_map(grid, header_row)
    rows: List[dict] = []
    region = "NR"
    for r in range(header_row + 1, len(grid)):
        label0, raw_state = _text(grid, r, 0), _text(grid, r, 1)
        if (SECTION_BREAK_RE.match(label0) or SECTION_BREAK_RE.match(raw_state)
                or "transnational" in raw_state.casefold()):
            break
        if label0 in REGION_CODES:
            region = label0
        if not raw_state or raw_state.casefold().startswith("total"):
            continue
        state = normalise_state_name(raw_state)
        rows.append({
            "date": date_str,
            "Region": grid_region_code(state, default=region),
            "States": state,
            "Max.Demand Met during the day(MW)": _num(grid, r, cols["max_demand"]),
            "Shortage during maximum Demand(MW)": _num(grid, r, cols["shortage"]),
            "Energy Met (MU)": _num(grid, r, cols["energy_met"]),
        })
    return rows


def _parse_all_india_energy_met(grid: Grid) -> float:
    """All-India 'Energy Met (MU)' (the TOTAL column of section A)."""
    for r in range(min(15, len(grid))):
        total_col = next((c for c in range(len(grid[r])) if _text(grid, r, c).upper() == "TOTAL"), None)
        if total_col is None:
            continue
        for rr in range(r + 1, min(r + 12, len(grid))):
            if _text(grid, rr, 0).casefold().startswith("energy met"):
                return _num(grid, rr, total_col)
        return float("nan")
    return float("nan")


def _parse_generation_table(grid: Grid, date_str: str) -> List[dict]:
    start = next((r for r in range(len(grid)) if "sourcewise generation" in _text(grid, r, 0).casefold()), None)
    if start is None or start + 1 >= len(grid):
        return []
    header = start + 1
    col_of: Dict[str, int] = {}
    for c in range(len(grid[header])):
        label = _text(grid, header, c)
        if label in REGION_CODES:
            col_of[label] = c
        elif label.casefold() == "all india":
            col_of["All India"] = c
    if not col_of:
        return []

    rows: List[dict] = []
    for r in range(header + 1, len(grid)):
        source = _text(grid, r, 0)
        if not source or source.casefold().startswith("share"):
            break
        rows.append({"date": date_str, "source": source, **{k: _num(grid, r, c) for k, c in col_of.items()}})
        if source.casefold() == "total":
            break
    return rows


def parse_psp_grid(grid: Grid, date_str: str) -> dict:
    """Parse a PSP sheet (list of rows) into state demand and regional generation records."""
    long_rows = _parse_state_table(grid, date_str)
    wide: Dict[str, object] = {"Date": date_str}
    for row in long_rows:
        wide[row["States"]] = row["Energy Met (MU)"]
    wide["Total"] = _parse_all_india_energy_met(grid)
    return {
        "date": date_str,
        "state_demand_wide": wide,
        "state_demand_long": long_rows,
        "generation_long": _parse_generation_table(grid, date_str),
    }


def parse_psp_xls(filepath: str) -> Optional[dict]:
    """Parse one NLDC PSP report (.xls/.xlsx; .pdf as fallback). Returns None when it cannot be parsed."""
    if Path(filepath).suffix.lower() == ".pdf":
        return parse_psp_pdf(filepath)
    try:
        grid = _load_grid(filepath)
    except Exception as exc:  # corrupt or unreadable workbook
        logger.error("Error opening %s: %s", filepath, exc)
        return None
    if not grid:
        return None
    date_str = parse_date_from_filename_or_sheet(filepath, grid)
    if not date_str:
        logger.warning("Could not determine date for %s", filepath)
        return None
    result = parse_psp_grid(grid, date_str)
    return result if result["state_demand_long"] else None


NUMBER = r"-?\d+(?:\.\d+)?"
PDF_STATE_ROW_RE = re.compile(
    rf"^(?:(?P<region>NR|WR|SR|ER|NER)\s+)?(?P<state>[A-Za-z][^\d]*?)\s+"
    rf"(?P<max_demand>{NUMBER})\s+(?P<shortage>{NUMBER})\s+(?P<energy>{NUMBER})(?:\s+{NUMBER}){{4}}$"
)
PDF_REGION_HEADER_RE = re.compile(r"^NR WR SR ER NER TOTAL$")


def parse_psp_pdf_text(text: str, date_str: str) -> Optional[dict]:
    """
    Parse the text of an NLDC PSP PDF (fallback for dates without a spreadsheet).

    Only the state demand table and the All-India energy met are extracted; the PDF
    text does not keep generation row labels next to their values, so generation
    records are left empty rather than guessed.
    """
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    long_rows: List[dict] = []
    region = "NR"
    in_states = False
    for ln in lines:
        if not in_states:
            in_states = ln.startswith("Region States") or "Shortage (MU)" in ln
            continue
        if ln.startswith("Bhutan"):
            break
        m = PDF_STATE_ROW_RE.match(ln)
        if not m:
            continue
        if m.group("region"):
            region = m.group("region")
        state = normalise_state_name(m.group("state"))
        long_rows.append({
            "date": date_str,
            "Region": grid_region_code(state, default=region),
            "States": state,
            "Max.Demand Met during the day(MW)": float(m.group("max_demand")),
            "Shortage during maximum Demand(MW)": float(m.group("shortage")),
            "Energy Met (MU)": float(m.group("energy")),
        })
    if not long_rows:
        return None

    total = float("nan")
    for i, ln in enumerate(lines):
        if PDF_REGION_HEADER_RE.match(ln) and i + 3 < len(lines):
            nums = lines[i + 3].split()  # demand, shortage, ENERGY MET rows follow the header
            if len(nums) == 6:
                total = to_float(nums[5])
            break

    wide: Dict[str, object] = {"Date": date_str}
    for row in long_rows:
        wide[row["States"]] = row["Energy Met (MU)"]
    wide["Total"] = total
    return {"date": date_str, "state_demand_wide": wide, "state_demand_long": long_rows, "generation_long": []}


def parse_psp_pdf(filepath: str) -> Optional[dict]:
    """Parse one NLDC PSP PDF via pypdf text extraction."""
    from pypdf import PdfReader

    date_str = parse_date_from_filename(filepath)
    if not date_str:
        return None
    try:
        for page in PdfReader(filepath).pages:
            text = page.extract_text() or ""
            if "Shortage (MU)" in text:
                return parse_psp_pdf_text(text, date_str)
    except Exception as exc:  # corrupt PDF
        logger.error("Error reading PDF %s: %s", filepath, exc)
    return None


def _file_rank(path: Path) -> tuple:
    """Sort key for revisions of the same date: highest numeric id, then newest mtime."""
    m = FILENAME_ID_RE.search(path.name)
    return (int(m.group(1)) if m else 0, path.stat().st_mtime)


def select_files_per_date(psp_dir: Path) -> Dict[str, Path]:
    """
    One report per date: a spreadsheet when available (latest revision = highest id),
    otherwise a PDF.
    """
    sheets: Dict[str, List[Path]] = {}
    pdfs: Dict[str, List[Path]] = {}
    for path in Path(psp_dir).iterdir():
        day = parse_date_from_filename(str(path))
        if not day:
            continue
        ext = path.suffix.lower()
        if ext in SUPPORTED_EXTS:
            sheets.setdefault(day, []).append(path)
        elif ext == ".pdf":
            pdfs.setdefault(day, []).append(path)
    chosen = {day: max(paths, key=_file_rank) for day, paths in pdfs.items()}
    chosen.update({day: max(paths, key=_file_rank) for day, paths in sheets.items()})
    return dict(sorted(chosen.items()))


def parse_all_psp_files(psp_dir: Path = PSP_DIR) -> tuple:
    """
    Parse every date available in ``data/psp/`` (spreadsheet preferred, PDF fallback).

    Returns:
        (df_wide_demand, df_long_demand, df_generation)
    """
    files = select_files_per_date(Path(psp_dir))
    logger.info("Found %d PSP reports (one per date) in %s", len(files), psp_dir)

    all_wide, all_long_demand, all_long_gen = [], [], []
    for day, path in files.items():
        res = parse_psp_xls(str(path))
        if res is None:
            logger.warning("Unparseable PSP file for %s: %s", day, path.name)
            continue
        all_wide.append(res["state_demand_wide"])
        all_long_demand.extend(res["state_demand_long"])
        all_long_gen.extend(res["generation_long"])

    return pd.DataFrame(all_wide), pd.DataFrame(all_long_demand), pd.DataFrame(all_long_gen)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df_w, df_l, df_g = parse_all_psp_files()
    print(f"Parsed demand (wide): {df_w.shape}")
    print(df_w.head(3))
