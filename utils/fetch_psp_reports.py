"""
Download real Grid-India (NLDC) Daily PSP reports into ``data/psp/``.

Public, unauthenticated Grid-India web API:
  * ``POST /api/v1/file`` lists the files of one fiscal year + month.
  * ``https://webcdn.grid-india.in/<FilePath>`` serves the file.

Spreadsheets (.xls/.xlsx) are preferred; a PDF is only downloaded for dates that
have no spreadsheet. When a date has several files (revisions) the one with the
highest API ``Id`` (= most recently uploaded) is chosen.

Files are saved as ``DD.MM.YY_NLDC_PSP_<Id>.<ext>`` which matches the glob used by
``utils/parse_psp_reports.py``. Dates that already have a file of the wanted kind
are skipped, so the script is safe to re-run. Dates with no report are logged and
written to ``data/psp/missing_dates.txt`` -- nothing is ever fabricated.

Usage:
    python utils/fetch_psp_reports.py --start 2024-04-29 --end 2026-11-20
"""

import argparse
import datetime as dt
import logging
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
PSP_DIR = ROOT_DIR / "data" / "psp"

API_FILES_URL = "https://webapi.grid-india.in/api/v1/file"
CDN_BASE_URL = "https://webcdn.grid-india.in/"
USER_AGENT = "Mozilla/5.0 (compatible; psp-report-fetcher/1.0)"

REQUEST_TIMEOUT_S = 60
REQUEST_DELAY_S = 0.3
MAX_RETRIES = 4
BACKOFF_BASE_S = 2.0
MIN_VALID_FILE_BYTES = 10_000

SPREADSHEET_EXTS = ("xls", "xlsx")
PDF_EXT = "pdf"
ALLOWED_EXTS = SPREADSHEET_EXTS + (PDF_EXT,)

# Every real listing record has a numeric Id, an ISO Field2 and a FilePath under
# files/grdw/ (e.g. files/grdw/2026/11/20.11.26_NLDC_PSP_523.xls or
# files/grdw/uploads/daily-reports/psp-reports/2024-2025/31.05.24_NLDC_PSP.xls).
# Path segments may not start with '.', so '..' traversal is impossible.
MAX_ID_DIGITS = 12
ITEM_ID_RE = re.compile(rf"[0-9]{{1,{MAX_ID_DIGITS}}}")      # all three used with fullmatch
ITEM_FIELD2_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_SEGMENT = r"[A-Za-z0-9_-][A-Za-z0-9._-]*"
ITEM_FILEPATH_RE = re.compile(rf"files/grdw/(?:{_SEGMENT}/)*{_SEGMENT}\.(?:xls|xlsx|pdf)", re.IGNORECASE)

logger = logging.getLogger("fetch_psp_reports")


def fiscal_year_label(day: dt.date) -> str:
    """Indian fiscal year label for a date, e.g. 2024-05-31 -> '2024-25'."""
    start = day.year if day.month >= 4 else day.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def month_keys(start: dt.date, end: dt.date) -> List[Tuple[str, str]]:
    """(fiscal_year, 'MM') pairs covering [start, end]."""
    keys: List[Tuple[str, str]] = []
    cur = start.replace(day=1)
    while cur <= end:
        keys.append((fiscal_year_label(cur), f"{cur.month:02d}"))
        cur = (cur.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    return keys


class UpstreamError(RuntimeError):
    """An upstream HTTP service failed or answered unusably (message may contain the URL)."""


def request_with_retry(session: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
    """HTTP request with exponential backoff on network errors / 5xx / 429."""
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.request(method, url, timeout=REQUEST_TIMEOUT_S, **kwargs)
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                # Permanent client error (e.g. listed file missing on the CDN): do not retry.
                raise UpstreamError(f"HTTP {resp.status_code} for {url}")
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            wait = BACKOFF_BASE_S * (2 ** attempt)
            logger.warning("%s %s failed (%s); retry %d/%d in %.0fs", method, url, exc, attempt + 1, MAX_RETRIES, wait)
            time.sleep(wait)
    raise UpstreamError(f"giving up on {method} {url}: {last_exc}")


_request_with_retry = request_with_retry  # backwards-compatible alias


def list_month_files(session: requests.Session, fiscal_year: str, month: str) -> List[dict]:
    """List PSP file records for one fiscal-year month."""
    payload = {
        "_source": "GRDW",
        "_type": "DAILY_PSP_REPORT",
        "_fileDate": fiscal_year,
        "_month": month,
    }
    resp = request_with_retry(session, "POST", API_FILES_URL, json=payload)
    return resp.json().get("retData") or []


FILE_DATE_RE = re.compile(r"^(\d{2})\.(\d{2})\.(\d{2,4})")


def item_date(item: dict) -> Optional[dt.date]:
    """
    Data date of an API file record.

    The date in the file name / title (``DD.MM.YY_NLDC_PSP``) is the day the report
    describes. The API's ``Field2`` is NOT reliable for this: for some uploads it is
    the following day, which makes two different reports collide on one date.
    """
    for text in (Path(item.get("FilePath") or "").name, (item.get("Title_") or "").strip()):
        m = FILE_DATE_RE.match(text)
        if m:
            d, mth, y = m.groups()
            year = int(y) + 2000 if len(y) == 2 else int(y)
            try:
                return dt.date(year, int(mth), int(d))
            except ValueError:
                continue
    for key, fmt in (("Field2", "%Y-%m-%d"), ("Field1", "%d-%m-%Y")):
        raw = (item.get(key) or "").strip()
        if raw:
            try:
                return dt.datetime.strptime(raw[:10], fmt).date()
            except ValueError:
                continue
    return None


def validate_item(item) -> Optional[dict]:
    """
    Normalised copy of an API file record (``Id`` as int), or None when the record is
    malformed: Id not all digits, Field2 not an ISO date, or FilePath outside files/grdw/
    or without a whitelisted extension. Invalid records are logged and skipped.
    """
    if not isinstance(item, dict):
        logger.warning("Skipping malformed PSP listing record (not an object)")
        return None
    raw_id = item.get("Id")
    id_text = str(raw_id) if isinstance(raw_id, (int, str)) and not isinstance(raw_id, bool) else ""
    field2 = item.get("Field2")
    file_path = item.get("FilePath")
    problems = []
    if not ITEM_ID_RE.fullmatch(id_text) or int(id_text) <= 0:
        problems.append("Id")
    if not isinstance(field2, str) or not ITEM_FIELD2_RE.fullmatch(field2.strip()):
        problems.append("Field2")
    else:
        try:
            dt.date.fromisoformat(field2.strip())
        except ValueError:
            problems.append("Field2")
    if not isinstance(file_path, str) or not ITEM_FILEPATH_RE.fullmatch(file_path):
        problems.append("FilePath")
    if problems:
        logger.warning("Skipping malformed PSP listing record (invalid %s): %r",
                       ", ".join(problems), {k: item.get(k) for k in ("Id", "Field2", "FilePath")})
        return None
    return {**item, "Id": int(id_text), "Field2": field2.strip()}


def item_ext(item: dict) -> str:
    """Lower-case file extension of an API file record."""
    return Path(item.get("FilePath") or "").suffix.lstrip(".").lower()


def rank_candidates(items: Iterable[dict], start: dt.date, end: dt.date) -> Dict[dt.date, List[dict]]:
    """
    Ordered download candidates per date: spreadsheets first (latest ``Id`` first),
    then PDFs (latest ``Id`` first). The caller tries them in order.
    """
    by_date: Dict[dt.date, List[dict]] = {}
    for raw in items:
        item = validate_item(raw)
        if item is None:
            continue
        day = item_date(item)
        if day is None or not (start <= day <= end) or not item.get("FilePath"):
            continue
        by_date.setdefault(day, []).append(item)

    def newest_first(cands: List[dict]) -> List[dict]:
        return sorted(cands, key=lambda c: c["Id"], reverse=True)

    ranked: Dict[dt.date, List[dict]] = {}
    for day, cands in by_date.items():
        ordered = (newest_first([c for c in cands if item_ext(c) in SPREADSHEET_EXTS])
                   + newest_first([c for c in cands if item_ext(c) == PDF_EXT]))
        if ordered:
            ranked[day] = ordered
    return ranked


def select_best_files(items: Iterable[dict], start: dt.date, end: dt.date) -> Dict[dt.date, dict]:
    """Single best candidate per date (see ``rank_candidates``)."""
    return {day: cands[0] for day, cands in rank_candidates(items, start, end).items()}


def local_filename(day: dt.date, item: dict) -> str:
    """``DD.MM.YY_NLDC_PSP_<Id>.<ext>`` built only from validated parts (raises ValueError otherwise)."""
    item_id, ext = str(item.get("Id")), item_ext(item)
    if not ITEM_ID_RE.fullmatch(item_id) or ext not in ALLOWED_EXTS:
        raise ValueError("refusing to build a file name from an unvalidated listing record")
    return f"{day.strftime('%d.%m.%y')}_NLDC_PSP_{int(item_id)}.{ext}"


def safe_destination(psp_dir: Path, name: str) -> Path:
    """``psp_dir / name``, asserting the resolved path stays directly inside ``psp_dir``."""
    base = Path(psp_dir).resolve()
    dest = (base / name).resolve()
    if dest.parent != base:
        raise ValueError("download destination escapes the PSP directory")
    return dest


def already_present(psp_dir: Path, day: dt.date, ext: str) -> bool:
    """True when a file of the wanted kind already exists locally for the date."""
    prefix = f"{day.strftime('%d.%m.%y')}_NLDC_PSP"
    exts = SPREADSHEET_EXTS if ext in SPREADSHEET_EXTS else (ext,)
    return any(p.stat().st_size >= MIN_VALID_FILE_BYTES
               for e in exts for p in psp_dir.glob(f"{prefix}*.{e}"))


def download_file(session: requests.Session, item: dict, dest: Path) -> None:
    """Download one file atomically (write to .part then rename)."""
    if validate_item(item) is None:
        raise ValueError("refusing to download an unvalidated listing record")
    url = CDN_BASE_URL + item["FilePath"]
    resp = request_with_retry(session, "GET", url)
    if len(resp.content) < MIN_VALID_FILE_BYTES:
        raise UpstreamError(f"suspiciously small download ({len(resp.content)} bytes) from {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(resp.content)
    tmp.replace(dest)


def make_session() -> requests.Session:
    """HTTP session with the generic client User-Agent."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def collect_items(session: requests.Session, start: dt.date, end: dt.date) -> List[dict]:
    """List every API file record for the fiscal-year months touching [start, end] (+-3 days)."""
    items: List[dict] = []
    for fiscal_year, month in month_keys(start - dt.timedelta(days=3), end + dt.timedelta(days=3)):
        try:
            items.extend(list_month_files(session, fiscal_year, month))
        except RuntimeError as exc:
            logger.error("Could not list %s/%s: %s", fiscal_year, month, exc)
        time.sleep(REQUEST_DELAY_S)
    return items


def download_best(
    session: requests.Session, day: dt.date, cands: List[dict], psp_dir: Path
) -> Tuple[Optional[str], Optional[str]]:
    """
    Download the first candidate that works (spreadsheet first, PDF only if every
    spreadsheet fails). Returns (saved file name, None) or (None, last error text).
    """
    last_error: Optional[str] = None
    for item in cands:
        try:
            name = local_filename(day, item)
            download_file(session, item, safe_destination(psp_dir, name))
            logger.info("Downloaded %s", name)
            return name, None
        except (RuntimeError, ValueError) as exc:
            last_error = str(exc)
            logger.error("Download failed for %s (%s): %s", day, item.get("FilePath"), exc)
        time.sleep(REQUEST_DELAY_S)
    return None, last_error


def fetch_range(start: dt.date, end: dt.date, psp_dir: Path = PSP_DIR) -> Dict[str, List[dt.date]]:
    """Download reports for [start, end]. Returns {'downloaded','skipped','missing','failed'} date lists."""
    psp_dir.mkdir(parents=True, exist_ok=True)
    session = make_session()
    ranked = rank_candidates(collect_items(session, start, end), start, end)
    result: Dict[str, List[dt.date]] = {"downloaded": [], "skipped": [], "missing": [], "failed": []}

    day = start
    while day <= end:
        cands = ranked.get(day)
        if not cands:
            result["missing"].append(day)
        elif any(already_present(psp_dir, day, item_ext(c)) for c in cands):
            result["skipped"].append(day)
        else:
            ok = download_best(session, day, cands, psp_dir)[0] is not None
            result["downloaded" if ok else "failed"].append(day)
            time.sleep(REQUEST_DELAY_S)
        day += dt.timedelta(days=1)

    missing_file = psp_dir / "missing_dates.txt"
    missing_file.write_text("\n".join(d.isoformat() for d in result["missing"] + result["failed"]) + "\n")
    return result


def _parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def main(argv: Optional[List[str]] = None) -> int:
    yesterday = dt.date.today() - dt.timedelta(days=1)
    parser = argparse.ArgumentParser(description="Download Grid-India daily PSP reports.")
    parser.add_argument("--start", type=_parse_date, default=dt.date(2024, 4, 29))
    parser.add_argument("--end", type=_parse_date, default=yesterday)
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("--start must not be after --end")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = fetch_range(args.start, args.end)
    logger.info(
        "Done: %d downloaded, %d already present, %d no report, %d failed",
        len(res["downloaded"]), len(res["skipped"]), len(res["missing"]), len(res["failed"]),
    )
    if res["missing"]:
        logger.warning("Dates with no report on Grid-India: %s", ", ".join(d.isoformat() for d in res["missing"]))
    if res["failed"]:
        logger.warning("Dates that failed to download (re-run to retry): %s", ", ".join(d.isoformat() for d in res["failed"]))
    return 1 if res["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
