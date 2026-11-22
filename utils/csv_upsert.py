"""
Streaming, atomic, text-preserving CSV upserts.

Existing rows are copied through as the exact strings they already hold (no float
re-formatting, no dtype inference), so rewriting a 150 MB file does not alter any
value. New rows are merged in key order. The target is replaced with ``os.replace``
only after the temp file is fully written, so an error never truncates the original.
"""

import csv
import os
from pathlib import Path
from typing import Callable, Hashable, Iterable, List, Sequence, Set, Tuple

import pandas as pd

Row = List[str]
Key = Tuple[str, ...]


def _fmt(value) -> str:
    return "" if pd.isna(value) else str(value)


def frame_to_rows(df: pd.DataFrame, header: Sequence[str]) -> List[Row]:
    """Rows of ``df`` as strings in ``header`` column order (missing columns -> empty)."""
    if df.empty:
        return []
    aligned = df.reindex(columns=list(header))
    return [[_fmt(v) for v in rec] for rec in aligned.itertuples(index=False, name=None)]


def read_header(path: Path) -> List[str]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def _line_terminator(path: Path) -> str:
    """The file's own line ending (judged by its header line), so a rewrite does not change it."""
    with Path(path).open("rb") as f:
        return "\r\n" if f.readline().endswith(b"\r\n") else "\n"


def _key_fn(header: Sequence[str], key_cols: Sequence[str]) -> Callable[[Row], Key]:
    idx = [header.index(c) for c in key_cols]
    return lambda row: tuple(row[i] for i in idx)


def scan_keys(path: Path, key_cols: Sequence[str]) -> Set[Key]:
    """Set of key tuples present in the file (empty set when the file is absent)."""
    path = Path(path)
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        key = _key_fn(next(reader), key_cols)
        return {key(r) for r in reader if r}


def upsert_csv(
    path: Path,
    new_df: pd.DataFrame,
    key_cols: Sequence[str],
    replace_keys: Iterable[Key] = (),
) -> Tuple[int, int]:
    """
    Merge ``new_df`` into the CSV at ``path``. Returns (rows_added, rows_replaced).

    * A new row whose key is not in the file is added.
    * A new row whose key is in the file is IGNORED (existing real rows are never
      overwritten) unless that key is listed in ``replace_keys``; then every old row
      with that key is dropped and the new row(s) take its place.
    """
    path = Path(path)
    if new_df is None or new_df.empty:
        return 0, 0

    header = read_header(path)
    key = _key_fn(header, key_cols)
    existing = scan_keys(path, key_cols)
    replace = {tuple(k) for k in replace_keys}

    new_rows = [r for r in frame_to_rows(new_df, header) if key(r) not in existing or key(r) in replace]
    if not new_rows:
        return 0, 0
    new_rows.sort(key=key)  # stable: keeps the caller's order within one key
    drop = {key(r) for r in new_rows} & existing & replace

    eol = _line_terminator(path)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    replaced = 0
    try:
        with path.open("r", newline="", encoding="utf-8") as src, \
                tmp.open("w", newline="", encoding="utf-8") as dst:
            reader = csv.reader(src)
            writer = csv.writer(dst, lineterminator=eol)
            writer.writerow(next(reader))
            pending = iter(new_rows)
            nxt = next(pending, None)
            for old in reader:
                if not old:
                    continue
                old_key = key(old)
                if old_key in drop:
                    replaced += 1
                    continue
                while nxt is not None and key(nxt) < old_key:
                    writer.writerow(nxt)
                    nxt = next(pending, None)
                writer.writerow(old)
            while nxt is not None:
                writer.writerow(nxt)
                nxt = next(pending, None)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return len(new_rows), replaced
