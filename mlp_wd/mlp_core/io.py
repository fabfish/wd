"""CSV I/O with file-locked appends + resume helpers."""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Iterable

import filelock
import pandas as pd

from .runner import CSV_FIELDS, get_run_key


def append_result(record: dict[str, Any], output_file: str | Path) -> None:
    if record is None:
        return
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file = str(output_file) + ".lock"
    with filelock.FileLock(lock_file, timeout=60):
        new_file = not output_file.exists()
        with output_file.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow({k: record.get(k, "") for k in CSV_FIELDS})


def load_completed_keys(output_file: str | Path) -> set[str]:
    output_file = Path(output_file)
    if not output_file.exists():
        return set()
    try:
        df = pd.read_csv(output_file)
    except Exception as exc:
        print(f"[load_completed_keys] failed to read {output_file}: {exc}")
        return set()
    keys: set[str] = set()
    for _, row in df.iterrows():
        keys.add(get_run_key({k: row[k] for k in CSV_FIELDS if k in row.index}))
    return keys
