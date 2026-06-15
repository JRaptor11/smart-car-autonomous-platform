from typing import List, Tuple, Dict, Optional, Any

import csv
import os
import time
from dataclasses import dataclass


@dataclass
class CsvLoggerConfig:
    log_dir: str = "logs"
    filename_prefix: str = "run"
    flush_every: int = 10  # flush every N rows


class CsvLogger:
    """
    CSV logger that can ACCEPT NEW KEYS mid-run.

    How it works:
    - Keeps an internal ordered List of fieldnames.
    - If a new key appears, it will:
        1) close current file
        2) rewrite the CSV with a new header including the new key(s)
        3) backfill previous rows with empty values for those new columns
    This is slower when new keys appear, but it only happens when the schema changes.
    """

    def __init__(self, config: Optional[CsvLoggerConfig] = None):
        self.cfg = config or CsvLoggerConfig()
        os.makedirs(self.cfg.log_dir, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(self.cfg.log_dir, f"{self.cfg.filename_prefix}_{ts}.csv")

        self._fieldnames: List[str] = []
        self._rows: List[Dict[str, Any]] = []  # stored for rewrite-on-schema-change

        self._file = open(self.path, "w", newline="")
        self._writer: Optional[csv.DictWriter] = None
        self._row_count = 0

    def _init_writer_if_needed(self, row: Dict[str, Any]) -> None:
        if self._writer is not None:
            return

        self._fieldnames = list(row.keys())
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        self._writer.writeheader()

    def _rewrite_with_new_fields(self, new_fields: List[str]) -> None:
        # Extend fieldnames preserving order
        for f in new_fields:
            if f not in self._fieldnames:
                self._fieldnames.append(f)

        # Close current file
        self._file.flush()
        self._file.close()

        # Rewrite entire file with updated header
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        self._writer.writeheader()

        # Rewrite stored rows, backfilling missing fields with ""
        for r in self._rows:
            out = {k: r.get(k, "") for k in self._fieldnames}
            self._writer.writerow(out)

        self._file.flush()

    def write(self, row: Dict[str, Any]) -> None:
        self._init_writer_if_needed(row)

        # Detect new keys
        new_keys = [k for k in row.keys() if k not in self._fieldnames]
        if new_keys:
            self._rewrite_with_new_fields(new_keys)

        # Store row for possible future rewrite
        self._rows.append(dict(row))

        # Write row with current fieldnames
        out = {k: row.get(k, "") for k in self._fieldnames}
        assert self._writer is not None
        self._writer.writerow(out)

        self._row_count += 1
        if self._row_count % self.cfg.flush_every == 0:
            self._file.flush()

    def close(self) -> None:
        try:
            self._file.flush()
        finally:
            self._file.close()
